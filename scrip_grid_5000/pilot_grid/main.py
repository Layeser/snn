import time
import json
import os
import tarfile
import shutil
import glob
import argparse
import paramiko
from ssh_utils import *
from config import load_config

# La configuration (specifique a la personne / installation) est chargee au
# demarrage dans main() puis exposee via cette variable de module.
CFG = None

# Ligne imprimee par start_run.sh a la fin d'un run reussi. Doit rester
# STRICTEMENT identique a celle de scrip_grid_5000/start_run.sh.
SENTINELLE_FIN = "=== EXPERIENCE TERMINEE AVEC SUCCES ==="

# =============================================================
# 1. ANALYSE ET SÉCURITÉ DES CHEMINS
# =============================================================

def trouver_chemin_output_distant(ssh_client, fichier_distant):
    cmd_cat = f"cat {fichier_distant}"
    stdin, stdout, stderr = ssh_client.exec_command(cmd_cat)
    contenu = stdout.read().decode('utf-8')
    
    run_name = None
    output_dir_template = None
    
    for ligne in contenu.splitlines():
        ligne = ligne.strip()
        if ligne.startswith("RUN_NAME="):
            run_name = ligne.split("=", 1)[1].strip('"\'')
        elif "OUTPUT_DIR=" in ligne and not ligne.startswith("#"):
            output_dir_template = ligne.split("=", 1)[1].strip('"\'')

    if not run_name or not output_dir_template:
        return None, None
        
    chemin_avec_home = output_dir_template.replace("$RUN_NAME", run_name).replace("${RUN_NAME}", run_name)
    
    cmd_echo = f'echo "{chemin_avec_home}"'
    stdin, stdout, stderr = ssh_client.exec_command(cmd_echo)
    chemin_absolu_distant = stdout.read().decode('utf-8').strip()

    legacy = f"/home/{CFG.user}/snn"
    attendu = CFG.remote_project_abs()
    if chemin_absolu_distant.startswith(legacy + "/") or chemin_absolu_distant == legacy:
        corrige = chemin_absolu_distant.replace(legacy, attendu, 1)
        print(
            f"[Orchestrateur] OUTPUT_DIR obsolete dans le script : "
            f"{chemin_absolu_distant} -> {corrige}"
        )
        chemin_absolu_distant = corrige

    return chemin_absolu_distant, run_name


def _parse_oar_option_line(ligne: str) -> tuple[str, str] | None:
    """Extrait flag/valeur d'une ligne '# OAR_option ...' ou 'OAR_option ...'."""
    ligne = ligne.strip()
    for prefix in ("# OAR_option", "OAR_option"):
        if ligne.startswith(prefix):
            contenu = ligne[len(prefix) :].strip()
            parts = contenu.split(maxsplit=1)
            if len(parts) == 2:
                return parts[0], parts[1]
    return None


def extraire_options_oar(ssh_client, chemin_distant_script):
    options = {}
    try:
        stdin, stdout, stderr = ssh_client.exec_command(f"cat {chemin_distant_script}")
        for ligne in stdout:
            parsed = _parse_oar_option_line(ligne)
            if parsed:
                flag, valeur = parsed
                options[flag] = valeur
    except Exception as e:
        print(f"Erreur lors de la lecture distante des options OAR : {e}")
    return options


def _pilot_site_depuis_script(chemin_local: str) -> str | None:
    """Lit '# Pilot_site lille|lyon' dans un script d'experience."""
    try:
        with open(chemin_local, "r") as f:
            for ligne in f:
                ligne = ligne.strip()
                if ligne.startswith("# Pilot_site"):
                    parts = ligne.split()
                    if len(parts) >= 2:
                        return parts[1].lower()
    except OSError:
        pass
    return None


def _eligible_pour_site(chemin_local: str, site: str) -> bool:
    """Un script sans Pilot_site est attribue au premier site de config.yaml."""
    cible = _pilot_site_depuis_script(chemin_local)
    if cible is None:
        return site == CFG.sites[0]
    return cible == site.lower()


def generer_commande_soumission(ssh_client, chemin_distant_sh):
    options_script = extraire_options_oar(ssh_client, chemin_distant_sh)

    # Priorite : config.yaml > # OAR_option du script > valeurs par defaut.
    base_l = CFG.oar_resources or options_script.get("-l", "host=1/gpu=1")
    ressources = f"{base_l},walltime={CFG.walltime}"

    arguments_oar = [f'-l "{ressources}"']

    # -t du script (ex. exotic) prime sur oar_type du config (ex. night).
    oar_type = options_script.get("-t") or CFG.oar_type
    if oar_type:
        arguments_oar.append(f'-t "{oar_type}"')

    queue = CFG.oar_queue or options_script.get("-q")
    if queue:
        arguments_oar.append(f'-q "{queue}"')

    for flag, valeur in options_script.items():
        if flag in ("-l", "-q", "-t"):
            continue
        arguments_oar.append(f'{flag} "{valeur}"')

    options_string = " ".join(arguments_oar)
    script_start = CFG.remote_start_script_path()
    
    return f'oarsub {options_string} "{script_start} {chemin_distant_sh}"'


# =============================================================
# 2. GESTION DE L'ÉTAT LOCAL (JSON ÉVOLUÉ)
# =============================================================

def _normaliser_chemin_distant(chemin: str | None) -> str | None:
    """Corrige les chemins distants obsoletes (ex: ~/snn -> ~/internship/snn)."""
    if not chemin:
        return chemin

    attendu = CFG.remote_project_abs()
    legacy = f"/home/{CFG.user}/snn"
    if chemin.startswith(legacy + "/") or chemin == legacy:
        corrige = chemin.replace(legacy, attendu, 1)
        print(f"[Orchestrateur] Chemin distant corrige : {chemin} -> {corrige}")
        return corrige
    return chemin


def _migrer_etats_obsoletes(etats: dict) -> dict:
    """Reecrit les chemins distants legacy dans run_status.json."""
    modifie = False
    for info in etats.values():
        chemin = info.get("chemin_distant")
        corrige = _normaliser_chemin_distant(chemin)
        if corrige != chemin:
            info["chemin_distant"] = corrige
            modifie = True
    if modifie:
        os.makedirs(os.path.dirname(CFG.state_file), exist_ok=True)
        with open(CFG.state_file, "w") as f:
            json.dump(etats, f, indent=4)
    return etats


def charger_tous_les_etats():
    if not os.path.exists(CFG.state_file):
        return {}
    try:
        with open(CFG.state_file, "r") as f:
            contenu = f.read().strip()
        if not contenu:
            return {}
        etats = json.loads(contenu)
        if not isinstance(etats, dict):
            print(f"[Orchestrateur] {CFG.state_file} invalide (pas un objet JSON) — reinitialise.")
            return {}
        return _migrer_etats_obsoletes(etats)
    except json.JSONDecodeError:
        print(f"[Orchestrateur] {CFG.state_file} illisible — reinitialise (utilisez '{{}}').")
        return {}

def sauvegarder_etat_fichier(nom_fichier, etape, statut_oar, job_id, site, chemin_local, chemin_distant):
    tous_les_etats = charger_tous_les_etats()
    tous_les_etats[nom_fichier] = {
        "etape": etape,
        "statut_oar": statut_oar,
        "job_id": job_id,
        "site": site,
        "chemin_local": chemin_local,
        "chemin_distant": chemin_distant,
        "derniere_verification": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    os.makedirs(os.path.dirname(CFG.state_file), exist_ok=True)
    with open(CFG.state_file, 'w') as f:
        json.dump(tous_les_etats, f, indent=4)


def obtenir_etat_specifique(nom_fichier):
    tous_les_etats = charger_tous_les_etats()
    if nom_fichier in tous_les_etats:
        return tous_les_etats[nom_fichier]
    return {"etape": "AUCUNE", "statut_oar": "AUCUN", "job_id": None, "site": None, "chemin_local": None, "chemin_distant": None}


def decompte_jobs_actifs_site(tous_les_etats, site):
    """ Compte combien de scripts sont actuellement en cours sur un site donné """
    return sum(1 for info in tous_les_etats.values() if info.get("site") == site and info.get("etape") in ["ETAPE_1", "ETAPE_2", "ETAPE_3"])


# =============================================================
# 3. SUIVI OAR ET SYNC
# =============================================================
def verifier_si_travail_fini(ssh_client, job_id, chemin_distant):
    fichier_stderr = f"{chemin_distant}/OAR.{job_id}.stderr"
    fichier_stdout = f"{chemin_distant}/OAR.{job_id}.stdout"
    
    stdin, stdout, stderr = ssh_client.exec_command(f"tail -n 10 {fichier_stderr} 2>/dev/null")
    if f"Job {job_id} KILLED" in stdout.read().decode('utf-8'):
        return "KILLED"

    stdin, stdout, stderr = ssh_client.exec_command(f"tail -n 20 {fichier_stdout} 2>/dev/null")
    if SENTINELLE_FIN in stdout.read().decode('utf-8'):
        return "FINI"

    return "ERREUR"


def obtenir_statut_oar(ssh_client, user, job_id):
    stdin, stdout, stderr = ssh_client.exec_command(f"oarstat -u {user}")
    lignes = stdout.read().decode('utf-8').splitlines()
    for ligne in lignes:
        if str(job_id) in ligne:
            elements = ligne.split()
            if len(elements) >= 5:
                return elements[-2]
    return "ABSENT"


def make_job(ssh_client, commande_soumission, site):
    print(f"Soumission du job sur {site} avec la commande : {commande_soumission}")
    stdin, stdout, stderr = ssh_client.exec_command(commande_soumission)
    code = stdout.channel.recv_exit_status()
    sortie = stdout.read().decode("utf-8").strip()
    erreur = stderr.read().decode("utf-8").strip()

    for flux in (sortie, erreur):
        for ligne in flux.splitlines():
            if "OAR_JOB_ID=" in ligne:
                job_id = ligne.split("=", 1)[1].strip()
                print(f"-> Job lance avec succes. ID : {job_id}")
                return job_id

    print("[oarsub] Echec de soumission (aucun OAR_JOB_ID recu).")
    if sortie:
        print(f"[oarsub stdout] {sortie}")
    if erreur:
        print(f"[oarsub stderr] {erreur}")
    if code != 0:
        print(f"[oarsub] code retour : {code}")
    return None


# =============================================================
# 4. LES ÉTAPES DU PIPELINE (ADAPTÉES)
# =============================================================

def _executer_et_journaliser(ssh_client, commande, titre):
    """Execute une commande distante, affiche sa sortie et renvoie le code retour."""
    print(f"[git] {titre}...")
    stdin, stdout, stderr = ssh_client.exec_command(commande)
    code = stdout.channel.recv_exit_status()
    sortie = stdout.read().decode('utf-8').strip()
    erreur = stderr.read().decode('utf-8').strip()
    if sortie:
        print(sortie)
    if erreur:
        print(erreur)
    return code


def synchroniser_git(ssh_client):
    """Option A : met a jour le code du projet SUR LE FRONTEND avant le oarsub.

    - clone automatiquement le repo s'il est absent (si git_repo est fourni) ;
    - se place sur la branche configuree puis fait un pull en fast-forward.
    Renvoie True si le code est a jour, False sinon (le lancement est alors
    annule pour eviter d'executer une version obsolete du code).
    """
    if not CFG.git_enabled:
        return True

    projet = CFG.remote_project_home()
    projet_abs = CFG.remote_project_abs()

    # Clone auto si le projet n'existe pas encore et qu'une URL est fournie.
    if CFG.git_repo:
        parent = os.path.dirname(projet_abs)
        cmd_clone = (
            f'mkdir -p "{parent}" && '
            f'[ -e "{projet}/.git" ] || git clone {CFG.git_repo} "{projet_abs}"'
        )
        _executer_et_journaliser(ssh_client, cmd_clone, "clone du repo (si absent)")

    cmd_pull = (
        f'cd "{projet}" && '
        f'git fetch --all --prune && '
        f'git checkout {CFG.git_branch} && '
        f'git pull --ff-only origin {CFG.git_branch}'
    )
    code = _executer_et_journaliser(ssh_client, cmd_pull, f"pull de la branche '{CFG.git_branch}'")
    if code != 0:
        print("[git] Echec de la synchronisation -> lancement annule pour ce script.")
        print(f"[git] Chemin attendu sur Grid5000 : {projet_abs}")
        print("[git] Verifiez 'remote_project_dir' dans config.yaml (doit correspondre au clone).")
        return False

    # Trace du commit deploye (utile pour la reproductibilite).
    _executer_et_journaliser(ssh_client, f'cd "{projet}" && git rev-parse --short HEAD', "commit deploye")
    return True


def etape1_association(ssh_client, site, nom_fichier, chemin_local, chemin_distant, *, git_sync=True):
    print(f"--- ÉTAPE 1 : Transfert et Lancement de {nom_fichier} ---")

    if git_sync and not synchroniser_git(ssh_client):
        return None

    if not televerser_fichier(ssh_client, chemin_local, chemin_distant):
        return None

    stdin, stdout, stderr = ssh_client.exec_command(f"chmod +x {chemin_distant}")
    stdout.channel.recv_exit_status()
    stdin, stdout, stderr = ssh_client.exec_command(f"chmod +x {CFG.remote_start_script_path()}")
    stdout.channel.recv_exit_status()

    chemin_distant_out, run_name = trouver_chemin_output_distant(ssh_client, chemin_distant)
    if not chemin_distant_out:
        return None

    ssh_client.exec_command(f"mkdir -p {chemin_distant_out}")
    commande_oar = generer_commande_soumission(ssh_client, chemin_distant)
    commande_totale = f"cd {chemin_distant_out} && {commande_oar}"

    job_id = make_job(ssh_client, commande_totale, site)
    if not job_id:
        return None

    sauvegarder_etat_fichier(nom_fichier, "ETAPE_2", "Lancement", job_id, site, chemin_local, chemin_distant)
    return job_id


def etape2_verification(ssh_client, job_id, nom_fichier, chemin_distant):
    print(f"--- ÉTAPE 2 : Vérification du Statut pour {nom_fichier} ---")
    info = obtenir_etat_specifique(nom_fichier)
    
    statut = obtenir_statut_oar(ssh_client, CFG.user, job_id)
    print(f"Statut OAR actuel : {statut}")
    
    if statut in ["R", "W", "F"]:
        sauvegarder_etat_fichier(nom_fichier, "ETAPE_2", statut, job_id, info["site"], info["chemin_local"], chemin_distant)
        return "EN_COURS"
        
    elif statut == "ABSENT":
        chemin_distant_out, _ = trouver_chemin_output_distant(ssh_client, chemin_distant)
        resultat_job = verifier_si_travail_fini(ssh_client, job_id, chemin_distant_out)
        
        if resultat_job == "FINI":
            sauvegarder_etat_fichier(nom_fichier, "ETAPE_3", "FINI", None, info["site"], info["chemin_local"], chemin_distant)
            return "FINI"
        elif resultat_job == "KILLED":
            sauvegarder_etat_fichier(nom_fichier, "ETAPE_2", "KILLED", None, info["site"], info["chemin_local"], chemin_distant)
            return "KILLED"
        else:
            sauvegarder_etat_fichier(nom_fichier, "ETAPE_3", "ERREUR", None, info["site"], info["chemin_local"], chemin_distant)
            return "ERREUR"


def etape3_recuperation(ssh_client, job_id, nom_fichier, chemin_distant, statut_flux="FINI"):
    print(f"--- ÉTAPE 3 : Récupération des résultats de {nom_fichier} ---")
    info = obtenir_etat_specifique(nom_fichier)
    
    chemin_distant_out, run_name = trouver_chemin_output_distant(ssh_client, chemin_distant)
    if not chemin_distant_out or not run_name:
        return False

    dossier_local_cible = f"{CFG.local_outputs_dir}/{run_name}"
    archive_distante = f"/tmp/{run_name}.tar"
    archive_locale = f"{CFG.local_outputs_dir}/{run_name}.tar"
    os.makedirs(CFG.local_outputs_dir, exist_ok=True)

    try:
        print(f"[SSH] Compression distante...")
        stdin, stdout, stderr = ssh_client.exec_command(f"tar -cf {archive_distante} -C {os.path.dirname(chemin_distant_out)} {run_name}")
        if stdout.channel.recv_exit_status() != 0:
            return False

        print(f"[SCP] Téléchargement...")
        if not telecharger_fichier(ssh_client, archive_distante, archive_locale):
            return False

        with tarfile.open(archive_locale, "r") as tar:
            tar.extractall(path=CFG.local_outputs_dir)
        os.remove(archive_locale)
        print(f"-> Sauvegardé localement dans : {dossier_local_cible}")

        if statut_flux == "FINI" and os.path.exists(dossier_local_cible) and len(os.listdir(dossier_local_cible)) > 0:
            ssh_client.exec_command(f"rm -rf {chemin_distant_out}")
            ssh_client.exec_command(f"rm -f {archive_distante}")
            
        sauvegarder_etat_fichier(nom_fichier, "TERMINE", statut_flux, None, info["site"], info["chemin_local"], chemin_distant)
        return True
    except Exception as e:
        print(f"Erreur lors de la récupération : {e}")
        return False


def archiver_script_local(nom_fichier, chemin_actuel):
    """ Déplace physiquement le script exécuté dans le dossier 'archive' """
    dossier_archive = f"{CFG.local_scripts_root}/archive"
    os.makedirs(dossier_archive, exist_ok=True)
    chemin_dest = f"{dossier_archive}/{nom_fichier}"
    
    if os.path.exists(chemin_actuel):
        shutil.move(chemin_actuel, chemin_dest)
        print(f"[Orchestrateur] Fichier script déplacé dans l'archive : {chemin_dest}")
        info = obtenir_etat_specifique(nom_fichier)
        sauvegarder_etat_fichier(nom_fichier, "TERMINE", info["statut_oar"], None, info["site"], chemin_dest, info["chemin_distant"])


# =============================================================
# 5. PILOTE DE RUN
# =============================================================

def piloter_un_script(ssh_client, site, nom_fichier, info, *, git_sync=True):
    """ Gère le cycle de vie d'un script spécifique sur un site précis """
    job_id = info["job_id"]
    etape = info["etape"]
    chemin_local = info["chemin_local"]
    chemin_distant = info["chemin_distant"]
    
    print(f"\n[Pilote] Analyse de {nom_fichier} sur {site} (Étape : {etape})")
    
    if etape == "ETAPE_1":
        etape1_association(
            ssh_client, site, nom_fichier, chemin_local, chemin_distant, git_sync=git_sync
        )
        
    elif etape == "ETAPE_2":
        statut_flux = etape2_verification(ssh_client, job_id, nom_fichier, chemin_distant)
        
        if statut_flux in ["FINI", "ERREUR"]:
            if etape3_recuperation(ssh_client, job_id, nom_fichier, chemin_distant, statut_flux):
                archiver_script_local(nom_fichier, chemin_local)
                
        elif statut_flux == "KILLED":
            # Relancement automatique (Walltime dépassé)
            chemin_distant_out, _ = trouver_chemin_output_distant(ssh_client, chemin_distant)
            ssh_client.exec_command(f"mkdir -p {chemin_distant_out}")
            commande_totale = f"cd {chemin_distant_out} && {generer_commande_soumission(ssh_client, chemin_distant)}"
            
            nouveau_job_id = make_job(ssh_client, commande_totale, site)
            if nouveau_job_id:
                sauvegarder_etat_fichier(nom_fichier, "ETAPE_2", "Lancement", nouveau_job_id, site, chemin_local, chemin_distant)

    elif etape == "ETAPE_3":
        if etape3_recuperation(ssh_client, job_id, nom_fichier, chemin_distant, info.get("statut_oar", "FINI")):
            archiver_script_local(nom_fichier, chemin_local)


# =============================================================
# 6. CHEF D'ORCHESTRE (MAIN PRINCIPAL)
# =============================================================

def main():
    global CFG

    parser = argparse.ArgumentParser(description="Orchestrateur Grid'5000")
    parser.add_argument(
        "--config",
        default=None,
        help="Chemin vers le fichier de configuration YAML (defaut: config.yaml).",
    )
    args = parser.parse_args()
    CFG = load_config(args.config)

    print("=======================================================")
    print("LANCEMENT DE LA TOURNÉE DE L'ORCHESTRATEUR GRID'5000")
    print(f"Utilisateur : {CFG.user} | Sites : {CFG.sites}")
    print("=======================================================")
    
    tous_les_etats = charger_tous_les_etats()
    
    for site in CFG.sites:
        # Trouver les scripts liés à ce site qui tournent ou attendent d'être récupérés
        scripts_du_site = [(nom, info) for nom, info in tous_les_etats.items() if info.get("site") == site and info.get("etape") != "TERMINE"]
        
        # Compter les jobs actifs
        nb_actifs = decompte_jobs_actifs_site(tous_les_etats, site)
        places_libres = CFG.max_jobs_per_site - nb_actifs
        
        # Chercher s'il y a des scripts orphelins à la racine du dossier local
        scripts_racine = [f for f in glob.glob(f"{CFG.local_scripts_root}/*.sh") if os.path.isfile(f)]
        
        # On ne se connecte à un serveur QUE s'il y a des fichiers à checker OU s'il y a de la place pour lancer
        if scripts_du_site or (places_libres > 0 and scripts_racine):
            print(f"\n>>> Connexion au site : {site.upper()} (Jobs actifs : {nb_actifs}/{CFG.max_jobs_per_site})")
            bastion, ssh_client = connecter_serveur_final(CFG.user, site, CFG.ssh_gateway)
            if not ssh_client:
                continue
                
            try:
                git_ok = synchroniser_git(ssh_client) if CFG.git_enabled else True
                if not git_ok:
                    print("[git] Sync echouee — suivi des jobs existants uniquement (pas de nouveaux lancements).")

                # Mise à jour des scripts en cours sur ce site
                for nom_fichier, info in scripts_du_site:
                    piloter_un_script(ssh_client, site, nom_fichier, info, git_sync=False)
                
                # On recharge les états (car des scripts ont pu passer en "TERMINE")
                tous_les_etats = charger_tous_les_etats()
                nb_actifs = decompte_jobs_actifs_site(tous_les_etats, site)
                places_libres = CFG.max_jobs_per_site - nb_actifs
                
                # ACTION 2 : Si places libres, on attribue de nouveaux scripts de la racine
                scripts_eligibles = [
                    s for s in scripts_racine if _eligible_pour_site(s, site)
                ]
                if places_libres > 0 and scripts_eligibles:
                    if not git_ok:
                        print(
                            f"[Orchestrateur] {len(scripts_eligibles)} script(s) "
                            f"eligible(s) pour {site} — git non synchronise."
                        )
                    else:
                        print(
                            f"[Orchestrateur] Il reste {places_libres} place(s) sur {site}. "
                            f"Attribution ({len(scripts_eligibles)} eligible(s))..."
                        )
                    
                    for i in range(min(places_libres, len(scripts_eligibles))):
                        if not git_ok:
                            break
                        script_a_attribuer = scripts_eligibles[i]
                        nom_f = os.path.basename(script_a_attribuer)
                        
                        # Déplacement physique vers le sous-dossier du site (ex: scrip_run/lille/)
                        dossier_site_local = f"{CFG.local_scripts_root}/{site}"
                        os.makedirs(dossier_site_local, exist_ok=True)
                        nouveau_chemin_local = f"{dossier_site_local}/{nom_f}"
                        shutil.move(script_a_attribuer, nouveau_chemin_local)
                        
                        # Chemin distant correspondant au sous-dossier du site (miroir du local)
                        chemin_distant_site = CFG.remote_script_path(f"{dossier_site_local}/{nom_f}")
                        
                        print(f" -> Déplacement local : {script_a_attribuer} -> {nouveau_chemin_local}")
                        
                        # Initialisation de l'état dans le JSON
                        sauvegarder_etat_fichier(nom_f, "ETAPE_1", "AUCUN", None, site, nouveau_chemin_local, chemin_distant_site)
                        
                        # Lancement immédiat du cycle de vie
                        info_initiale = obtenir_etat_specifique(nom_f)
                        piloter_un_script(ssh_client, site, nom_f, info_initiale, git_sync=False)
                        
            finally:
                deconnecter_serveurs(bastion, ssh_client)
                
    print("\n=======================================================")
    print("FIN DE LA TOURNÉE DE L'ORCHESTRATEUR")
    print("=======================================================")

if __name__ == "__main__":
    main()