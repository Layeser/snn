import time
import json
import os
import tarfile
import shutil
import glob
import argparse
import paramiko
import yaml
from pathlib import Path
from ssh_utils import *
from config import load_config

# La configuration (specifique a la personne / installation) est chargee au
# demarrage dans main() puis exposee via cette variable de module.
CFG = None
CLUSTER_DEFAULTS = {}

ETAPES_ACTIVES = ("ETAPE_1", "ETAPE_2", "ETAPE_3")
PILOT_GRID_DIR = os.path.dirname(os.path.abspath(__file__))

# Ligne imprimee par start_run.sh a la fin d'un run reussi. Doit rester
# STRICTEMENT identique a celle de scrip_grid_5000/start_run.sh.
SENTINELLE_FIN = "=== EXPERIENCE TERMINEE AVEC SUCCES ==="
# Fin d'un job OAR cluster (run_gpu_queue.sh sur tout le dossier).
SENTINELLE_CLUSTER = "=== CLUSTER QUEUE TERMINEE AVEC SUCCES ==="

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
    """Lit le script distant et extrait les options OAR (# OAR_option ...)."""
    options = {}
    oar_types: list[str] = []
    try:
        stdin, stdout, stderr = ssh_client.exec_command(f"cat {chemin_distant_script}")
        for ligne in stdout:
            parsed = _parse_oar_option_line(ligne)
            if parsed:
                flag, valeur = parsed
                if flag == "-t":
                    oar_types.append(valeur)
                else:
                    options[flag] = valeur
    except Exception as e:
        print(f"Erreur lors de la lecture distante des options OAR : {e}")
    return options, oar_types


def load_cluster_defaults() -> dict:
    rel = getattr(CFG, "cluster_defaults_file", None) or "cluster_defaults.yaml"
    chemin = rel if os.path.isabs(rel) else os.path.join(PILOT_GRID_DIR, rel)
    if not os.path.exists(chemin):
        return {}
    with open(chemin, "r") as f:
        return yaml.safe_load(f) or {}


def _repo_root() -> str:
    return os.path.dirname(PILOT_GRID_DIR)


def _scripts_root_abs() -> str:
    root = CFG.local_scripts_root
    if not os.path.isabs(root):
        root = os.path.join(_repo_root(), root)
    return os.path.normpath(root)


def _site_scripts_root(site: str) -> str:
    """Racine locale des scripts pour un site (besteffort_lille ou scrip_run/lille)."""
    custom = getattr(CFG, "site_scripts_root", None) or {}
    if site in custom:
        root = custom[site]
        if not os.path.isabs(root):
            root = os.path.join(_repo_root(), root)
        return os.path.normpath(root)
    return os.path.join(_scripts_root_abs(), site)


def meta_script(chemin_local: str) -> tuple[str, str, str, str] | None:
    """Retourne (rel_key, site, cluster, basename) pour un script d'experience."""
    chemin_local = os.path.normpath(chemin_local)
    repo = _repo_root()
    try:
        rel = os.path.relpath(chemin_local, repo).replace("\\", "/")
    except ValueError:
        return None
    parts = Path(rel).parts
    if len(parts) < 4 or not parts[-1].endswith(".sh"):
        return None
    if parts[0] != "scrip_grid_5000":
        return None

    nom = parts[-1]
    if parts[1].startswith("besteffort_"):
        site = parts[1].replace("besteffort_", "", 1)
        cluster = parts[2]
        rel_key = rel
        return rel_key, site, cluster, nom

    if parts[1] == "scrip_run" and len(parts) >= 5:
        site, cluster = parts[2], parts[3]
        rel_key = rel
        return rel_key, site, cluster, nom

    return None


def cluster_rel_key(site: str, cluster: str) -> str:
    return f"{site}/{cluster}"


def est_cle_cluster(rel_key: str) -> bool:
    parts = Path(rel_key).parts
    return len(parts) == 2 and not parts[-1].endswith(".sh")


def lister_scripts_cluster(site: str, cluster: str) -> list[str]:
    cluster_dir = os.path.join(_scripts_root_abs(), site, cluster)
    return sorted(glob.glob(os.path.join(cluster_dir, "*.sh")))


def decouvrir_clusters_en_attente(site: str, etats: dict) -> list[tuple[str, str, str, list[str]]]:
    """Clusters avec des .sh et sans job cluster actif/termine."""
    resultat = []
    site_dir = os.path.join(_scripts_root_abs(), site)
    if not os.path.isdir(site_dir):
        return resultat

    for cluster in sorted(os.listdir(site_dir)):
        if cluster in ("archive",):
            continue
        cluster_dir = os.path.join(site_dir, cluster)
        if not os.path.isdir(cluster_dir):
            continue
        scripts = lister_scripts_cluster(site, cluster)
        if not scripts:
            continue
        rel_key = cluster_rel_key(site, cluster)
        info = etats.get(rel_key, {})
        if info.get("etape") == "TERMINE":
            continue
        if info.get("etape") in ETAPES_ACTIVES:
            continue
        resultat.append((rel_key, site, cluster, scripts))
    return resultat


def decouvrir_scripts_en_attente(site: str, etats: dict) -> list[tuple[str, str, str, str]]:
    """Scripts locaux pas encore TERMINE (besteffort_* ou scrip_run/<site>/)."""
    resultat = []
    site_dir = _site_scripts_root(site)
    if not os.path.isdir(site_dir):
        return resultat

    for cluster in sorted(os.listdir(site_dir)):
        if cluster in ("archive",):
            continue
        cluster_dir = os.path.join(site_dir, cluster)
        if not os.path.isdir(cluster_dir):
            continue
        for chemin in sorted(glob.glob(os.path.join(cluster_dir, "*.sh"))):
            meta = meta_script(chemin)
            if not meta:
                continue
            rel_key, site_lu, cluster_lu, nom = meta
            if site_lu != site:
                continue
            info = etats.get(rel_key, {})
            if info.get("etape") == "TERMINE":
                continue
            if info.get("etape") in ETAPES_ACTIVES:
                continue
            resultat.append((rel_key, chemin, cluster_lu, nom))
    return resultat


def generer_commande_soumission_cluster(site: str, cluster: str) -> str:
    """Un job OAR par dossier cluster ; run_gpu_queue.sh gere la file sur le noeud."""
    defaults = CLUSTER_DEFAULTS.get(cluster, {})
    max_gpu = int(defaults.get("max_jobs", 1))
    base_l = defaults.get("oar_resources") or f"host=1/gpu={max_gpu}"
    ressources = f"{base_l},walltime={CFG.walltime}"

    arguments_oar = [f'-l "{ressources}"']
    oar_types = defaults.get("oar_types") or ([CFG.oar_type] if CFG.oar_type else [])
    for oar_type in oar_types:
        arguments_oar.append(f'-t "{oar_type}"')

    queue = CFG.oar_queue or defaults.get("oar_queue")
    if queue:
        arguments_oar.append(f'-q "{queue}"')
    arguments_oar.append(f'-p "{cluster}"')

    rel_queue = os.path.join(CFG.local_scripts_root, site, cluster).replace("\\", "/")
    remote_queue = f"{CFG.remote_project_home()}/{rel_queue}"
    remote_runner = f"{CFG.remote_project_home()}/scrip_grid_5000/run_gpu_queue.sh"
    inner = (
        f"bash {remote_runner} --cluster {cluster} "
        f'--queue-dir {remote_queue} --max-gpus {max_gpu} --no-git-pull'
    )
    return f'oarsub {" ".join(arguments_oar)} "{inner}"'


def generer_commande_soumission(ssh_client, chemin_distant_sh, cluster: str | None = None):
    options_script, oar_types_script = extraire_options_oar(ssh_client, chemin_distant_sh)
    defaults = CLUSTER_DEFAULTS.get(cluster or {}, {})

    base_l = (
        options_script.get("-l")
        or defaults.get("oar_resources")
        or CFG.oar_resources
        or "host=1/gpu=1"
    )
    ressources = f"{base_l},walltime={CFG.walltime}"

    arguments_oar = [f'-l "{ressources}"']

    oar_types = oar_types_script or defaults.get("oar_types") or ([CFG.oar_type] if CFG.oar_type else [])
    for oar_type in oar_types:
        arguments_oar.append(f'-t "{oar_type}"')

    queue = CFG.oar_queue or options_script.get("-q")
    if queue:
        arguments_oar.append(f'-q "{queue}"')

    cluster_oar = options_script.get("-p") or cluster
    if cluster_oar:
        arguments_oar.append(f'-p "{cluster_oar}"')

    for flag, valeur in options_script.items():
        if flag in ("-l", "-q", "-p"):
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

def sauvegarder_etat_fichier(rel_key, etape, statut_oar, job_id, site, cluster, chemin_local, chemin_distant, scripts=None):
    tous_les_etats = charger_tous_les_etats()
    entree = {
        "etape": etape,
        "statut_oar": statut_oar,
        "job_id": job_id,
        "site": site,
        "cluster": cluster,
        "chemin_local": chemin_local,
        "chemin_distant": chemin_distant,
        "derniere_verification": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    if scripts is not None:
        entree["scripts"] = scripts
    elif rel_key in tous_les_etats and "scripts" in tous_les_etats[rel_key]:
        entree["scripts"] = tous_les_etats[rel_key]["scripts"]
    tous_les_etats[rel_key] = entree
    os.makedirs(os.path.dirname(CFG.state_file), exist_ok=True)
    with open(CFG.state_file, 'w') as f:
        json.dump(tous_les_etats, f, indent=4)


def obtenir_etat_specifique(rel_key):
    tous_les_etats = charger_tous_les_etats()
    if rel_key in tous_les_etats:
        return tous_les_etats[rel_key]
    return {
        "etape": "AUCUNE",
        "statut_oar": "AUCUN",
        "job_id": None,
        "site": None,
        "cluster": None,
        "chemin_local": None,
        "chemin_distant": None,
    }


def decompte_jobs_actifs_site(tous_les_etats, site):
    return sum(
        1
        for info in tous_les_etats.values()
        if info.get("site") == site and info.get("etape") in ETAPES_ACTIVES
    )


def decompte_jobs_actifs_cluster(tous_les_etats, site, cluster):
    return sum(
        1
        for info in tous_les_etats.values()
        if info.get("site") == site
        and info.get("cluster") == cluster
        and info.get("etape") in ETAPES_ACTIVES
    )


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


def verifier_cluster_fini(ssh_client, job_id, log_dir):
    """Detecte la fin du job OAR cluster (queue complete sur le noeud)."""
    for nom in (f"OAR.{job_id}.stderr", f"OAR.{job_id}.stdout"):
        stdin, stdout, stderr = ssh_client.exec_command(f"tail -n 30 {log_dir}/{nom} 2>/dev/null")
        contenu = stdout.read().decode("utf-8")
        if f"Job {job_id} KILLED" in contenu:
            return "KILLED"
    stdin, stdout, stderr = ssh_client.exec_command(f"tail -n 30 {log_dir}/OAR.{job_id}.stdout 2>/dev/null")
    if SENTINELLE_CLUSTER in stdout.read().decode("utf-8"):
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


def synchroniser_scripts_cluster(ssh_client, site: str, cluster: str, scripts_locaux: list[str]) -> bool:
    """Envoie les scripts locaux vers scrip_run/<site>/<cluster>/ sur la frontale.

    scrip_run/lille|lyon/ est gitignore : le git pull seul ne suffit pas.
    """
    if not scripts_locaux:
        print(f"[Cluster] Aucun script local pour {site}/{cluster}")
        return False

    repo_root = os.path.dirname(os.path.dirname(_scripts_root_abs()))
    rel_queue = os.path.join(CFG.local_scripts_root, site, cluster).replace("\\", "/")
    remote_queue = CFG.remote_script_path(rel_queue)

    print(f"[Cluster] Sync scripts → {remote_queue} ({len(scripts_locaux)} fichier(s))")
    ssh_client.exec_command(f'mkdir -p "{remote_queue}"')
    ssh_client.exec_command(f'rm -f "{remote_queue}"/*.sh')

    ok = True
    for chemin_local in scripts_locaux:
        if not os.path.isabs(chemin_local):
            candidat = os.path.join(repo_root, chemin_local)
            chemin_local = candidat if os.path.isfile(candidat) else os.path.normpath(chemin_local)
        if not os.path.isfile(chemin_local):
            print(f"[Cluster] Script local introuvable : {chemin_local}")
            ok = False
            continue
        rel = os.path.relpath(chemin_local, repo_root).replace("\\", "/")
        chemin_distant = CFG.remote_script_path(rel)
        if not televerser_fichier(ssh_client, chemin_local, chemin_distant):
            ok = False
            continue
        ssh_client.exec_command(f'chmod +x "{chemin_distant}"')

    if ok:
        stdin, stdout, stderr = ssh_client.exec_command(f'ls -1 "{remote_queue}"/*.sh 2>/dev/null | wc -l')
        nb = stdout.read().decode("utf-8").strip()
        print(f"[Cluster] Frontale : {nb} script(s) dans {rel_queue}/")
    return ok


def etape1_cluster(ssh_client, site, cluster, rel_key, scripts_locaux, *, git_sync=True):
    """Soumet un job OAR par cluster ; la file GPU tourne sur le noeud."""
    print(f"--- ETAPE 1 : Job cluster {rel_key} ({len(scripts_locaux)} script(s)) ---")

    if git_sync and not synchroniser_git(ssh_client):
        return None

    if not synchroniser_scripts_cluster(ssh_client, site, cluster, scripts_locaux):
        print(f"[Cluster] Echec sync scripts → oarsub annule pour {rel_key}")
        return None

    repo_root = os.path.dirname(os.path.dirname(_scripts_root_abs()))
    rel_scripts = [os.path.relpath(p, repo_root).replace("\\", "/") for p in scripts_locaux]

    project = CFG.remote_project_abs()
    commande_oar = generer_commande_soumission_cluster(site, cluster)
    commande_totale = f'cd "{project}" && {commande_oar}'

    job_id = make_job(ssh_client, commande_totale, site)
    if not job_id:
        return None

    sauvegarder_etat_fichier(
        rel_key, "ETAPE_2", "Lancement", job_id, site, cluster,
        None, project, scripts=rel_scripts,
    )
    return job_id


def recuperer_cluster(ssh_client, rel_key, info, statut_flux="FINI"):
    """Rapatrie les resultats de chaque script du cluster."""
    repo_root = os.path.dirname(os.path.dirname(_scripts_root_abs()))
    scripts = info.get("scripts") or []
    if not scripts:
        print(f"[Cluster] Aucun script enregistre pour {rel_key}")
        return False

    ok = True
    for rel_script in scripts:
        chemin_local = os.path.join(repo_root, rel_script)
        chemin_distant = CFG.remote_script_path(rel_script)
        archive_key = f"{rel_key}/{os.path.basename(rel_script)}"

        if not os.path.isfile(chemin_local):
            archive_path = os.path.join(
                _scripts_root_abs(), "archive", rel_key, os.path.basename(rel_script)
            )
            if os.path.isfile(archive_path):
                chemin_local = archive_path
            else:
                print(f"[Cluster] Script local introuvable : {rel_script}")
                ok = False
                continue

        if not etape3_recuperation(
            ssh_client, info.get("job_id"), archive_key, chemin_distant, statut_flux,
            site=info.get("site"), cluster=info.get("cluster"), persist_state=False,
        ):
            ok = False
            continue
        if "archive" not in chemin_local.replace("\\", "/").split("/"):
            archiver_script_local(archive_key, chemin_local, persist_state=False)

    sauvegarder_etat_fichier(
        rel_key, "TERMINE", statut_flux, None, info["site"], info.get("cluster"),
        None, info.get("chemin_distant"), scripts=scripts,
    )
    return ok


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


def etape1_association(ssh_client, site, cluster, rel_key, chemin_local, chemin_distant, *, git_sync=True):
    print(f"--- ÉTAPE 1 : Transfert et Lancement de {rel_key} ---")

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
    commande_oar = generer_commande_soumission(ssh_client, chemin_distant, cluster)
    commande_totale = f"cd {chemin_distant_out} && {commande_oar}"

    job_id = make_job(ssh_client, commande_totale, site)
    if not job_id:
        return None

    sauvegarder_etat_fichier(
        rel_key, "ETAPE_2", "Lancement", job_id, site, cluster, chemin_local, chemin_distant
    )
    return job_id


def etape2_verification(ssh_client, job_id, rel_key, chemin_distant):
    print(f"--- ÉTAPE 2 : Vérification du Statut pour {rel_key} ---")
    info = obtenir_etat_specifique(rel_key)
    
    statut = obtenir_statut_oar(ssh_client, CFG.user, job_id)
    print(f"Statut OAR actuel : {statut}")
    
    if statut in ["R", "W", "F"]:
        sauvegarder_etat_fichier(
            rel_key,
            "ETAPE_2",
            statut,
            job_id,
            info["site"],
            info.get("cluster"),
            info["chemin_local"],
            chemin_distant,
        )
        return "EN_COURS"
        
    elif statut == "ABSENT":
        chemin_distant_out, _ = trouver_chemin_output_distant(ssh_client, chemin_distant)
        resultat_job = verifier_si_travail_fini(ssh_client, job_id, chemin_distant_out)
        
        if resultat_job == "FINI":
            sauvegarder_etat_fichier(
                rel_key, "ETAPE_3", "FINI", None, info["site"], info.get("cluster"),
                info["chemin_local"], chemin_distant,
            )
            return "FINI"
        elif resultat_job == "KILLED":
            sauvegarder_etat_fichier(
                rel_key, "ETAPE_2", "KILLED", None, info["site"], info.get("cluster"),
                info["chemin_local"], chemin_distant,
            )
            return "KILLED"
        else:
            sauvegarder_etat_fichier(
                rel_key, "ETAPE_3", "ERREUR", None, info["site"], info.get("cluster"),
                info["chemin_local"], chemin_distant,
            )
            return "ERREUR"


def etape3_recuperation(
    ssh_client, job_id, rel_key, chemin_distant, statut_flux="FINI",
    *, site=None, cluster=None, persist_state=True,
):
    print(f"--- ÉTAPE 3 : Récupération des résultats de {rel_key} ---")
    info = obtenir_etat_specifique(rel_key)
    site = site or info.get("site")
    cluster = cluster or info.get("cluster")
    
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
            
        if persist_state:
            sauvegarder_etat_fichier(
                rel_key, "TERMINE", statut_flux, None, site, cluster,
                info.get("chemin_local"), chemin_distant,
            )
        return True
    except Exception as e:
        print(f"Erreur lors de la récupération : {e}")
        return False


def archiver_script_local(rel_key, chemin_actuel, *, persist_state=True):
    """Déplace le script exécuté dans archive/done/ (à côté du script ou scrip_run/archive/)."""
    if getattr(CFG, "submission_mode", "per_cluster") == "per_script":
        archive_dir = os.path.join(os.path.dirname(chemin_actuel), "archive", "done")
        os.makedirs(archive_dir, exist_ok=True)
        name = os.path.basename(chemin_actuel)
        chemin_dest = os.path.join(archive_dir, name)
        if os.path.exists(chemin_dest):
            chemin_dest = os.path.join(
                archive_dir, f"{time.strftime('%Y%m%d_%H%M%S')}_{name}"
            )
    else:
        chemin_dest = os.path.join(_scripts_root_abs(), "archive", rel_key)
        os.makedirs(os.path.dirname(chemin_dest), exist_ok=True)

    if os.path.exists(chemin_actuel):
        shutil.move(chemin_actuel, chemin_dest)
        print(f"[Orchestrateur] Script archivé : {chemin_actuel} -> {chemin_dest}")
        if persist_state:
            info = obtenir_etat_specifique(rel_key)
            sauvegarder_etat_fichier(
                rel_key, "TERMINE", info["statut_oar"], None, info["site"], info.get("cluster"),
                chemin_dest, info["chemin_distant"],
            )


# =============================================================
# 5. PILOTE DE RUN
# =============================================================

def piloter_un_cluster(ssh_client, site, rel_key, info, *, git_sync=True):
    """Cycle de vie d'un job cluster (cle site/cluster)."""
    job_id = info.get("job_id")
    etape = info.get("etape")
    cluster = info.get("cluster") or rel_key.split("/")[-1]
    project = info.get("chemin_distant") or CFG.remote_project_abs()

    print(f"\n[Pilote cluster] {rel_key} ({len(info.get('scripts') or [])} script(s), etape {etape})")

    if etape == "ETAPE_1":
        scripts = lister_scripts_cluster(site, cluster)
        if not scripts:
            print(f"[Cluster] Aucun .sh dans scrip_run/{site}/{cluster}/")
            return
        etape1_cluster(ssh_client, site, cluster, rel_key, scripts, git_sync=git_sync)

    elif etape == "ETAPE_2" and job_id:
        statut = obtenir_statut_oar(ssh_client, CFG.user, job_id)
        print(f"Statut OAR job cluster {job_id} : {statut}")

        if statut in ["R", "W", "F"]:
            sauvegarder_etat_fichier(
                rel_key, "ETAPE_2", statut, job_id, site, cluster,
                None, project, scripts=info.get("scripts"),
            )
            return

        if statut == "ABSENT":
            resultat = verifier_cluster_fini(ssh_client, job_id, project)
            if resultat == "FINI":
                sauvegarder_etat_fichier(
                    rel_key, "ETAPE_3", "FINI", job_id, site, cluster,
                    None, project, scripts=info.get("scripts"),
                )
                recuperer_cluster(ssh_client, rel_key, obtenir_etat_specifique(rel_key), "FINI")
            elif resultat == "KILLED":
                sauvegarder_etat_fichier(
                    rel_key, "ETAPE_2", "KILLED", None, site, cluster,
                    None, project, scripts=info.get("scripts"),
                )
                scripts = lister_scripts_cluster(site, cluster)
                if scripts:
                    etape1_cluster(ssh_client, site, cluster, rel_key, scripts, git_sync=False)
            else:
                sauvegarder_etat_fichier(
                    rel_key, "ETAPE_3", "ERREUR", None, site, cluster,
                    None, project, scripts=info.get("scripts"),
                )
                recuperer_cluster(ssh_client, rel_key, obtenir_etat_specifique(rel_key), "ERREUR")

    elif etape == "ETAPE_3":
        recuperer_cluster(ssh_client, rel_key, info, info.get("statut_oar", "FINI"))


def piloter_un_script(ssh_client, site, rel_key, info, *, git_sync=True):
    """Gère le cycle de vie d'un script (clé scrip_run/<site>/<cluster>/x.sh)."""
    job_id = info["job_id"]
    etape = info["etape"]
    chemin_local = info["chemin_local"]
    chemin_distant = info["chemin_distant"]
    cluster = info.get("cluster") or (meta_script(chemin_local) or (None, None, None, None))[2]
    
    print(f"\n[Pilote] Analyse de {rel_key} sur {site}/{cluster} (Étape : {etape})")
    
    if etape == "ETAPE_1":
        etape1_association(
            ssh_client, site, cluster, rel_key, chemin_local, chemin_distant, git_sync=git_sync
        )
        
    elif etape == "ETAPE_2":
        statut_flux = etape2_verification(ssh_client, job_id, rel_key, chemin_distant)
        
        if statut_flux in ["FINI", "ERREUR"]:
            if etape3_recuperation(ssh_client, job_id, rel_key, chemin_distant, statut_flux):
                archiver_script_local(rel_key, chemin_local)
                
        elif statut_flux == "KILLED":
            chemin_distant_out, _ = trouver_chemin_output_distant(ssh_client, chemin_distant)
            ssh_client.exec_command(f"mkdir -p {chemin_distant_out}")
            commande_totale = f"cd {chemin_distant_out} && {generer_commande_soumission(ssh_client, chemin_distant, cluster)}"
            
            nouveau_job_id = make_job(ssh_client, commande_totale, site)
            if nouveau_job_id:
                sauvegarder_etat_fichier(
                    rel_key, "ETAPE_2", "Lancement", nouveau_job_id, site, cluster,
                    chemin_local, chemin_distant,
                )

    elif etape == "ETAPE_3":
        if etape3_recuperation(ssh_client, job_id, rel_key, chemin_distant, info.get("statut_oar", "FINI")):
            archiver_script_local(rel_key, chemin_local)


# =============================================================
# 6. CHEF D'ORCHESTRE — mode per_script (besteffort)
# =============================================================

def main_per_script(args):
    """1 job OAR par .sh ; resoumission automatique apres preemption (KILLED)."""
    print("=======================================================")
    if args.follow_only:
        print("MODE SUIVI BESTEFFORT — pas de nouvelle soumission OAR")
    else:
        print("ORCHESTRATEUR BESTEFFORT — 1 GPU par experience")
    print(f"Utilisateur : {CFG.user} | Sites : {CFG.sites}")
    print(f"Parallele max / site : {CFG.max_jobs_per_site}")
    print("=======================================================")

    tous_les_etats = charger_tous_les_etats()

    for site in CFG.sites:
        scripts_actifs = [
            (cle, info)
            for cle, info in tous_les_etats.items()
            if info.get("site") == site
            and info.get("etape") != "TERMINE"
            and not est_cle_cluster(cle)
        ]
        scripts_en_attente = decouvrir_scripts_en_attente(site, tous_les_etats)

        if not scripts_actifs and not scripts_en_attente:
            print(f"\n[{site}] Aucun script actif ni en attente.")
            continue

        nb_actifs = sum(1 for _, info in scripts_actifs if info.get("job_id"))
        print(
            f"\n>>> Site {site.upper()} — suivis {len(scripts_actifs)}, "
            f"jobs OAR {nb_actifs}, en attente {len(scripts_en_attente)}"
        )

        bastion, ssh_client = connecter_serveur_final(CFG.user, site, CFG.ssh_gateway)
        if not ssh_client:
            continue

        try:
            git_ok = synchroniser_git(ssh_client) if CFG.git_enabled else True
            if not git_ok:
                print("[git] Sync echouee — suivi des jobs existants uniquement.")

            for rel_key, info in scripts_actifs:
                piloter_un_script(ssh_client, site, rel_key, info, git_sync=False)

            tous_les_etats = charger_tous_les_etats()
            slots_libres = CFG.max_jobs_per_site - decompte_jobs_actifs_site(
                tous_les_etats, site
            )

            if git_ok and scripts_en_attente and not args.follow_only and slots_libres > 0:
                a_soumettre = scripts_en_attente[:slots_libres]
                print(
                    f"[Besteffort] Soumission de {len(a_soumettre)} job(s) sur {site} "
                    f"({slots_libres} slot(s) libre(s))."
                )
                for rel_key, chemin_local, cluster, nom in a_soumettre:
                    chemin_distant = CFG.remote_script_path(rel_key)
                    print(f"[Besteffort] oarsub {nom} (cluster {cluster})")
                    sauvegarder_etat_fichier(
                        rel_key, "ETAPE_1", "AUCUN", None, site, cluster,
                        chemin_local, chemin_distant,
                    )
                    info_init = obtenir_etat_specifique(rel_key)
                    piloter_un_script(ssh_client, site, rel_key, info_init, git_sync=False)
                    tous_les_etats = charger_tous_les_etats()

        finally:
            deconnecter_serveurs(bastion, ssh_client)

    print("\n=======================================================")
    print("FIN DE LA TOURNEE BESTEFFORT")
    print("=======================================================")


# =============================================================
# 7. CHEF D'ORCHESTRE (MAIN PRINCIPAL)
# =============================================================

def main():
    global CFG

    parser = argparse.ArgumentParser(description="Orchestrateur Grid'5000")
    parser.add_argument(
        "--config",
        default=None,
        help="Chemin vers le fichier de configuration YAML (defaut: config.yaml).",
    )
    parser.add_argument(
        "--follow-only",
        action="store_true",
        help="Suivre les jobs deja soumis et rapatrier les resultats ; ne pas soumettre de nouveaux scripts.",
    )
    parser.add_argument(
        "--sites",
        nargs="+",
        default=None,
        help="Limiter aux sites indiques (ex. --sites lyon). Defaut : tous les sites du config.",
    )
    args = parser.parse_args()
    CFG = load_config(args.config)
    if args.sites:
        CFG.sites = args.sites
    global CLUSTER_DEFAULTS
    CLUSTER_DEFAULTS = load_cluster_defaults()

    if getattr(CFG, "submission_mode", "per_cluster") == "per_script":
        main_per_script(args)
        return

    print("=======================================================")
    if args.follow_only:
        print("MODE SUIVI — pas de nouvelle soumission OAR")
    else:
        print("LANCEMENT DE LA TOURNÉE DE L'ORCHESTRATEUR GRID'5000")
    print(f"Utilisateur : {CFG.user} | Sites : {CFG.sites}")
    print(f"File d'attente : 1 job OAR par dossier scrip_run/<site>/<cluster>/ (file GPU sur le noeud)")
    print("=======================================================")
    
    tous_les_etats = charger_tous_les_etats()
    
    for site in CFG.sites:
        clusters_actifs = [
            (cle, info)
            for cle, info in tous_les_etats.items()
            if info.get("site") == site
            and info.get("etape") != "TERMINE"
            and est_cle_cluster(cle)
        ]
        clusters_en_attente = decouvrir_clusters_en_attente(site, tous_les_etats)

        if not clusters_actifs and not clusters_en_attente:
            continue

        nb_soumis = sum(1 for _, info in clusters_actifs if info.get("job_id") is not None)
        nb_scripts = sum(len(c[3]) for c in clusters_en_attente)
        print(
            f"\n>>> Connexion au site : {site.upper()} "
            f"(clusters suivis {len(clusters_actifs)}, jobs OAR {nb_soumis}, "
            f"clusters a soumettre {len(clusters_en_attente)}, "
            f"scripts en file {nb_scripts})"
        )
        bastion, ssh_client = connecter_serveur_final(CFG.user, site, CFG.ssh_gateway)
        if not ssh_client:
            continue
                
        try:
            git_ok = synchroniser_git(ssh_client) if CFG.git_enabled else True
            if not git_ok:
                print("[git] Sync echouee — suivi des jobs existants uniquement.")

            for rel_key, info in clusters_actifs:
                piloter_un_cluster(ssh_client, site, rel_key, info, git_sync=False)

            tous_les_etats = charger_tous_les_etats()

            if git_ok and clusters_en_attente and not args.follow_only:
                print(
                    f"[Orchestrateur] Soumission de {len(clusters_en_attente)} job(s) cluster "
                    f"sur {site} (parallele GPU + file sur chaque noeud)."
                )
                for rel_key, site_c, cluster, scripts in clusters_en_attente:
                    print(
                        f"[Orchestrateur] oarsub cluster {cluster} "
                        f"({len(scripts)} script(s), max_jobs={CLUSTER_DEFAULTS.get(cluster, {}).get('max_jobs', '?')} GPU)"
                    )
                    sauvegarder_etat_fichier(
                        rel_key, "ETAPE_1", "AUCUN", None, site_c, cluster,
                        None, CFG.remote_project_abs(), scripts=[
                            os.path.relpath(
                                p, os.path.dirname(os.path.dirname(_scripts_root_abs()))
                            ).replace("\\", "/")
                            for p in scripts
                        ],
                    )
                    info_init = obtenir_etat_specifique(rel_key)
                    piloter_un_cluster(ssh_client, site_c, rel_key, info_init, git_sync=False)
                        
        finally:
            deconnecter_serveurs(bastion, ssh_client)
                
    print("\n=======================================================")
    print("FIN DE LA TOURNÉE DE L'ORCHESTRATEUR")
    print("=======================================================")

if __name__ == "__main__":
    main()