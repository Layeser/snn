import os

import paramiko

# Le bastion Grid5000 ne change jamais, on le définit en constante globale ici
BASTION_HOST = "access.grid5000.fr"

def connecter_serveur_final(username, final_host):
    """
    Se connecte à access.grid5000.fr puis rebondit sur le serveur final choisi.
    Retourne les objets clients pour pouvoir exécuter des commandes et fermer la session.
    """
    try:
        # 1. Connexion au serveur de rebond (Grid5000)
        print(f"Connexion au bastion {BASTION_HOST}...")
        bastion = paramiko.SSHClient()
        bastion.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        bastion.connect(hostname=BASTION_HOST, username=username)

        # 2. Création du tunnel vers le serveur final
        print(f"Création du tunnel vers {final_host}...")
        bastion_transport = bastion.get_transport()
        dest_addr = (final_host, 22)
        local_addr = ('localhost', 0)
        tunnel_channel = bastion_transport.open_channel("direct-tcpip", dest_addr, local_addr)

        # 3. Connexion au serveur final à travers le tunnel
        print(f"Connexion finale à {final_host}...")
        client_final = paramiko.SSHClient()
        client_final.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client_final.connect(hostname=final_host, username=username, sock=tunnel_channel)

        # On retourne les deux clients pour pouvoir les utiliser et les fermer plus tard
        return bastion, client_final

    except Exception as e:
        print(f"Erreur lors de la connexion : {e}")
        # En cas d'erreur, on s'assure de ne pas laisser de connexions fantômes
        if 'bastion' in locals(): bastion.close()
        return None, None


def deconnecter_serveurs(bastion, client_final):
    """
    Ferme proprement les connexions SSH passées en argument.
    """
    try:
        if client_final:
            client_final.close()
        if bastion:
            bastion.close()
        print("Connexions SSH fermées proprement.")
    except Exception as e:
        print(f"Erreur lors de la déconnexion : {e}")

def extraire_options_oar(chemin_local_script):
    """
    Lit le script bash local et extrait toutes les lignes commençant par '## OAR_option'.
    Retourne un dictionnaire contenant le flag (ex: '-p') et sa valeur.
    """
    options = {}
    try:
        with open(chemin_local_script, 'r') as f:
            for ligne in f:
                if ligne.startswith("## OAR_option"):
                    # On nettoie la ligne pour enlever "## OAR_option"
                    contenu = ligne.replace("## OAR_option", "").strip()
                    
                    # On sépare le flag du reste de la valeur au premier espace trouvé
                    # ex: "-p gpu-16GB..." -> ["-p", "gpu-16GB..."]
                    parts = contenu.split(maxsplit=1)
                    if len(parts) == 2:
                        flag = parts[0]   # ex: "-p" ou "-l"
                        valeur = parts[1] # ex: "gpu-16GB AND..." ou "host=1,gpu=2"
                        options[flag] = valeur
    except Exception as e:
        print(f"Erreur lors de la lecture des options OAR du script : {e}")
        
    return options

# def generer_commande_soumission(chemin_test_train_sh):
#     """Génère la commande oarsub pour soumettre le script de test/train sur le serveur final avec les options appropriées."""
#     home = "$HOME"
#     script_start = f"{home}/detr-projet/scrip_grid_5000/start_run.sh"
#     options_oar = '-q default -p "chifflot" -l host=1,gpu=2,walltime=10:00:00 -t night'
#     commande = f'oarsub {options_oar} "{script_start} {chemin_test_train_sh}"'
#     return commande

def extraire_options_oar(ssh_client, chemin_distant_script):
    """
    Lit le script bash directement SUR LE SERVEUR DISTANT et extrait 
    toutes les lignes commençant par '## OAR_option'.
    """
    options = {}
    try:
        # On utilise 'cat' pour lire le fichier qui vient d'être téléversé
        stdin, stdout, stderr = ssh_client.exec_command(f"cat {chemin_distant_script}")
        
        # stdout permet de boucler directement sur les lignes de texte renvoyées par le serveur
        for ligne in stdout:
            if ligne.startswith("# OAR_option"):
                # On nettoie la ligne pour enlever "# OAR_option"
                contenu = ligne.replace("# OAR_option", "").strip()
                
                # On sépare le flag du reste de la valeur au premier espace
                parts = contenu.split(maxsplit=1)
                if len(parts) == 2:
                    flag = parts[0]   # ex: "-p" ou "-l"
                    valeur = parts[1] # ex: "gpu-16GB AND..." ou "host=1,gpu=2"
                    options[flag] = valeur
    except Exception as e:
        print(f"Erreur lors de la lecture distante des options OAR : {e}")
        
    return options


def generer_commande_soumission(ssh_client, chemin_distant_sh):
    """
    Génère la commande oarsub finale en lisant le fichier déjà présent sur le serveur.
    """
    walltime_global = "10:00:00"
    type_oar = "night"
    
    # 1. Récupération des options depuis le fichier DISTANT (on passe le ssh_client)
    options_script = extraire_options_oar(ssh_client, chemin_distant_sh)
    
    # 2. Gestion du cas d'exception : le "-l" (Ressources + Walltime)
    if "-l" in options_script:
        ressources = f"{options_script['-l']},walltime={walltime_global}"
    else:
        ressources = f"host=1,gpu=1,walltime={walltime_global}"
        
    # 3. Assemblage de la liste des arguments d'oarsub
    arguments_oar = []
    arguments_oar.append(f'-l "{ressources}"')
    arguments_oar.append(f'-t "{type_oar}"')
    
    for flag, valeur in options_script.items():
        if flag != "-l":  
            # Le f'{flag} "{valeur}"' permet d'obtenir par exemple : -p "gpu_compute_capability..."
            arguments_oar.append(f'{flag} "{valeur}"')
            
    options_string = " ".join(arguments_oar)
    
    # 4. Chemins et commande finale
    home = "$HOME"
    script_start = f"{home}/detr-projet/scrip_grid_5000/start_run.sh"
    
    commande_finale = f'oarsub {options_string} "{script_start} {chemin_distant_sh}"'
    return commande_finale




def televerser_fichier(client_final, chemin_local, chemin_distant):
    """ Équivalent de SCP : Envoie un fichier local vers le serveur final """
    try:
        # Vérification locale avant de lancer le SSH
        if not os.path.exists(chemin_local):
            print(f"Erreur locale : Le fichier '{chemin_local}' n'existe pas sur votre machine.")
            return False

        print("Ouverture du canal SFTP...")
        sftp = client_final.open_sftp()
        
        # Astuce : On essaie de créer les dossiers distants s'ils n'existent pas
        dossier_distant = os.path.dirname(chemin_distant)
        try:
            # Cette commande crée les dossiers un par un sur le serveur (équivalent de mkdir -p)
            print(f"Vérification/Création du dossier distant : {dossier_distant}")
            current_dir = ""
            for dir_part in dossier_distant.split('/'):
                if dir_part:
                    current_dir += f"/{dir_part}"
                    try:
                        sftp.mkdir(current_dir)
                    except IOError:
                        pass # Le dossier existe déjà, on continue
        except Exception as e:
            print(f"Note (Dossiers distants) : {e}")

        print(f"Copie en cours : {chemin_local} -> {chemin_distant}...")
        sftp.put(chemin_local, chemin_distant)
        sftp.close()
        
        print(f"Fichier copié avec succès vers {chemin_distant}")
        return True
    except Exception as e:
        print(f"❌ Erreur lors du transfert SFTP : {e}")
        return False
    
def telecharger_fichier(client_final, chemin_distant, chemin_local):
    """ Équivalent de SCP : Récupère un fichier distant et le sauvegarde en local """
    try:
        print("Ouverture du canal SFTP pour téléchargement...")
        sftp = client_final.open_sftp()
        
        # Sécurité : On s'assure que le dossier local de destination existe
        dossier_local = os.path.dirname(chemin_local)
        if dossier_local and not os.path.exists(dossier_local):
            print(f"[Local] Création du dossier manquant : {dossier_local}")
            os.makedirs(dossier_local, exist_ok=True)

        print(f"Téléchargement en cours : {chemin_distant} -> {chemin_local}...")
        sftp.get(chemin_distant, chemin_local)
        sftp.close()
        
        print(f"Fichier téléchargé avec succès dans {chemin_local}")
        return True
    except Exception as e:
        print(f"❌ Erreur lors du transfert SFTP (Download) : {e}")
        return False