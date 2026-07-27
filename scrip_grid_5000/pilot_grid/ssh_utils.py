import os

import paramiko

# Valeur de repli si aucune passerelle SSH n'est fournie par la configuration.
SSH_GATEWAY_DEFAUT = "access.grid5000.fr"

def connecter_serveur_final(username, final_host, ssh_gateway=SSH_GATEWAY_DEFAUT):
    """
    Se connecte a la passerelle SSH puis rebondit sur le serveur final choisi.
    Retourne les objets clients pour pouvoir exécuter des commandes et fermer la session.
    """
    try:
        # 1. Connexion au serveur de rebond (passerelle SSH)
        print(f"Connexion à la passerelle SSH {ssh_gateway}...")
        bastion = paramiko.SSHClient()
        bastion.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        bastion.connect(hostname=ssh_gateway, username=username)

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

def extraire_options_oar(ssh_client, chemin_distant_script):
    """
    Lit le script bash directement SUR LE SERVEUR DISTANT et extrait 
    toutes les lignes commençant par '# OAR_option'.
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