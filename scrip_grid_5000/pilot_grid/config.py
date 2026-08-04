import os
from dataclasses import dataclass, field

import yaml

CHEMIN_CONFIG_DEFAUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")


@dataclass
class Config:
    """Regroupe tout ce qui est specifique a une personne / installation.

    Aucune valeur specifique a un utilisateur ne doit rester en dur dans le
    code : tout passe par ce dataclass, alimente depuis config.yaml.
    """

    # Identite / connexion
    user: str
    ssh_gateway: str = "access.grid5000.fr"

    # Chemins distants (sur Grid'5000)
    remote_project_dir: str = "detr-projet"
    remote_start_script: str = "scrip_grid_5000/start_run.sh"

    # Chemins locaux
    local_scripts_root: str = "scrip_grid_5000/scrip_run"
    local_outputs_dir: str = "outputs"
    state_file: str = "scrip_grid_5000/pilot_grid/run_status.json"

    # Orchestration
    sites: list = field(default_factory=lambda: ["lille"])
    max_jobs_per_site: int = 2
    cluster_defaults_file: str = "cluster_defaults.yaml"
    # per_cluster : 1 job OAR par dossier cluster (file GPU sur le noeud)
    # per_script  : 1 job OAR par .sh (besteffort, reprise apres preemption)
    submission_mode: str = "per_cluster"
    # Racines par site (ex. besteffort_lille / besteffort_lyon). Vide = local_scripts_root/<site>/
    site_scripts_root: dict = field(default_factory=dict)

    # Options OAR par defaut (prioritaires sur les # OAR_option des scripts)
    walltime: str = "10:00:00"
    oar_type: str = "night"  # day | night | besteffort | ...
    oar_queue: str = ""  # default | besteffort | vide = pas de -q explicite
    oar_resources: str = "host=1/gpu=1"  # ressources -l (sans walltime)

    # Synchronisation git (Option A : pull sur le frontend avant chaque soumission)
    git_enabled: bool = True
    git_branch: str = "main"
    git_repo: str = ""  # URL optionnelle : clone auto si le projet est absent

    # --- Helpers de chemins distants -------------------------------------
    def remote_project_home(self) -> str:
        """Chemin projet via $HOME (le shell distant developpe $HOME).

        ex: $HOME/detr-projet
        """
        return f"$HOME/{self.remote_project_dir}"

    def remote_project_abs(self) -> str:
        """Chemin projet absolu (pour construire des chemins stockes).

        ex: /home/tgroussa/detr-projet
        """
        return f"/home/{self.user}/{self.remote_project_dir}"

    def remote_start_script_path(self) -> str:
        """Chemin du script de lancement OAR sur le serveur (via $HOME)."""
        return f"{self.remote_project_home()}/{self.remote_start_script}"

    def remote_script_path(self, relatif_local: str) -> str:
        """Chemin distant d'un script, en miroir de son chemin local.

        Le repertoire distant reproduit l'arborescence locale sous le
        dossier du projet. ex: scrip_grid_5000/scrip_run/lille/x.sh
        -> /home/tgroussa/detr-projet/scrip_grid_5000/scrip_run/lille/x.sh
        """
        return f"{self.remote_project_abs()}/{relatif_local}"


def load_config(chemin: str = None) -> Config:
    """Charge la configuration depuis un fichier YAML.

    Ordre de priorite du chemin :
      1. argument `chemin`
      2. variable d'environnement PILOT_CONFIG
      3. config.yaml a cote de ce module
    """
    chemin = chemin or os.environ.get("PILOT_CONFIG") or CHEMIN_CONFIG_DEFAUT

    if not os.path.exists(chemin):
        raise FileNotFoundError(
            f"Fichier de configuration introuvable : {chemin}\n"
            f"Copiez/editez config.yaml puis renseignez au moins le champ 'user'."
        )

    with open(chemin, "r") as f:
        donnees = yaml.safe_load(f) or {}

    champs_connus = {f.name for f in Config.__dataclass_fields__.values()}
    inconnus = set(donnees) - champs_connus
    if inconnus:
        raise ValueError(
            f"Cle(s) inconnue(s) dans {chemin} : {sorted(inconnus)}.\n"
            f"Cles autorisees : {sorted(champs_connus)}"
        )

    if not donnees.get("user"):
        raise ValueError(f"Le champ 'user' est obligatoire dans {chemin}.")

    return Config(**donnees)
