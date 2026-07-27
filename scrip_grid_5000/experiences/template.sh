#!/bin/bash
# =============================================================
# GABARIT d'experience pour l'orchestrateur Grid'5000.
#
# Utilisation :
#   1. Copiez ce fichier (ex: dans experiences/cifar10/mon_exp.sh)
#   2. Adaptez les 3 zones ci-dessous (OAR / identite / commande)
#   3. Pour LANCER : copiez le .sh dans scrip_grid_5000/scrip_run/
#      puis lancez l'orchestrateur (python pilot_grid/main.py)
#
# CONTRAT avec l'orchestrateur (NE PAS casser le format) :
#   - les lignes commencant par '# OAR_option' -> ressources OAR
#   - une ligne 'RUN_NAME="..."'                -> nom du run
#   - une ligne 'OUTPUT_DIR=...'                 -> dossier resultats
#     (c'est ce dossier qui sera archive puis rapatrie en local)
# =============================================================

# ---- 1) Ressources OAR (adapter selon le besoin) ----
# OAR_option -l host=1/gpu=1
# OAR_option -q besteffort

# ---- 2) Identite du run (OBLIGATOIRE, garder ce format) ----
RUN_NAME="mon_experience"
export OUTPUT_DIR="$HOME/snn/HPSTAtten/save/$RUN_NAME"
mkdir -p "$OUTPUT_DIR"

# ---- 3) Commande d'entrainement (adapter a votre situation) ----
# Contexte a l'execution :
#   - CWD = racine du projet
#   - le venv (.venv) est deja active par start_run.sh
cd HPSTAtten
python -m scripts.train \
    --config config/train_cifar10.yml \
    --dataset cifar10 \
    --save-dir "$OUTPUT_DIR"
