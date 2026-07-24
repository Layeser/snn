#!/bin/bash
## OAR_option -p gpu-16GB AND gpu_compute_capability_major>=5  ## déactive
# OAR_option -p gpu_compute_capability IS NOT NULL AND cpuarch = 'x86_64'
# OAR_option -q default
# OAR_option -l host=1,gpu=2


echo "=== ÉTAPE 1 : Préparation du Dataset ==="
source scrip/prepare_dataset_coco.sh

# Sécurité : On vérifie si le script du dataset s'est bien terminé
if [ $? -ne 0 ]; then
    echo "Erreur lors de la préparation du dataset. Arrêt de l'orchestre."
    exit 1
fi

# ==============================================================================
# GESTION AUTOMATIQUE DU DOSSIER DE SORTIE
# ==============================================================================
# On définit le nom du dossier ici.
RUN_NAME="run_test_detr_$(date +%Y-%m-%d_%H-%M-%S)"
export OUTPUT_DIR="$HOME/detr-project/outputs/$RUN_NAME"

# On crée le dossier obligatoirement avant de lancer le python
mkdir -p "$OUTPUT_DIR"
echo "Les résultats seront enregistrés dans : $OUTPUT_DIR"

echo "=== ÉTAPE 2 : Lancement de l'Entraînement ==="
# Récupération du chemin du dataset généré par le script précédent
if [ -f "/tmp/$USER/.node_data_dir_path" ]; then
    NODE_DATA_DIR=$(cat "/tmp/$USER/.node_data_dir_path")
    echo "Chemin du dataset récupéré : $NODE_DATA_DIR"
else
    echo "Erreur : Impossible de trouver le fichier .node_data_dir_path !"
    exit 1
fi

BATCH_SIZE = 16  

# 3. Lancement de l'entraînement Python
echo "--- [2/2] Lancement de l'entraînement ---"

torchrun \
    --nnodes="$NNODES" \
    --nproc_per_node=gpu \
    --rdzv_id="$JOB_ID" \
    --rdzv_backend=c10d \
    --rdzv_endpoint="$MASTER_NODE:$PORT" \
    --rdzv_conf timeout=3600 \
    detr/main.py \
    --coco_path "$NODE_DATA_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --num_classes 91 \
    --auto_resume \
    --resume "" \
    --deterministic False \
    --batch_size "$BATCH_SIZE" \
    --dilation \
    --dim_feedforward 1024 \
    --dropout 0.0 \
    --num_queries 300 \
    --spatial_prior learned \
    --attention_type RCDA \
    --set_cost_class 2 \
    --cls_loss_coef 2 \
    --with_norm_proj True \
    --input_proj_init xavier \
    --matcher_class_type focal

echo "--- Entraînement terminé ! ---"