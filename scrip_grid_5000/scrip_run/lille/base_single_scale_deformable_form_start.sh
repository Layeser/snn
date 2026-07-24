#!/bin/bash

## OAR_option -p gpu-16GB AND gpu_compute_capability_major>=5  ## déactive
# OAR_option -p gpu_compute_capability IS NOT NULL AND cpuarch = 'x86_64'
# OAR_option -q default
# OAR_option -l host=1/gpu=2

echo "=== ÉTAPE 1 : Préparation du Dataset ==="
source scrip_grid_5000/prepare_dataset_coco.sh

# Sécurité : On vérifie si le script du dataset s'est bien terminé
if [ $? -ne 0 ]; then
    echo "Erreur lors de la préparation du dataset. Arrêt de l'orchestre."
    exit 1
fi
# ==============================================================================
# GESTION AUTOMATIQUE DU DOSSIER DE SORTIE
# ==============================================================================
# On définit le nom du dossier ici. Tu peux changer ce nom pour chaque nouvelle expérience.
RUN_NAME="run_test_detr_deformable_single_scale_start_epoch_0"
export OUTPUT_DIR="$HOME/detr-projet/outputs/$RUN_NAME"

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

BATCH_SIZE=8  # Taille du batch pour l'entraînement

# 3. Lancement de l'entraînement Python
echo "--- [2/2] Lancement de l'entraînement ---"

torchrun \
    --nnodes="$NNODES" \
    --nproc_per_node=2 \
    --rdzv_id="$JOB_ID" \
    --rdzv_backend=c10d \
    --rdzv_endpoint="$MASTER_NODE:$PORT" \
    --rdzv_conf timeout=3600 \
    detr/main.py \
    --coco_path "$NODE_DATA_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --num_classes 91 \
    --epochs 50 \
    --attention_type Deformable \
    --pre_norm \
    --num_feature_levels 1 \
    --lr 2e-4 \
    --lr_backbone 2e-5 \
    --lr_backbone_names "backbone.0" \
    --lr_linear_proj_names "reference_points" "sampling_offsets" \
    --batch_size "$BATCH_SIZE" \
    --dim_feedforward 1024 \
    --num_queries 300 \
    --set_cost_class 2 \
    --cls_loss_coef 2 \
    --matcher_class_type focal \
    --with_norm_proj True \
    --input_proj_init xavier \
    --stat_box_init -2.0 \
    --use_early_stopping False \
    --deterministic False \
    --auto_resume

echo "--- Entraînement terminé ! ---"