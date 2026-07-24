#!/bin/bash
set -e # <-- AJOUTER ICI : Arrête le script au moindre problème

# 1. Dossier de destination finale
NODE_DATA_DIR="/tmp/$USER/coco_mini"
rm -rf "$NODE_DATA_DIR"
mkdir -p "$NODE_DATA_DIR"

TEMP_FULL_DIR="/tmp/$USER/coco_temp_full"
rm -rf "$TEMP_FULL_DIR"
mkdir -p "$TEMP_FULL_DIR"

echo "=== [MINI COCO] Téléchargement de la base source (val2017) ==="
wget -q -nc http://images.cocodataset.org/zips/val2017.zip -O "$TEMP_FULL_DIR/val2017.zip"
unzip -q -n "$TEMP_FULL_DIR/val2017.zip" -d "$TEMP_FULL_DIR/"
rm "$TEMP_FULL_DIR/val2017.zip"

wget -q -nc http://images.cocodataset.org/annotations/annotations_trainval2017.zip -O "$TEMP_FULL_DIR/annotations.zip"
unzip -q -n "$TEMP_FULL_DIR/annotations.zip" -d "$TEMP_FULL_DIR/"
rm "$TEMP_FULL_DIR/annotations.zip"

echo "=== [MINI COCO] Génération du sous-ensemble via le script Python ==="
python datasets_img/coco_mini/gene_coco_mini.py \
    --annotations_path "$TEMP_FULL_DIR/annotations/instances_val2017.json" \
    --images_dir "$TEMP_FULL_DIR/val2017" \
    --output_dir "$NODE_DATA_DIR" \
    --train_size 3000 \
    --val_size 1000 \
    --test_size 1000

echo "=== [MINI COCO] Nettoyage des gros fichiers sources temporaires ==="
rm -rf "$TEMP_FULL_DIR"

# Transmission du chemin
echo "$NODE_DATA_DIR" > "/tmp/$USER/.node_data_dir_path"
echo "=== Mini dataset prêt et enregistré dans $NODE_DATA_DIR ==="