#!/bin/bash

NODE_DATA_DIR="/tmp/$USER/coco2017"
mkdir -p "$NODE_DATA_DIR"

echo "Copie du dataset en cours vers $NODE_DATA_DIR..."
# ==========================================
# GESTION DU DATASET (COCO 2017)
# ==========================================
echo "--- [1/2] Téléchargement et extraction de COCO 2017 dans $NODE_DATA_DIR ---"

# A. Set de Validation
# On télécharge dans le dossier utilisateur pour éviter les conflits de permissions
wget -q -nc http://images.cocodataset.org/zips/val2017.zip -O "$NODE_DATA_DIR/val2017.zip"
unzip -q -n "$NODE_DATA_DIR/val2017.zip" -d "$NODE_DATA_DIR/"
# On supprime le zip immédiatement pour libérer de l'espace RAM/Disque !
rm "$NODE_DATA_DIR/val2017.zip" 

# B. Annotations
wget -q -nc http://images.cocodataset.org/annotations/annotations_trainval2017.zip -O "$NODE_DATA_DIR/annotations.zip"
unzip -q -n "$NODE_DATA_DIR/annotations.zip" -d "$NODE_DATA_DIR/"
rm "$NODE_DATA_DIR/annotations.zip"

# C. Set de Train (18 Go)
wget -q -nc http://images.cocodataset.org/zips/train2017.zip -O "$NODE_DATA_DIR/train2017.zip"
unzip -q -n "$NODE_DATA_DIR/train2017.zip" -d "$NODE_DATA_DIR/"
rm "$NODE_DATA_DIR/train2017.zip"

echo "--- Dataset COCO 2017 prêt ! ---"

# ==========================================
# PRÉPARATION DU DOSSIER DE SORTIE
# ==========================================

#On écrit le chemin dans un fichier temporaire caché
echo "$NODE_DATA_DIR" > "/tmp/$USER/.node_data_dir_path"

echo "Dataset prêt sur le nœud."