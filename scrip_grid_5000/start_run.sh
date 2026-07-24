#!/bin/bash
# ex commande run : oarsub -S "./start_run scrip/ex_train_model.sh"

# Sécurité : On vérifie l'argument
if [ -z "$1" ]; then
    echo "Erreur : Tu dois donner le nom du script d'entraînement en argument !"
    exit 1
fi

SCRIPT_ENTRAINEMENT=$1

# ==============================================================================
# CONFIGURATION MULTI-NODE (GRID'5000 / OAR)
# ==============================================================================
# 1. Récupérer la liste des machines uniques attribuées par OAR
NODES=$(uniq "$OAR_NODE_FILE")

# 2. Définir le "Master Node"
export MASTER_NODE=$(echo "$NODES" | head -n 1)

# 3. Compter le nombre total de machines (hosts)
export NNODES=$(echo "$NODES" | wc -l)

# 4. Définir le port et l'ID du job
export PORT=29500
export JOB_ID=$OAR_JOB_ID


# ==============================================================================
# GESTION DYNAMIQUE ET SÉCURISÉE DU PORT DE COMMUNICATION
# ==============================================================================
# 2. Boucle de vérification : tant que le port est détecté comme "occupé"
while ss -tln | grep -q ":$PORT " 2>/dev/null; do
    echo "⚠️ Le port $PORT est déjà utilisé par un autre utilisateur."
    
    # Génération d'un port aléatoire sécurisé entre 20000 et 60000
    export PORT=$(( 20000 + RANDOM % 40000 ))
    echo "🔄 Tentative de repli sur le port aléatoire : $PORT"
done

echo "✅ Port de communication validé et libre : $PORT"


echo "=== CONFIGURATION MULTI-NODE DU CLUSTER ==="
echo "Master Node     : $MASTER_NODE"
echo "Nombre de hosts : $NNODES"
echo "==========================================="

# On boucle sur chaque machine pour lancer l'entraînement
for NODE in $NODES; do
    echo "-> Configuration de l'environnement et lancement sur : $NODE"
    
    # On ouvre un bloc multi-ligne propre pour oarsh
    oarsh "$NODE" "
        export MASTER_NODE='$MASTER_NODE'
        export NNODES='$NNODES'
        export PORT='$PORT'
        export JOB_ID='$JOB_ID'
        export SCRIPT_ENTRAINEMENT='$SCRIPT_ENTRAINEMENT'

        # ======================================================================
        # ACTIVATION DE L'ENVIRONNEMENT (Exécuté sur chaque hôte)
        # ======================================================================
        #source /etc/profile.d/modules.sh
        cd \$HOME/detr-projet || exit 1
        source .venv/bin/activate

        # Lancement effectif du script d'entraînement
        bash \$SCRIPT_ENTRAINEMENT
    " &
done

# TRÈS IMPORTANT : On attend que toutes les machines en arrière-plan finissent
wait
echo "=== Tout le cluster a terminé l'entraînement ! ==="