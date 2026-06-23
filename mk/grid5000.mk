# Grid5000 / OAR — réservation GPU (site Lille)
#
# -p  = contrainte SQL sur les ressources (PAS un nom de projet)
# --project = nom de projet OAR (optionnel)
# Ne pas utiliser -p "interactivite" sur Lille (colonne inexistante).

OAR_QUEUE ?= besteffort

# Contrainte SQL optionnelle : make reserve-spikformer OAR_PROPERTY='gpu_model="V100"'
OAR_PROPERTY ?=

# Nom de projet OAR optionnel : make reserve-spikformer OAR_PROJECT_NAME=mon_projet
OAR_PROJECT_NAME ?=

# Batch (make reserve-*) : soumission depuis la frontale, job en arrière-plan
WALLTIME ?= 2:00:00
OAR_GPU ?= nodes=1/gpu=1

# Interactif (make interactive) : shell sur un nœud GPU, puis make train-*
OAR_INTERACTIVE ?= nodes=1/gpu=1
INTERACTIVE_WALLTIME ?= 1:00:00

OAR_EXTRA_FLAGS = \
	$(if $(OAR_PROPERTY),-p "$(OAR_PROPERTY)",) \
	$(if $(OAR_PROJECT_NAME),--project=$(OAR_PROJECT_NAME),)

# Commande oarsub interactive (depuis la frontale uniquement)
OARSUB_INTERACTIVE := oarsub -I -q $(OAR_QUEUE) -l $(OAR_INTERACTIVE),walltime=$(INTERACTIVE_WALLTIME) $(OAR_EXTRA_FLAGS)

# Commande oarsub batch (sans -I) — -O/-E stdout/stderr, -n nom du job
OARSUB_BATCH = oarsub -q $(OAR_QUEUE) -l $(OAR_GPU),walltime=$(WALLTIME) $(OAR_EXTRA_FLAGS)
