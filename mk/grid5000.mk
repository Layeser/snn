# Grid5000 / OAR — réservation GPU

OAR_QUEUE ?= besteffort
OAR_PROJECT ?= interactivite

# Batch (make reserve-*) : soumission depuis la frontale, job en arrière-plan
WALLTIME ?= 2:00:00
OAR_GPU ?= gpu=1,besteffort

# Interactif (make interactive) : shell sur un nœud GPU, puis make train-*
OAR_INTERACTIVE ?= nodes=1/gpu=1
INTERACTIVE_WALLTIME ?= 1:00:00

# Commande oarsub interactive (depuis la frontale uniquement)
OARSUB_INTERACTIVE := oarsub -I -q $(OAR_QUEUE) -l $(OAR_INTERACTIVE),walltime=$(INTERACTIVE_WALLTIME) -p "$(OAR_PROJECT)"

# Commande oarsub batch (sans -I)
OARSUB_BATCH = oarsub -q $(OAR_QUEUE) -l $(OAR_GPU),walltime=$(WALLTIME) -p "$(OAR_PROJECT)"
