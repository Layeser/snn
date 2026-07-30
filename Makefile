# Makefile racine — orchestration de tous les projets SNN
#
# Exemples:
#   make help
#   make train-spikformer
#   make reserve-hpstattn WALLTIME=4:00:00
#   make job-status

DATA_DIR ?= $(SNN_ROOT)/data
DATASET ?= cifar10

SNN_ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
include $(SNN_ROOT)/mk/grid5000.mk

PROJECTS := spikformer spikdrivenformer spatialtemporal A2OS2A HPSTAtten

RESERVE_TARGETS := $(addprefix reserve-,$(PROJECTS))
TRAIN_TARGETS := $(addprefix train-,$(PROJECTS))
FRESH_TARGETS := $(addprefix train-fresh-,$(PROJECTS))

VENV := $(SNN_ROOT)/.venv
include $(SNN_ROOT)/mk/python.mk
BOOTSTRAP_PYTHON ?= $(DETECTED_PYTHON)

PILOT_MAIN := $(SNN_ROOT)/scrip_grid_5000/pilot_grid/main.py
PILOT_CONFIG ?= $(SNN_ROOT)/scrip_grid_5000/pilot_grid/config.yaml
PILOT_CONFIG_SMOKE := $(SNN_ROOT)/scrip_grid_5000/pilot_grid/config_smoke.yaml
PILOT_PYTHON := $(if $(wildcard $(VENV)/bin/python),$(VENV)/bin/python,python3)
PILOT_WATCH_INTERVAL ?= 300

.PHONY: help job-status setup setup-venv setup-g5k list-python-modules check-deps print-python interactive \
	download-data download-cifar10 download-cifar10-dvs prepare-cifar10-dvs \
	pilot-grid pilot-grid-watch grid-watch \
	prepare-pilot-smoke pilot-grid-smoke pilot-grid-smoke-watch pilot-smoke \
	prepare-chicoree-smoke chicoree-smoke-test \
	$(RESERVE_TARGETS) $(TRAIN_TARGETS) $(FRESH_TARGETS) reserve-all train-all

help:
	@echo "SNN — commandes racine"
	@echo ""
	@echo "Environnement:"
	@echo "  make setup              Crée .venv (Python 3.10+, auto-détecté sur Grid5000)"
	@echo "  make download-data      Télécharge CIFAR-10 + CIFAR-10-DVS (miroirs rapides)"
	@echo "  make download-cifar10   Télécharge uniquement CIFAR-10"
	@echo "  make download-cifar10-dvs  Télécharge les archives CIFAR-10-DVS (parallèle)"
	@echo "  make prepare-cifar10-dvs   Convertit DVS en frames (~15–30 min, une fois)"
	@echo "  make setup-g5k          Idem (charge module $(G5K_PYTHON_MODULE) explicitement)"
	@echo "  make list-python-modules  Liste les modules python dispo (module avail)"
	@echo "  make check-deps         Vérifie Python 3.10+ et torch"
	@echo "  make print-python       Affiche l'interpréteur utilisé"
	@echo ""
	@echo "  Grid5000 : make setup   # détecte python/3.10.8 automatiquement"
	@echo "             module load $(G5K_PYTHON_MODULE)   # si besoin manuel"
	@echo "  Forcer :   make setup BOOTSTRAP_PYTHON=/chemin/vers/python3.10"
	@echo ""
	@echo "─── Grid5000 : deux modes ───"
	@echo ""
	@echo "1) INTERACTIF (debug, MLflow ui, tests rapides)"
	@echo "   Depuis la frontale :"
	@echo "     make interactive              # oarsub -I → shell sur nœud GPU"
	@echo "   Puis sur le nœud GPU :"
	@echo "     make train-spikformer         # lance l'entraînement ici"
	@echo "     cd spikformer && make logs"
	@echo ""
	@echo "2) BATCH (entraînement long, déconnexion OK)"
	@echo "   Depuis la frontale (sans session interactive) :"
	@echo "     make reserve-spikformer       # oarsub batch → job en arrière-plan"
	@echo "     make job-status               # suivre le job"
	@echo "     tail -f spikformer/save/run.out"
	@echo ""
	@echo "Entraînement (sur nœud GPU ou machine locale — pas depuis la frontale CPU):"
	@echo "  make train-spikformer DATASET=cifar10-dvs
	@echo "  make train-spikformer DATA_DIR=$(DATA_DIR) DATASET=$(DATASET)"
	@echo "  make train-spikdrivenformer"
	@echo "  make train-spatialtemporal"
	@echo "  make train-a2os2a"
	@echo "  make train-hpstattn"
	@echo "  make train-all"
	@echo ""
	@echo "From scratch:"
	@echo "  make train-fresh-spikformer"
	@echo ""
	@echo "Réservation batch (depuis la frontale):"
	@echo "  make reserve-spikformer"
	@echo "  make reserve-hpstattn"
	@echo "  make reserve-all"
	@echo ""
	@echo "Session GPU interactive (depuis la frontale):"
	@echo "  make interactive   WALLTIME=$(INTERACTIVE_WALLTIME) via INTERACTIVE_WALLTIME=..."
	@echo ""
	@echo "Suivi:"
	@echo "  make job-status"
	@echo "  cd spikformer && make logs"
	@echo ""
	@echo "Orchestrateur Grid5000 (local, scripts dans scrip_grid_5000/scrip_run/):"
	@echo "  make pilot-grid              # une tournée (config prod, nuit)"
	@echo "  make pilot-grid-watch        # relance toutes les $(PILOT_WATCH_INTERVAL)s (Ctrl+C pour arrêter)"
	@echo "  make grid-watch              # alias de pilot-grid-watch"
	@echo "  make pilot-grid-watch PILOT_WATCH_INTERVAL=600"
	@echo ""
	@echo "Smoke test orchestrateur (jour, walltime 10 min, 2 epochs, petites données) :"
	@echo "  make prepare-pilot-smoke     # 3 scripts test (Lille chicoree/chuc + Lyon sirius)"
	@echo "  git push                     # configs smoke sur Grid5000"
	@echo "  make pilot-grid-smoke        # soumet avec config_smoke.yaml"
	@echo "  make pilot-grid-smoke-watch  # suivre jusqu'à récupération"
	@echo "  make pilot-smoke             # prepare-pilot-smoke + pilot-grid-smoke"
	@echo ""
	@echo "Test file chicorée (smoke, day 15 min, 6 jobs → 4 GPU + queue) :"
	@echo "  make prepare-chicoree-smoke  # 6 scripts dans chicoree_experiences/"
	@echo "  make chicoree-smoke-test     # rappel des commandes flille"
	@echo ""
	@echo "Variables globales:"
	@echo "  DATA_DIR=$(DATA_DIR)  DATASET=$(DATASET)"
	@echo "  WALLTIME=$(WALLTIME)  INTERACTIVE_WALLTIME=$(INTERACTIVE_WALLTIME)"
	@echo "  OAR_GPU=$(OAR_GPU)  OAR_QUEUE=$(OAR_QUEUE)"
	@echo ""
	@echo "Aide détaillée par projet: cd spikformer && make help"

job-status:
	@oarstat -u $$USER 2>/dev/null || oarstat 2>/dev/null || echo "oarstat indisponible (hors Grid5000 ?)"

# Orchestrateur pilot_grid : lancer depuis la racine du repo (machine locale).
pilot-grid:
	PILOT_CONFIG=$(PILOT_CONFIG) $(PILOT_PYTHON) $(PILOT_MAIN) --config $(PILOT_CONFIG)

pilot-grid-watch:
	watch -n $(PILOT_WATCH_INTERVAL) "PILOT_CONFIG=$(PILOT_CONFIG) $(PILOT_PYTHON) $(PILOT_MAIN) --config $(PILOT_CONFIG)"

prepare-pilot-smoke:
	bash $(SNN_ROOT)/scrip_grid_5000/prepare_pilot_smoke.sh

pilot-grid-smoke:
	PILOT_CONFIG=$(PILOT_CONFIG_SMOKE) $(PILOT_PYTHON) $(PILOT_MAIN) --config $(PILOT_CONFIG_SMOKE)

pilot-grid-smoke-watch:
	watch -n $(PILOT_WATCH_INTERVAL) "PILOT_CONFIG=$(PILOT_CONFIG_SMOKE) $(PILOT_PYTHON) $(PILOT_MAIN) --config $(PILOT_CONFIG_SMOKE)"

pilot-smoke: prepare-pilot-smoke pilot-grid-smoke

prepare-chicoree-smoke:
	bash $(SNN_ROOT)/scrip_grid_5000/prepare_chicoree_smoke.sh

chicoree-smoke-test: prepare-chicoree-smoke
	@echo ""
	@echo "=== Réservation day 15 min (flille) — choisir UNE option ==="
	@echo ""
	@echo "A) Interactif (debug, recommandé pour le 1er test) :"
	@echo "   oarsub -I -p chicoree -t exotic -t day \\"
	@echo "     -l host=1/gpu=4,walltime=0:15:00 -q default"
	@echo "   # sur le nœud :"
	@echo "   cd ~/internship/snn && bash scrip_grid_5000/run_chicoree_queue.sh"
	@echo ""
	@echo "B) Batch (sleep + connexion) :"
	@echo "   oarsub -p chicoree -t exotic -t day \\"
	@echo "     -l host=1/gpu=4,walltime=0:15:00 -q default \\"
	@echo "     -n chicoree_smoke_test -- /bin/sleep 999999"
	@echo "   bash scrip_grid_5000/run_chicoree_queue.sh --job-id <JOB_ID>"
	@echo ""
	@echo "Suivi pendant le test :"
	@echo "   watch -n 3 nvidia-smi"
	@echo "   tail -f outputs/chicoree_queue/scheduler.log"
	@echo "   ls scrip_grid_5000/chicoree_experiences/archive/done/"

# Alias court pour la surveillance continue de l'orchestrateur.
grid-watch: pilot-grid-watch

# Réservation interactive GPU (frontale → shell sur nœud, puis make train-*)
interactive:
	@echo "→ Session interactive GPU ($(OAR_INTERACTIVE), $(INTERACTIVE_WALLTIME))"
	@echo "  Une fois connecté au nœud : make train-spikformer"
	$(OARSUB_INTERACTIVE)

setup: setup-venv

DOWNLOAD_PYTHON := $(if $(wildcard $(VENV)/bin/python),$(VENV)/bin/python,$(BOOTSTRAP_PYTHON))

download-data:
	$(DOWNLOAD_PYTHON) $(SNN_ROOT)/scripts/download_data.py all --data-dir $(DATA_DIR)

download-cifar10:
	$(DOWNLOAD_PYTHON) $(SNN_ROOT)/scripts/download_data.py cifar10 --data-dir $(DATA_DIR)

download-cifar10-dvs:
	$(DOWNLOAD_PYTHON) $(SNN_ROOT)/scripts/download_data.py cifar10-dvs --data-dir $(DATA_DIR) --workers 4

prepare-cifar10-dvs:
	$(DOWNLOAD_PYTHON) $(SNN_ROOT)/scripts/download_data.py cifar10-dvs --data-dir $(DATA_DIR) --workers 4 --prepare-frames --frames 4

setup-g5k:
	@$(MAKE) setup BOOTSTRAP_PYTHON=$$(bash -lc 'module load $(G5K_PYTHON_MODULE) && command -v python3')

list-python-modules:
	@module avail python 2>&1 | head -40 || echo "module non disponible"

setup-venv:
	@echo "Bootstrap Python: $(BOOTSTRAP_PYTHON)"
	@$(BOOTSTRAP_PYTHON) --version 2>&1 || true
	@$(BOOTSTRAP_PYTHON) -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null || ( \
		echo "Erreur: Python 3.10+ requis pour créer le venv (actuel: $$($(BOOTSTRAP_PYTHON) --version 2>&1))."; \
		echo "  Grid5000 : make setup          # auto-détecte $(G5K_PYTHON_MODULE)"; \
		echo "  Ou       : module load $(G5K_PYTHON_MODULE) && make setup"; \
		echo "  Ou       : make list-python-modules"; \
		exit 1)
	$(BOOTSTRAP_PYTHON) -m venv $(VENV)
	$(VENV)/bin/pip install -U pip
	$(VENV)/bin/pip install -r $(SNN_ROOT)/requirements.txt
	@echo "Environnement prêt: $(VENV)/bin/python ($$($(VENV)/bin/python --version))"

check-deps:
	@$(MAKE) -C spikformer check-deps

print-python:
	@$(MAKE) -C spikformer print-python

define PROJECT_RULES
train-$(1):
	$$(MAKE) -C $(1) train DATA_DIR=$$(DATA_DIR) DATASET=$$(DATASET) WALLTIME=$$(WALLTIME) \
		OAR_GPU=$$(OAR_GPU)

train-fresh-$(1):
	$$(MAKE) -C $(1) train-fresh DATA_DIR=$$(DATA_DIR) DATASET=$$(DATASET)

reserve-$(1):
	$$(MAKE) -C $(1) reserve DATA_DIR=$$(DATA_DIR) DATASET=$$(DATASET) WALLTIME=$$(WALLTIME) \
		OAR_GPU=$$(OAR_GPU)
endef

$(foreach p,$(PROJECTS),$(eval $(call PROJECT_RULES,$(p))))

train-all: $(TRAIN_TARGETS)

reserve-all: $(RESERVE_TARGETS)

# Alias courts (projet HP-STAtten = hpstattn)
train-a2os2a: train-A2OS2A
train-fresh-a2os2a: train-fresh-A2OS2A
reserve-a2os2a: reserve-A2OS2A

train-hpstattn: train-HPSTAtten
train-fresh-hpstattn: train-fresh-HPSTAtten
reserve-hpstattn: reserve-HPSTAtten
