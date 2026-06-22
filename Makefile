# Makefile racine — orchestration de tous les projets SNN
#
# Exemples:
#   make help
#   make train-spikformer
#   make reserve-hpstattn WALLTIME=4:00:00
#   make job-status

DATA_DIR ?= $(HOME)/internship/snn/data
WALLTIME ?= 2:00:00
OAR_PROJECT ?= interactivite
OAR_GPU ?= gpu=1,besteffort

PROJECTS := spikformer spikdrivenformer spatialtemporal A2OS2A HPSTAtten

RESERVE_TARGETS := $(addprefix reserve-,$(PROJECTS))
TRAIN_TARGETS := $(addprefix train-,$(PROJECTS))
FRESH_TARGETS := $(addprefix train-fresh-,$(PROJECTS))

SNN_ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
VENV := $(SNN_ROOT)/.venv
include $(SNN_ROOT)/mk/python.mk
BOOTSTRAP_PYTHON ?= $(DETECTED_PYTHON)

.PHONY: help job-status setup setup-venv setup-g5k list-python-modules check-deps print-python $(RESERVE_TARGETS) $(TRAIN_TARGETS) $(FRESH_TARGETS) \
	reserve-all train-all

help:
	@echo "SNN — commandes racine"
	@echo ""
	@echo "Environnement:"
	@echo "  make setup              Crée .venv (Python 3.10+, auto-détecté sur Grid5000)"
	@echo "  make setup-g5k          Idem (charge module $(G5K_PYTHON_MODULE) explicitement)"
	@echo "  make list-python-modules  Liste les modules python dispo (module avail)"
	@echo "  make check-deps         Vérifie Python 3.10+ et torch"
	@echo "  make print-python       Affiche l'interpréteur utilisé"
	@echo ""
	@echo "  Grid5000 : make setup   # détecte python/3.10.8 automatiquement"
	@echo "             module load $(G5K_PYTHON_MODULE)   # si besoin manuel"
	@echo "  Forcer :   make setup BOOTSTRAP_PYTHON=/chemin/vers/python3.10"
	@echo ""
	@echo "Entraînement local (reprise auto depuis save/last.pt):"
	@echo "  make train-spikformer"
	@echo "  make train-spikdrivenformer"
	@echo "  make train-spatialtemporal"
	@echo "  make train-a2os2a"
	@echo "  make train-hpstattn"
	@echo "  make train-all"
	@echo ""
	@echo "From scratch:"
	@echo "  make train-fresh-spikformer   (idem pour chaque projet)"
	@echo ""
	@echo "Réservation Grid5000 (oarsub, GPU besteffort):"
	@echo "  make reserve-spikformer"
	@echo "  make reserve-hpstattn"
	@echo "  make reserve-all"
	@echo ""
	@echo "Variantes oarsub (depuis le sous-projet):"
	@echo "  cd HPSTAtten && make reserve-fresh"
	@echo "  cd HPSTAtten && make reserve-resume"
	@echo ""
	@echo "Suivi:"
	@echo "  make job-status"
	@echo "  cd spikformer && make logs"
	@echo ""
	@echo "Variables globales:"
	@echo "  DATA_DIR=$(DATA_DIR)"
	@echo "  WALLTIME=$(WALLTIME)  OAR_GPU=$(OAR_GPU)  OAR_PROJECT=$(OAR_PROJECT)"
	@echo ""
	@echo "Aide détaillée par projet: cd spikformer && make help"

job-status:
	@oarstat -u $$USER 2>/dev/null || oarstat 2>/dev/null || echo "oarstat indisponible (hors Grid5000 ?)"

setup: setup-venv

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
	$$(MAKE) -C $(1) train DATA_DIR=$$(DATA_DIR) WALLTIME=$$(WALLTIME) \
		OAR_PROJECT=$$(OAR_PROJECT) OAR_GPU=$$(OAR_GPU)

train-fresh-$(1):
	$$(MAKE) -C $(1) train-fresh DATA_DIR=$$(DATA_DIR)

reserve-$(1):
	$$(MAKE) -C $(1) reserve DATA_DIR=$$(DATA_DIR) WALLTIME=$$(WALLTIME) \
		OAR_PROJECT=$$(OAR_PROJECT) OAR_GPU=$$(OAR_GPU)
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
