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
BESTEFFORT_PILOT := $(SNN_ROOT)/grid5k/besteffort_pilot.py
BESTEFFORT_CONFIG := $(SNN_ROOT)/grid5k/config.yaml
BESTEFFORT_WATCH_INTERVAL ?= 600

MANUAL_SCRIPT := $(SNN_ROOT)/scrip_grid_5000/run_manual_site.sh
SYNC_SCRIP_SCRIPT := $(SNN_ROOT)/scrip_grid_5000/sync_scrip_run_local.sh
RESERVE_SCRIPT := $(SNN_ROOT)/scrip_grid_5000/reserve_manual.sh
RESERVE_SMOKE_CONFIG ?= $(SNN_ROOT)/scrip_grid_5000/reserve_smoke.yaml
BOOK_SMOKE_SCRIPT := $(SNN_ROOT)/scrip_grid_5000/book_smoke_local.sh
RESERVE_START ?=
RESERVE_END ?=
RESERVE_TAG ?= run

.PHONY: help job-status setup setup-venv setup-g5k list-python-modules check-deps print-python interactive \
	download-data download-cifar10 download-cifar10-dvs prepare-cifar10-dvs \
	g5k-auto g5k-auto-follow g5k-auto-watch g5k-auto-restart g5k-auto-clean g5k-fresh g5k-auto-smoke g5k-auto-smoke-watch g5k-test-auto g5k-test-auto-smoke \
	besteffort besteffort-watch besteffort-check besteffort-fresh \
	besteffort-lille besteffort-lyon besteffort-watch-lille besteffort-watch-lyon \
	besteffort-list besteffort-check-lille besteffort-check-lyon \
	g5k-book-lille g5k-book-lyon g5k-book-smoke g5k-book-smoke-check \
	g5k-run-lille g5k-run-lyon g5k-run-lille-scrip g5k-run-lyon-scrip \
	g5k-sync-scrip-run g5k-sync-scrip-run-check \
	g5k-restart-lille g5k-restart-lyon g5k-clean-manual \
	g5k-check-lille g5k-check-lyon g5k-check-lille-scrip g5k-check-lyon-scrip \
	g5k-test-chicoree g5k-test-chuc g5k-test-sirius \
	g5k-run-smoke-reserved-lille g5k-run-smoke-reserved-lyon \
	g5k-run-smoke-reserved-lille-check g5k-run-smoke-reserved-lyon-check \
	g5k-help \
	pilot-grid pilot-grid-watch pilot-grid-fresh pilot-grid-clean grid-watch \
	prepare-pilot-smoke pilot-grid-smoke pilot-grid-smoke-watch pilot-smoke \
	manual-reserve-lille manual-reserve-lyon manual-run-lille manual-run-lyon \
	manual-run-lille-fresh manual-run-lyon-fresh manual-fresh manual-dry-run-lille manual-dry-run-lyon \
	prepare-chicoree-smoke prepare-sirius-smoke \
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
	@echo "─── Grid'5000 — campagnes HP-STAtten (make g5k-help) ───"
	@echo ""
	@echo "  AUTO (local, file OAR nuit)     scrip_run/<site>/<cluster>/*.sh"
	@echo "    make g5k-auto                 soumettre tout (1 fois, puis PC peut s'éteindre)"
	@echo "    make g5k-auto-follow          suivre + rapatrier (sans resoumettre)"
	@echo "    make g5k-auto-restart         nettoyer + soumettre"
	@echo ""
	@echo "  MANUEL (frontale, créneau -r)   <cluster>_experiences/*.sh"
	@echo "    make g5k-book-lille  RESERVE_START=... RESERVE_END=...   # flille"
	@echo "    make g5k-book-lyon   RESERVE_START=... RESERVE_END=...   # flyon"
	@echo "    make g5k-run-lille                 # lancer au créneau"
	@echo "    make g5k-restart-lille             # nettoyer + lancer"
	@echo ""
	@echo "  BESTEFFORT (GPU quelconque, reprise auto)  besteffort_lille/  besteffort_lyon/"
	@echo "    make besteffort-watch             # 1 GPU/exp., relance toutes les 10 min"
	@echo "    make besteffort-watch-lyon          # Lyon seulement"
	@echo "    Doc : grid5k/README.md"
	@echo ""
	@echo ""
	@echo "Variables globales:"
	@echo "  DATA_DIR=$(DATA_DIR)  DATASET=$(DATASET)"
	@echo "  WALLTIME=$(WALLTIME)  INTERACTIVE_WALLTIME=$(INTERACTIVE_WALLTIME)"
	@echo "  OAR_GPU=$(OAR_GPU)  OAR_QUEUE=$(OAR_QUEUE)"
	@echo ""
	@echo "Aide détaillée par projet: cd spikformer && make help"

job-status:
	@oarstat -u $$USER 2>/dev/null || oarstat 2>/dev/null || echo "oarstat indisponible (hors Grid5000 ?)"

# =============================================================
# Grid'5000 — noms mémorables (préfixe g5k-)
#   auto   = file OAR nuit, depuis la machine locale
#   book   = réserver un créneau (-r), depuis la frontale
#   run    = lancer les files sur nœud, au créneau
#   restart = nettoyer + relancer
# =============================================================

g5k-help:
	@echo "Grid'5000 — commandes g5k-*"
	@echo ""
	@echo "AUTO (machine locale) — dossier scrip_run/<site>/<cluster>/"
	@echo "  g5k-auto              1 job OAR par dossier cluster (file GPU auto sur le nœud)"
	@echo "  g5k-auto-follow       suivre les jobs + rapatrier outputs/ (sans resoumettre)"
	@echo "  g5k-auto-restart      nettoyer local + frontales + soumettre"
	@echo "  g5k-auto-clean        nettoyer local seulement"
	@echo "  g5k-fresh             nettoyer local + frontales (git restore scrip_grid_5000/)"
	@echo ""
	@echo "  Les jobs OAR tournent sur Grid'5000 même PC éteint après g5k-auto."
	@echo "  g5k-auto-follow est optionnel (rapatriement local des résultats)."
	@echo ""
	@echo "MANUEL (frontale) — dossiers chicoree_experiences/, chuc_experiences/, sirius_experiences/"
	@echo "  g5k-book-lille        réserver chicorée + chuc  (flille)"
	@echo "  g5k-book-lyon         réserver sirius           (flyon)"
	@echo "  g5k-book-smoke        réserver smoke 3 clusters depuis le PC (reserve_smoke.yaml)"
	@echo "  g5k-run-lille         lancer les files Lille (*_experiences/)"
	@echo "  g5k-run-lille-scrip   lancer campagne réelle (scrip_run/ + job_id)"
	@echo "  g5k-sync-scrip-run    envoyer scrip_run/ local → frontales (PC)"
	@echo "  g5k-run-lyon          lancer la file Lyon"
	@echo "  g5k-restart-lille     g5k-fresh + lancer Lille"
	@echo "  g5k-restart-lyon      g5k-fresh + lancer Lyon"
	@echo "  g5k-clean-manual      alias g5k-fresh --local"
	@echo "  g5k-check-lille       afficher la file sans lancer"
	@echo ""
	@echo "Tests smoke (jour) :"
	@echo "  BOOK   make g5k-book-smoke              (PC local, reserve_smoke.yaml)"
	@echo "  AUTO   make g5k-test-auto-smoke"
	@echo "  MANUEL make g5k-test-chicoree | g5k-test-chuc | g5k-test-sirius  (flille / flyon)"
	@echo "  RESERVE make g5k-run-smoke-reserved-lille | g5k-run-smoke-reserved-lyon  (créneau -r, sans nouvel oarsub)"
	@echo ""
	@echo "Variables réservation : RESERVE_START  RESERVE_END  RESERVE_TAG"
	@echo "Doc : scrip_grid_5000/README.md"

# --- Mode auto ---

g5k-auto:
	PILOT_CONFIG=$(PILOT_CONFIG) $(PILOT_PYTHON) $(PILOT_MAIN) --config $(PILOT_CONFIG)

g5k-auto-follow:
	watch -n $(PILOT_WATCH_INTERVAL) "PILOT_CONFIG=$(PILOT_CONFIG) $(PILOT_PYTHON) $(PILOT_MAIN) --config $(PILOT_CONFIG) --follow-only"

g5k-auto-watch: g5k-auto-follow

g5k-auto-restart:
	bash $(SNN_ROOT)/scrip_grid_5000/g5k_fresh.sh
	$(MAKE) g5k-auto

g5k-auto-clean:
	bash $(SNN_ROOT)/scrip_grid_5000/g5k_fresh.sh --local

g5k-fresh:
	bash $(SNN_ROOT)/scrip_grid_5000/g5k_fresh.sh

g5k-test-auto:
	bash $(SNN_ROOT)/scrip_grid_5000/prepare_pilot_smoke.sh

g5k-test-auto-smoke: g5k-test-auto
	PILOT_CONFIG=$(PILOT_CONFIG_SMOKE) $(PILOT_PYTHON) $(PILOT_MAIN) --config $(PILOT_CONFIG_SMOKE)

g5k-auto-smoke:
	PILOT_CONFIG=$(PILOT_CONFIG_SMOKE) $(PILOT_PYTHON) $(PILOT_MAIN) --config $(PILOT_CONFIG_SMOKE)

g5k-auto-smoke-watch:
	watch -n $(PILOT_WATCH_INTERVAL) "PILOT_CONFIG=$(PILOT_CONFIG_SMOKE) $(PILOT_PYTHON) $(PILOT_MAIN) --config $(PILOT_CONFIG_SMOKE)"

# --- Besteffort (grid5k/ — indépendant de scrip_grid_5000) ---

besteffort:
	$(PILOT_PYTHON) $(BESTEFFORT_PILOT) --config $(BESTEFFORT_CONFIG) $(if $(BESTEFFORT_SITES),--sites $(BESTEFFORT_SITES),)

besteffort-watch:
	watch -n $(BESTEFFORT_WATCH_INTERVAL) "$(PILOT_PYTHON) $(BESTEFFORT_PILOT) --config $(BESTEFFORT_CONFIG) $(if $(BESTEFFORT_SITES),--sites $(BESTEFFORT_SITES),)"

besteffort-check:
	$(PILOT_PYTHON) $(BESTEFFORT_PILOT) --config $(BESTEFFORT_CONFIG) --follow-only $(if $(BESTEFFORT_SITES),--sites $(BESTEFFORT_SITES),)

besteffort-fresh:
	bash $(SNN_ROOT)/grid5k/fresh.sh

besteffort-lille:
	$(MAKE) besteffort BESTEFFORT_SITES=lille

besteffort-lyon:
	$(MAKE) besteffort BESTEFFORT_SITES=lyon

besteffort-watch-lille:
	$(MAKE) besteffort-watch BESTEFFORT_SITES=lille

besteffort-watch-lyon:
	$(MAKE) besteffort-watch BESTEFFORT_SITES=lyon

besteffort-check-lille:
	$(MAKE) besteffort-check BESTEFFORT_SITES=lille

besteffort-list:
	@echo "=== besteffort_lille/ ==="
	@ls -1 besteffort_lille/*.sh 2>/dev/null || echo "  (vide)"
	@echo ""
	@echo "=== besteffort_lyon/ ==="
	@ls -1 besteffort_lyon/*.sh 2>/dev/null || echo "  (vide)"

# --- Mode manuel ---

g5k-book-lille:
	@test -n "$(RESERVE_START)" && test -n "$(RESERVE_END)" || \
		(echo "Usage: make g5k-book-lille RESERVE_START='2026-08-04 19:00:00' RESERVE_END='2026-08-05 09:00:00' RESERVE_TAG=04"; exit 1)
	RESERVE_START="$(RESERVE_START)" RESERVE_END="$(RESERVE_END)" RESERVE_TAG="$(RESERVE_TAG)" \
		CHICOREE_GPU="$(CHICOREE_GPU)" CHUC_GPU="$(CHUC_GPU)" \
		bash $(RESERVE_SCRIPT) lille

g5k-book-lyon:
	@test -n "$(RESERVE_START)" && test -n "$(RESERVE_END)" || \
		(echo "Usage: make g5k-book-lyon RESERVE_START='2026-08-04 19:00:00' RESERVE_END='2026-08-05 09:00:00' RESERVE_TAG=04"; exit 1)
	RESERVE_START="$(RESERVE_START)" RESERVE_END="$(RESERVE_END)" RESERVE_TAG="$(RESERVE_TAG)" \
		SIRIUS_GPU="$(SIRIUS_GPU)" \
		bash $(RESERVE_SCRIPT) lyon

g5k-book-smoke:
	RESERVE_SMOKE_CONFIG="$(RESERVE_SMOKE_CONFIG)" bash $(BOOK_SMOKE_SCRIPT)

g5k-book-smoke-check:
	RESERVE_SMOKE_CONFIG="$(RESERVE_SMOKE_CONFIG)" bash $(BOOK_SMOKE_SCRIPT) --dry-run

g5k-run-lille:
	bash $(MANUAL_SCRIPT) lille

g5k-run-lille-scrip:
	bash $(MANUAL_SCRIPT) lille --scrip-run

g5k-run-lyon:
	bash $(MANUAL_SCRIPT) lyon

g5k-run-lyon-scrip:
	bash $(MANUAL_SCRIPT) lyon --scrip-run

g5k-sync-scrip-run:
	bash $(SYNC_SCRIP_SCRIPT)

g5k-sync-scrip-run-check:
	bash $(SYNC_SCRIP_SCRIPT) --dry-run

g5k-restart-lille: g5k-fresh
	$(MAKE) g5k-run-lille

g5k-restart-lyon: g5k-fresh
	$(MAKE) g5k-run-lyon

g5k-clean-manual:
	bash $(SNN_ROOT)/scrip_grid_5000/g5k_fresh.sh --local

g5k-check-lille:
	bash $(MANUAL_SCRIPT) lille --dry-run

g5k-check-lille-scrip:
	bash $(MANUAL_SCRIPT) lille --scrip-run --dry-run

g5k-check-lyon:
	bash $(MANUAL_SCRIPT) lyon --dry-run

g5k-check-lyon-scrip:
	bash $(MANUAL_SCRIPT) lyon --scrip-run --dry-run

g5k-test-chicoree:
	bash $(SNN_ROOT)/scrip_grid_5000/prepare_chicoree_smoke.sh

g5k-test-chuc:
	bash $(SNN_ROOT)/scrip_grid_5000/prepare_chuc_smoke.sh

g5k-test-sirius:
	bash $(SNN_ROOT)/scrip_grid_5000/prepare_sirius_smoke.sh

g5k-run-smoke-reserved-lille:
	bash $(SNN_ROOT)/scrip_grid_5000/run_smoke_reserved.sh lille

g5k-run-smoke-reserved-lyon:
	bash $(SNN_ROOT)/scrip_grid_5000/run_smoke_reserved.sh lyon

g5k-run-smoke-reserved-lille-check:
	bash $(SNN_ROOT)/scrip_grid_5000/run_smoke_reserved.sh lille --dry-run

g5k-run-smoke-reserved-lyon-check:
	bash $(SNN_ROOT)/scrip_grid_5000/run_smoke_reserved.sh lyon --dry-run

# --- Alias anciens noms (compatibilité) ---

pilot-grid: g5k-auto
pilot-grid-watch grid-watch: g5k-auto-watch
pilot-grid-fresh: g5k-auto-restart
pilot-grid-clean: g5k-auto-clean
prepare-pilot-smoke: g5k-test-auto
pilot-grid-smoke: g5k-auto-smoke
pilot-grid-smoke-watch: g5k-auto-smoke-watch
pilot-smoke: g5k-test-auto-smoke
manual-reserve-lille: g5k-book-lille
manual-reserve-lyon: g5k-book-lyon
manual-run-lille: g5k-run-lille
manual-run-lyon: g5k-run-lyon
manual-run-lille-fresh: g5k-restart-lille
manual-run-lyon-fresh: g5k-restart-lyon
manual-fresh: g5k-clean-manual
manual-dry-run-lille: g5k-check-lille
manual-dry-run-lyon: g5k-check-lyon
prepare-chicoree-smoke: g5k-test-chicoree
prepare-chuc-smoke: g5k-test-chuc
prepare-sirius-smoke: g5k-test-sirius

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
