PROJECT := HPSTAtten

# Config par dataset (override possible : make train CONFIG=... DATASET=...)
ifeq ($(DATASET),cifar10-dvs)
CONFIG := config/train_cifar10-dvs.yml
else
CONFIG := config/train_cifar10.yml
endif

include ../mk/vars.mk
include ../mk/train.mk

config-help:
	@echo "Configs HP-STAtten (choix auto via DATASET=)"
	@echo "  DATASET=cifar10      → $(CONFIG)  (T=4, tet_loss=false, augment RGB)"
	@echo "  DATASET=cifar10-dvs  → config/train_cifar10-dvs.yml  (T=16, tet_loss=true, DVS aug)"
	@echo "  Fichiers : config/train_cifar10.yml | config/train_cifar10-dvs.yml"
	@echo "  Override : make train CONFIG=config/train_cifar10-dvs.yml DATASET=cifar10-dvs"

# ============================================================================
# Grille d'ablation attention : attention_mode x hybrid_qkv (4 cases)
#   grid-hp         factorized + hybride  -> HP-STAtten (référence, défaut)
#   grid-hp-linear  sdt        + hybride  -> HP-STAtten-linéaire (Hadamard)
#   grid-statten    factorized + binaire  -> STAtten
#   grid-sdt        sdt        + binaire  -> SDT pur (mask + add)
# Chaque variante a son propre SAVE_DIR (checkpoints/logs séparés).
# Override dataset/params : make grid-hp-linear DATASET=cifar10-dvs
# ============================================================================
.PHONY: grid-hp grid-hp-linear grid-statten grid-sdt grid-all grid-help \
        grid-fresh-hp grid-fresh-hp-linear grid-fresh-statten grid-fresh-sdt \
        reserve-grid-hp reserve-grid-hp-linear reserve-grid-statten reserve-grid-sdt

GRID_DIR := $(CURDIR)/save/grid
# Args supplémentaires passés à scripts.train (ex: petit GPU local)
#   make grid-hp-linear DATASET=cifar10-dvs EXTRA_ARGS="--batch-size 8 --T 8"
EXTRA_ARGS ?=

grid-hp:        SAVE_DIR := $(GRID_DIR)/hp_factorized_hybrid
grid-hp:        GRID_ARGS := --attention-mode factorized --hybrid-qkv true
grid-hp-linear: SAVE_DIR := $(GRID_DIR)/hp_sdt_hybrid
grid-hp-linear: GRID_ARGS := --attention-mode sdt --hybrid-qkv true
grid-statten:   SAVE_DIR := $(GRID_DIR)/statten_factorized_binary
grid-statten:   GRID_ARGS := --attention-mode factorized --hybrid-qkv false
grid-sdt:       SAVE_DIR := $(GRID_DIR)/sdt_binary
grid-sdt:       GRID_ARGS := --attention-mode sdt --hybrid-qkv false

define run_grid_train
	@mkdir -p $(SAVE_DIR)
	$(PYTHON) -m scripts.train --config $(CONFIG) --save-dir $(SAVE_DIR) \
		--data-dir $(DATA_DIR) --dataset $(DATASET) $(GRID_ARGS) $(1) $(EXTRA_ARGS)
endef

grid-hp grid-hp-linear grid-statten grid-sdt: check-deps
	$(call run_grid_train,)

grid-fresh-hp:        SAVE_DIR := $(GRID_DIR)/hp_factorized_hybrid
grid-fresh-hp:        GRID_ARGS := --attention-mode factorized --hybrid-qkv true
grid-fresh-hp-linear: SAVE_DIR := $(GRID_DIR)/hp_sdt_hybrid
grid-fresh-hp-linear: GRID_ARGS := --attention-mode sdt --hybrid-qkv true
grid-fresh-statten:   SAVE_DIR := $(GRID_DIR)/statten_factorized_binary
grid-fresh-statten:   GRID_ARGS := --attention-mode factorized --hybrid-qkv false
grid-fresh-sdt:       SAVE_DIR := $(GRID_DIR)/sdt_binary
grid-fresh-sdt:       GRID_ARGS := --attention-mode sdt --hybrid-qkv false

grid-fresh-hp grid-fresh-hp-linear grid-fresh-statten grid-fresh-sdt: check-deps
	$(call run_grid_train,--fresh)

grid-all: grid-hp grid-hp-linear grid-statten grid-sdt

# --- Grid5000 (batch OAR, un job par variante) ---
reserve-grid-hp:        SAVE_DIR := $(GRID_DIR)/hp_factorized_hybrid
reserve-grid-hp-linear: SAVE_DIR := $(GRID_DIR)/hp_sdt_hybrid
reserve-grid-statten:   SAVE_DIR := $(GRID_DIR)/statten_factorized_binary
reserve-grid-sdt:       SAVE_DIR := $(GRID_DIR)/sdt_binary

reserve-grid-hp:
	$(call write_oar_job_script,grid-hp)
	$(call oarsub_batch_job,grid-hp,$(OAR_JOB_NAME)_hp)

reserve-grid-hp-linear:
	$(call write_oar_job_script,grid-hp-linear)
	$(call oarsub_batch_job,grid-hp-linear,$(OAR_JOB_NAME)_hplin)

reserve-grid-statten:
	$(call write_oar_job_script,grid-statten)
	$(call oarsub_batch_job,grid-statten,$(OAR_JOB_NAME)_statten)

reserve-grid-sdt:
	$(call write_oar_job_script,grid-sdt)
	$(call oarsub_batch_job,grid-sdt,$(OAR_JOB_NAME)_sdt)

# ============================================================================
# Optuna : optimisation d'hyperparamètres
#   make tune DATASET=cifar10-dvs N_TRIALS=30 TUNE_EPOCHS=30
#   Petit GPU : make tune DATASET=cifar10-dvs TUNE_ARGS="--batch-size 8 --T 8"
# ============================================================================
.PHONY: tune tune-help reserve-tune

N_TRIALS    ?= 20
TUNE_EPOCHS ?= 30
TUNE_ARGS   ?=
TUNE_SAVE   := $(CURDIR)/save/optuna

tune: check-deps
	@mkdir -p $(TUNE_SAVE)
	$(PYTHON) -m scripts.tune --config $(CONFIG) --data-dir $(DATA_DIR) \
		--dataset $(DATASET) --save-dir $(TUNE_SAVE) \
		--n-trials $(N_TRIALS) --tune-epochs $(TUNE_EPOCHS) $(TUNE_ARGS)

reserve-tune: SAVE_DIR := $(TUNE_SAVE)
reserve-tune:
	@mkdir -p $(SAVE_DIR)
	@printf '%s\n' \
		'#!/bin/bash' \
		'set -euo pipefail' \
		'$(G5K_MODULE_LOAD)' \
		'cd "$(PROJECT_DIR)"' \
		'exec $(MAKE) tune DATA_DIR="$(DATA_DIR)" DATASET="$(DATASET)" N_TRIALS="$(N_TRIALS)" TUNE_EPOCHS="$(TUNE_EPOCHS)" TUNE_ARGS="$(TUNE_ARGS)"' \
		> "$(SAVE_DIR)/run_tune.sh"
	@chmod +x "$(SAVE_DIR)/run_tune.sh"
	$(OARSUB_BATCH) \
		-O "$(SAVE_DIR)/tune.out" \
		-E "$(SAVE_DIR)/tune.err" \
		-n "$(OAR_JOB_NAME)_tune" \
		-- "$(SAVE_DIR)/run_tune.sh"

tune-help:
	@echo "Optuna (optimisation d'hyperparamètres):"
	@echo "  make tune DATASET=cifar10-dvs N_TRIALS=30 TUNE_EPOCHS=30"
	@echo "  Petit GPU : make tune DATASET=cifar10-dvs TUNE_ARGS=\"--batch-size 8 --T 8\""
	@echo "  + archi   : make tune DATASET=cifar10-dvs TUNE_ARGS=\"--tune-arch\""
	@echo "  Grid5000  : make reserve-tune DATASET=cifar10-dvs N_TRIALS=50"
	@echo "  Reprise auto des essais interrompus (défaut) ; TUNE_ARGS=\"--no-resume-interrupted\" pour désactiver"
	@echo "  Résultats : save/optuna/<study>/best_params.yml + optuna.db (versionné)"

# ============================================================================
# Ablation schedulers LR (30 epochs, demande tuteur)
#   Reprise auto : relancer la même cible après Ctrl+C / préemption besteffort
#   Variantes terminées (30/30) sont ignorées au prochain ablation-lr-all
#   make ablation-lr-all DATASET=cifar10
#   make ablation-lr-cosine-warmup          # une variante
#   make ablation-fresh-lr-cosine-warmup    # from scratch
#   make reserve-ablation-lr-all            # Grid5000 batch (frontale)
# ============================================================================
.PHONY: ablation-lr-help ablation-lr-all \
        ablation-lr-cosine-warmup ablation-lr-cosine-annealing \
        ablation-lr-plateau-1e3 ablation-lr-plateau-005 ablation-lr-step \
        ablation-fresh-lr-cosine-warmup ablation-fresh-lr-cosine-annealing \
        ablation-fresh-lr-plateau-1e3 ablation-fresh-lr-plateau-005 ablation-fresh-lr-step \
        ablation-fresh-lr-all reserve-ablation-lr-all

ABLATION_LR_DIR := $(CURDIR)/save/ablation/lr
ABLATION_ARGS   ?=

define run_ablation_lr
	@mkdir -p $(SAVE_DIR)
	$(PYTHON) -m scripts.train --config $(CONFIG) --save-dir $(SAVE_DIR) \
		--data-dir $(DATA_DIR) --dataset $(DATASET) $(1) $(ABLATION_ARGS)
endef

ablation-lr-cosine-warmup:     CONFIG := config/ablation/cosine_warmup.yml
ablation-lr-cosine-warmup:     SAVE_DIR := $(ABLATION_LR_DIR)/cosine_warmup
ablation-lr-cosine-annealing:  CONFIG := config/ablation/cosine_annealing.yml
ablation-lr-cosine-annealing:  SAVE_DIR := $(ABLATION_LR_DIR)/cosine_annealing
ablation-lr-plateau-1e3:       CONFIG := config/ablation/plateau_1e3.yml
ablation-lr-plateau-1e3:       SAVE_DIR := $(ABLATION_LR_DIR)/plateau_1e3
ablation-lr-plateau-005:       CONFIG := config/ablation/plateau_005.yml
ablation-lr-plateau-005:       SAVE_DIR := $(ABLATION_LR_DIR)/plateau_005
ablation-lr-step:              CONFIG := config/ablation/step.yml
ablation-lr-step:              SAVE_DIR := $(ABLATION_LR_DIR)/step

ablation-lr-cosine-warmup ablation-lr-cosine-annealing ablation-lr-plateau-1e3 \
ablation-lr-plateau-005 ablation-lr-step: check-deps
	$(call run_ablation_lr,)

ablation-fresh-lr-cosine-warmup:     CONFIG := config/ablation/cosine_warmup.yml
ablation-fresh-lr-cosine-warmup:     SAVE_DIR := $(ABLATION_LR_DIR)/cosine_warmup
ablation-fresh-lr-cosine-annealing:  CONFIG := config/ablation/cosine_annealing.yml
ablation-fresh-lr-cosine-annealing:  SAVE_DIR := $(ABLATION_LR_DIR)/cosine_annealing
ablation-fresh-lr-plateau-1e3:       CONFIG := config/ablation/plateau_1e3.yml
ablation-fresh-lr-plateau-1e3:       SAVE_DIR := $(ABLATION_LR_DIR)/plateau_1e3
ablation-fresh-lr-plateau-005:       CONFIG := config/ablation/plateau_005.yml
ablation-fresh-lr-plateau-005:       SAVE_DIR := $(ABLATION_LR_DIR)/plateau_005
ablation-fresh-lr-step:              CONFIG := config/ablation/step.yml
ablation-fresh-lr-step:              SAVE_DIR := $(ABLATION_LR_DIR)/step

ablation-fresh-lr-cosine-warmup ablation-fresh-lr-cosine-annealing ablation-fresh-lr-plateau-1e3 \
ablation-fresh-lr-plateau-005 ablation-fresh-lr-step: check-deps
	$(call run_ablation_lr,--fresh)

ablation-lr-all: ablation-lr-cosine-warmup ablation-lr-cosine-annealing \
	ablation-lr-plateau-1e3 ablation-lr-plateau-005 ablation-lr-step

ablation-fresh-lr-all: ablation-fresh-lr-cosine-warmup ablation-fresh-lr-cosine-annealing \
	ablation-fresh-lr-plateau-1e3 ablation-fresh-lr-plateau-005 ablation-fresh-lr-step

reserve-ablation-lr-all: SAVE_DIR := $(ABLATION_LR_DIR)
reserve-ablation-lr-all:
	@mkdir -p $(SAVE_DIR)
	@printf '%s\n' \
		'#!/bin/bash' \
		'set -euo pipefail' \
		'$(G5K_MODULE_LOAD)' \
		'cd "$(PROJECT_DIR)"' \
		'exec $(MAKE) ablation-lr-all DATA_DIR="$(DATA_DIR)" DATASET="$(DATASET)" ABLATION_ARGS="$(ABLATION_ARGS)"' \
		> "$(SAVE_DIR)/run_ablation_lr.sh"
	@chmod +x "$(SAVE_DIR)/run_ablation_lr.sh"
	$(OARSUB_BATCH) \
		-O "$(SAVE_DIR)/ablation_lr.out" \
		-E "$(SAVE_DIR)/ablation_lr.err" \
		-n "$(OAR_JOB_NAME)_abl_lr" \
		-- "$(SAVE_DIR)/run_ablation_lr.sh"

ablation-lr-help:
	@echo "Ablation schedulers (30 epochs, AdamW + LR schedule):"
	@echo "  make ablation-lr-all DATASET=cifar10     lance les 5 variantes (reprise auto)"
	@echo "  make ablation-lr-cosine-warmup         baseline cosine+warmup (lr=5e-4 wd=0.06)"
	@echo "  make ablation-lr-cosine-annealing      CosineAnnealingLR (lr=3e-4 wd=5e-4)"
	@echo "  make ablation-lr-plateau-1e3           ReduceLROnPlateau threshold=1e-3"
	@echo "  make ablation-lr-plateau-005           ReduceLROnPlateau threshold=0.005"
	@echo "  make ablation-lr-step                  StepLR step=10 gamma=0.5 (30 ep.)"
	@echo "  make ablation-fresh-lr-*               idem, from scratch (--fresh)"
	@echo "  make reserve-ablation-lr-all             Grid5000 batch depuis la frontale"
	@echo "  Reprise : relancer la même cible → last.pt (epoch suivant)"
	@echo "  Résultats : save/ablation/lr/<variante>/metrics.jsonl + MLflow"

grid-help:
	@echo "Grille d'ablation (attention_mode x hybrid_qkv):"
	@echo "  make grid-hp          factorized + hybride  (HP-STAtten, référence)"
	@echo "  make grid-hp-linear   sdt + hybride         (HP-STAtten-linéaire, Hadamard)"
	@echo "  make grid-statten     factorized + binaire  (STAtten)"
	@echo "  make grid-sdt         sdt + binaire         (SDT pur, mask+add)"
	@echo "  make grid-fresh-*     idem, from scratch (--fresh, ignore last.pt)"
	@echo "  make grid-all         lance les 4 séquentiellement"
	@echo "  Config auto : DATASET=cifar10 → train_cifar10.yml | cifar10-dvs → train_cifar10-dvs.yml"
	@echo "  Override : make grid-hp-linear DATASET=cifar10-dvs"
	@echo "  Petit GPU: make grid-hp-linear DATASET=cifar10-dvs EXTRA_ARGS=\"--batch-size 8 --T 8\""
