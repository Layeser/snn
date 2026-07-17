PROJECT := HPSTAtten
include ../mk/vars.mk
include ../mk/train.mk

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

grid-hp grid-hp-linear grid-statten grid-sdt: check-deps
	@mkdir -p $(SAVE_DIR)
	$(PYTHON) -m scripts.train --config $(CONFIG) --save-dir $(SAVE_DIR) \
		--data-dir $(DATA_DIR) --dataset $(DATASET) $(GRID_ARGS) $(EXTRA_ARGS)

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
	@echo "  Résultats : save/optuna/<study>/best_params.yml + optuna.db (versionné)"

grid-help:
	@echo "Grille d'ablation (attention_mode x hybrid_qkv):"
	@echo "  make grid-hp          factorized + hybride  (HP-STAtten, référence)"
	@echo "  make grid-hp-linear   sdt + hybride         (HP-STAtten-linéaire, Hadamard)"
	@echo "  make grid-statten     factorized + binaire  (STAtten)"
	@echo "  make grid-sdt         sdt + binaire         (SDT pur, mask+add)"
	@echo "  make grid-all         lance les 4 séquentiellement"
	@echo "  Grid5000 : reserve-grid-hp / -hp-linear / -statten / -sdt"
	@echo "  Override : make grid-hp-linear DATASET=cifar10-dvs"
	@echo "  Petit GPU: make grid-hp-linear DATASET=cifar10-dvs EXTRA_ARGS=\"--batch-size 8 --T 8\""
