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
# Grille d'ablation attention : attention_mode x hybrid_qkv (4×2 = 8 cases)
#   grid-hp                  factorized + hybride  -> HP-STAtten (référence)
#   grid-hp-linear           sdt        + hybride  -> HP-STAtten-linéaire (Hadamard)
#   grid-contrast            contrast   + hybride  -> VCA-light (agrégation KᵀV)
#   grid-contrast-sdt        contrast_sdt + hybride -> VCA-light (agrégation Σ K⊙V)
#   grid-statten             factorized + binaire  -> STAtten
#   grid-sdt                 sdt        + binaire  -> SDT pur (mask + add)
#   grid-contrast-binary     contrast   + binaire  -> contrast binaire
#   grid-contrast-sdt-binary contrast_sdt + binaire -> contrast_sdt binaire
# Chaque variante a son propre SAVE_DIR (checkpoints/logs séparés).
# Override dataset/params : make grid-hp-linear DATASET=cifar10-dvs
# ============================================================================
.PHONY: grid-hp grid-hp-linear grid-contrast grid-contrast-sdt grid-statten grid-sdt \
        grid-contrast-binary grid-contrast-sdt-binary grid-all grid-fresh-all grid-help \
        grid-fresh-hp grid-fresh-hp-linear grid-fresh-contrast grid-fresh-contrast-sdt \
        grid-fresh-statten grid-fresh-sdt grid-fresh-contrast-binary grid-fresh-contrast-sdt-binary \
        reserve-grid-hp reserve-grid-hp-linear reserve-grid-contrast reserve-grid-contrast-sdt \
        reserve-grid-statten reserve-grid-sdt reserve-grid-contrast-binary reserve-grid-contrast-sdt-binary

GRID_DIR := $(CURDIR)/save/grid/$(DATASET)
# CIFAR-10 grille : HP t27 + aug complète (cifar10_grid_sota*.yml) ; DVS → train_cifar10-dvs.yml
EMBED_DIM ?=
ifeq ($(DATASET),cifar10)
ifeq ($(EMBED_DIM),512)
GRID_CONFIG := config/campaigns/cifar10_grid_sota_512.yml
else
GRID_CONFIG := config/campaigns/cifar10_grid_sota.yml
endif
else
GRID_CONFIG := $(CONFIG)
endif
# Args supplémentaires passés à scripts.train (ex: petit GPU local)
#   make grid-hp-linear DATASET=cifar10-dvs EXTRA_ARGS="--batch-size 8 --T 8"
#   make grid-statten DATASET=cifar10 EMBED_DIM=512
EXTRA_ARGS ?=

grid-hp:              SAVE_DIR := $(GRID_DIR)/hp_factorized_hybrid
grid-hp:              GRID_ARGS := --attention-mode factorized --hybrid-qkv true
grid-hp-linear:       SAVE_DIR := $(GRID_DIR)/hp_sdt_hybrid
grid-hp-linear:       GRID_ARGS := --attention-mode sdt --hybrid-qkv true
grid-contrast:        SAVE_DIR := $(GRID_DIR)/hp_contrast_hybrid
grid-contrast:        GRID_ARGS := --attention-mode contrast --hybrid-qkv true
grid-contrast-sdt:    SAVE_DIR := $(GRID_DIR)/hp_contrast_sdt_hybrid
grid-contrast-sdt:    GRID_ARGS := --attention-mode contrast_sdt --hybrid-qkv true --vct-num 16
grid-statten:         SAVE_DIR := $(GRID_DIR)/statten_factorized_binary
grid-statten:         GRID_ARGS := --attention-mode factorized --hybrid-qkv false
grid-sdt:             SAVE_DIR := $(GRID_DIR)/sdt_binary
grid-sdt:             GRID_ARGS := --attention-mode sdt --hybrid-qkv false
grid-contrast-binary: SAVE_DIR := $(GRID_DIR)/contrast_binary
grid-contrast-binary: GRID_ARGS := --attention-mode contrast --hybrid-qkv false --vct-num 16
grid-contrast-sdt-binary: SAVE_DIR := $(GRID_DIR)/contrast_sdt_binary
grid-contrast-sdt-binary: GRID_ARGS := --attention-mode contrast_sdt --hybrid-qkv false --vct-num 16
grid-factorized-hgr:  SAVE_DIR := $(GRID_DIR)/factorized_hgr_hybrid
grid-factorized-hgr:  GRID_ARGS := --attention-mode factorized_hgr --hybrid-qkv true
grid-mk-hgr:          SAVE_DIR := $(GRID_DIR)/mk_hgr_hybrid
grid-mk-hgr:          GRID_ARGS := --attention-mode mk_hgr --hybrid-qkv true --mk-dual-scale true
grid-mk-hgr-triple:   SAVE_DIR := $(GRID_DIR)/mk_hgr_triple_hybrid
grid-mk-hgr-triple:   GRID_ARGS := --attention-mode mk_hgr --hybrid-qkv true --mk-dual-scale false

define run_grid_train
	@mkdir -p $(SAVE_DIR)
	$(PYTHON) -m scripts.train --config $(GRID_CONFIG) --save-dir $(SAVE_DIR) \
		--data-dir $(DATA_DIR) --dataset $(DATASET) $(GRID_ARGS) $(1) $(EXTRA_ARGS)
endef

grid-hp grid-hp-linear grid-contrast grid-contrast-sdt grid-statten grid-sdt \
grid-contrast-binary grid-contrast-sdt-binary grid-factorized-hgr grid-mk-hgr grid-mk-hgr-triple: check-deps
	$(call run_grid_train,)

grid-fresh-hp:              SAVE_DIR := $(GRID_DIR)/hp_factorized_hybrid
grid-fresh-hp:              GRID_ARGS := --attention-mode factorized --hybrid-qkv true
grid-fresh-hp-linear:       SAVE_DIR := $(GRID_DIR)/hp_sdt_hybrid
grid-fresh-hp-linear:       GRID_ARGS := --attention-mode sdt --hybrid-qkv true
grid-fresh-contrast:        SAVE_DIR := $(GRID_DIR)/hp_contrast_hybrid
grid-fresh-contrast:        GRID_ARGS := --attention-mode contrast --hybrid-qkv true --vct-num 16
grid-fresh-contrast-sdt:    SAVE_DIR := $(GRID_DIR)/hp_contrast_sdt_hybrid
grid-fresh-contrast-sdt:    GRID_ARGS := --attention-mode contrast_sdt --hybrid-qkv true --vct-num 16
grid-fresh-statten:         SAVE_DIR := $(GRID_DIR)/statten_factorized_binary
grid-fresh-statten:         GRID_ARGS := --attention-mode factorized --hybrid-qkv false
grid-fresh-sdt:             SAVE_DIR := $(GRID_DIR)/sdt_binary
grid-fresh-sdt:             GRID_ARGS := --attention-mode sdt --hybrid-qkv false
grid-fresh-contrast-binary: SAVE_DIR := $(GRID_DIR)/contrast_binary
grid-fresh-contrast-binary: GRID_ARGS := --attention-mode contrast --hybrid-qkv false --vct-num 16
grid-fresh-contrast-sdt-binary: SAVE_DIR := $(GRID_DIR)/contrast_sdt_binary
grid-fresh-contrast-sdt-binary: GRID_ARGS := --attention-mode contrast_sdt --hybrid-qkv false --vct-num 16
grid-fresh-factorized-hgr:  SAVE_DIR := $(GRID_DIR)/factorized_hgr_hybrid
grid-fresh-factorized-hgr:  GRID_ARGS := --attention-mode factorized_hgr --hybrid-qkv true
grid-fresh-mk-hgr:          SAVE_DIR := $(GRID_DIR)/mk_hgr_hybrid
grid-fresh-mk-hgr:          GRID_ARGS := --attention-mode mk_hgr --hybrid-qkv true --mk-dual-scale true

grid-fresh-hp grid-fresh-hp-linear grid-fresh-contrast grid-fresh-contrast-sdt \
grid-fresh-statten grid-fresh-sdt grid-fresh-contrast-binary grid-fresh-contrast-sdt-binary \
grid-fresh-factorized-hgr grid-fresh-mk-hgr: check-deps
	$(call run_grid_train,--fresh)

grid-all: grid-hp grid-hp-linear grid-contrast grid-contrast-sdt grid-statten grid-sdt \
          grid-contrast-binary grid-contrast-sdt-binary

grid-fresh-all: grid-fresh-hp grid-fresh-hp-linear grid-fresh-contrast grid-fresh-contrast-sdt \
                grid-fresh-statten grid-fresh-sdt grid-fresh-contrast-binary grid-fresh-contrast-sdt-binary

# --- Grid5000 (batch OAR, un job par variante) ---
reserve-grid-hp:              SAVE_DIR := $(GRID_DIR)/hp_factorized_hybrid
reserve-grid-hp-linear:       SAVE_DIR := $(GRID_DIR)/hp_sdt_hybrid
reserve-grid-contrast:        SAVE_DIR := $(GRID_DIR)/hp_contrast_hybrid
reserve-grid-contrast-sdt:    SAVE_DIR := $(GRID_DIR)/hp_contrast_sdt_hybrid
reserve-grid-statten:         SAVE_DIR := $(GRID_DIR)/statten_factorized_binary
reserve-grid-sdt:             SAVE_DIR := $(GRID_DIR)/sdt_binary
reserve-grid-contrast-binary: SAVE_DIR := $(GRID_DIR)/contrast_binary
reserve-grid-contrast-sdt-binary: SAVE_DIR := $(GRID_DIR)/contrast_sdt_binary

reserve-grid-hp:
	$(call write_oar_job_script,grid-hp)
	$(call oarsub_batch_job,grid-hp,$(OAR_JOB_NAME)_hp)

reserve-grid-hp-linear:
	$(call write_oar_job_script,grid-hp-linear)
	$(call oarsub_batch_job,grid-hp-linear,$(OAR_JOB_NAME)_hplin)

reserve-grid-contrast:
	$(call write_oar_job_script,grid-contrast)
	$(call oarsub_batch_job,grid-contrast,$(OAR_JOB_NAME)_contrast)

reserve-grid-contrast-sdt:
	$(call write_oar_job_script,grid-contrast-sdt)
	$(call oarsub_batch_job,grid-contrast-sdt,$(OAR_JOB_NAME)_csdt)

reserve-grid-statten:
	$(call write_oar_job_script,grid-statten)
	$(call oarsub_batch_job,grid-statten,$(OAR_JOB_NAME)_statten)

reserve-grid-sdt:
	$(call write_oar_job_script,grid-sdt)
	$(call oarsub_batch_job,grid-sdt,$(OAR_JOB_NAME)_sdt)

reserve-grid-contrast-binary:
	$(call write_oar_job_script,grid-contrast-binary)
	$(call oarsub_batch_job,grid-contrast-binary,$(OAR_JOB_NAME)_cbin)

reserve-grid-contrast-sdt-binary:
	$(call write_oar_job_script,grid-contrast-sdt-binary)
	$(call oarsub_batch_job,grid-contrast-sdt-binary,$(OAR_JOB_NAME)_csdtbin)

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
	@echo "  + archi   : make tune DATASET=cifar10 TUNE_ARGS=\"--tune-arch\""
	@echo "  + aug/batch: make tune DATASET=cifar10 CONFIG=config/campaigns/cifar10_grid_sota.yml TUNE_ARGS=\"--tune-aug --tune-batch --tune-arch\""
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
	@echo "Grille d'ablation 4×2 (attention_mode x hybrid_qkv):"
	@echo "  make grid-hp                    factorized + hybride      (HP-STAtten, référence)"
	@echo "  make grid-hp-linear             sdt + hybride           (Hadamard linéaire)"
	@echo "  make grid-contrast              contrast + hybride      (VCA, agrégation KᵀV)"
	@echo "  make grid-contrast-sdt          contrast_sdt + hybride  (VCA, agrégation Σ K⊙V)"
	@echo "  make grid-statten               factorized + binaire    (STAtten)"
	@echo "  make grid-sdt                   sdt + binaire           (SDT pur)"
	@echo "  make grid-contrast-binary       contrast + binaire"
	@echo "  make grid-contrast-sdt-binary   contrast_sdt + binaire"
	@echo "  make grid-fresh-all             8 runs from scratch (--fresh), séquentiel"
	@echo "  make grid-all                   8 runs (reprise auto si last.pt)"
	@echo "  Config auto : DATASET=cifar10 → cifar10_grid_sota.yml (HP t27 + aug) | cifar10-dvs → train_cifar10-dvs.yml"
	@echo "  embed 512   : make grid-statten DATASET=cifar10 EMBED_DIM=512"
	@echo "  Checkpoints : save/grid/<DATASET>/<variante>/  (cifar10 et cifar10-dvs séparés)"
	@echo "  Petit GPU: make grid-contrast EXTRA_ARGS=\"--batch-size 8 --T 8 --vct-num 16\""
	@echo ""
	@echo "Ablations complexité (factorized + hybrid_qkv, hors grille 4×2) :"
	@echo "  make complexity-baseline        HP-STAtten global (référence campagne)"
	@echo "  make complexity-win4            fenêtre locale 4×4"
	@echo "  make complexity-win4-shift      fenêtre 4×4 + shifted window"
	@echo "  make complexity-rank8           low-rank r=8 sur mixage KᵀV"
	@echo "  make complexity-rank16          low-rank r=16"
	@echo "  make complexity-nystrom16       Nyström 16 landmarks"
	@echo "  make complexity-win4-rank8      combiné fenêtre + low-rank"
	@echo "  make complexity-fresh-all       7 runs from scratch (--fresh)"
	@echo "  make complexity-all             7 runs (reprise auto)"
	@echo "  Checkpoints : save/complexity_ablation/<DATASET>/<variante>/"
	@echo "  MLflow : HP-STAtten-ComplexityAblation-<CIFAR10|CIFAR10-DVS>"

# ============================================================================
# Ablations complexité (local window, low-rank, Nyström) — HP baseline
# Référence : Notes/hpstatten_complexity_ablations.md
# ============================================================================
.PHONY: complexity-baseline complexity-win4 complexity-win4-shift complexity-rank8 \
        complexity-rank16 complexity-nystrom16 complexity-win4-rank8 \
        complexity-all complexity-fresh-all complexity-help \
        complexity-fresh-baseline complexity-fresh-win4 complexity-fresh-win4-shift \
        complexity-fresh-rank8 complexity-fresh-rank16 complexity-fresh-nystrom16 \
        complexity-fresh-win4-rank8

COMPLEXITY_DIR := $(CURDIR)/save/complexity_ablation/$(DATASET)
COMPLEXITY_BASE := --attention-mode factorized --hybrid-qkv true

complexity-baseline:       SAVE_DIR := $(COMPLEXITY_DIR)/baseline
complexity-baseline:       COMPLEXITY_ARGS := $(COMPLEXITY_BASE)
complexity-win4:           SAVE_DIR := $(COMPLEXITY_DIR)/win4
complexity-win4:           COMPLEXITY_ARGS := $(COMPLEXITY_BASE) --window-size 4
complexity-win4-shift:     SAVE_DIR := $(COMPLEXITY_DIR)/win4_shift
complexity-win4-shift:     COMPLEXITY_ARGS := $(COMPLEXITY_BASE) --window-size 4 --window-shift true
complexity-rank8:          SAVE_DIR := $(COMPLEXITY_DIR)/rank8
complexity-rank8:          COMPLEXITY_ARGS := $(COMPLEXITY_BASE) --mix-rank 8
complexity-rank16:         SAVE_DIR := $(COMPLEXITY_DIR)/rank16
complexity-rank16:         COMPLEXITY_ARGS := $(COMPLEXITY_BASE) --mix-rank 16
complexity-nystrom16:      SAVE_DIR := $(COMPLEXITY_DIR)/nystrom16
complexity-nystrom16:      COMPLEXITY_ARGS := $(COMPLEXITY_BASE) --num-landmarks 16
complexity-win4-rank8:     SAVE_DIR := $(COMPLEXITY_DIR)/win4_rank8
complexity-win4-rank8:     COMPLEXITY_ARGS := $(COMPLEXITY_BASE) --window-size 4 --mix-rank 8

define run_complexity_train
	@mkdir -p $(SAVE_DIR)
	$(PYTHON) -m scripts.train --config $(CONFIG) --save-dir $(SAVE_DIR) \
		--data-dir $(DATA_DIR) --dataset $(DATASET) $(COMPLEXITY_ARGS) $(1) $(EXTRA_ARGS)
endef

complexity-baseline complexity-win4 complexity-win4-shift complexity-rank8 complexity-rank16 \
complexity-nystrom16 complexity-win4-rank8: check-deps
	$(call run_complexity_train,)

complexity-fresh-baseline:       SAVE_DIR := $(COMPLEXITY_DIR)/baseline
complexity-fresh-baseline:       COMPLEXITY_ARGS := $(COMPLEXITY_BASE)
complexity-fresh-win4:           SAVE_DIR := $(COMPLEXITY_DIR)/win4
complexity-fresh-win4:           COMPLEXITY_ARGS := $(COMPLEXITY_BASE) --window-size 4
complexity-fresh-win4-shift:     SAVE_DIR := $(COMPLEXITY_DIR)/win4_shift
complexity-fresh-win4-shift:     COMPLEXITY_ARGS := $(COMPLEXITY_BASE) --window-size 4 --window-shift true
complexity-fresh-rank8:          SAVE_DIR := $(COMPLEXITY_DIR)/rank8
complexity-fresh-rank8:          COMPLEXITY_ARGS := $(COMPLEXITY_BASE) --mix-rank 8
complexity-fresh-rank16:         SAVE_DIR := $(COMPLEXITY_DIR)/rank16
complexity-fresh-rank16:         COMPLEXITY_ARGS := $(COMPLEXITY_BASE) --mix-rank 16
complexity-fresh-nystrom16:      SAVE_DIR := $(COMPLEXITY_DIR)/nystrom16
complexity-fresh-nystrom16:      COMPLEXITY_ARGS := $(COMPLEXITY_BASE) --num-landmarks 16
complexity-fresh-win4-rank8:     SAVE_DIR := $(COMPLEXITY_DIR)/win4_rank8
complexity-fresh-win4-rank8:     COMPLEXITY_ARGS := $(COMPLEXITY_BASE) --window-size 4 --mix-rank 8

complexity-fresh-baseline complexity-fresh-win4 complexity-fresh-win4-shift complexity-fresh-rank8 \
complexity-fresh-rank16 complexity-fresh-nystrom16 complexity-fresh-win4-rank8: check-deps
	$(call run_complexity_train,--fresh)

complexity-all: complexity-baseline complexity-win4 complexity-win4-shift complexity-rank8 \
                complexity-rank16 complexity-nystrom16 complexity-win4-rank8

complexity-fresh-all: complexity-fresh-baseline complexity-fresh-win4 complexity-fresh-win4-shift \
                      complexity-fresh-rank8 complexity-fresh-rank16 complexity-fresh-nystrom16 \
                      complexity-fresh-win4-rank8

complexity-help:
	@echo "Ablations complexité HP-STAtten (factorized + hybrid_qkv) :"
	@echo "  make complexity-all DATASET=cifar10       7 variantes séquentielles"
	@echo "  make complexity-fresh-all                 idem, from scratch"
	@echo "  make complexity-rank8                     low-rank r=8 (priorité campagne)"
	@echo "  Voir aussi : make grid-help"
