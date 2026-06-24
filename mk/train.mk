.PHONY: help train train-fresh train-resume train-from-last reserve reserve-fresh reserve-resume logs metrics job-status smoke check-deps print-python

help:
	@echo "Projet: $(PROJECT)"
	@echo ""
	@echo "Entraînement local:"
	@echo "  make train              Reprend depuis save/last.pt si présent (défaut)"
	@echo "  make train-fresh        Repart de zéro (--fresh)"
	@echo "  make train-resume       Reprise explicite (--resume auto)"
	@echo "  make train-from-last    Reprise depuis save/last.pt"
	@echo ""
	@echo "─── Grid5000 ───"
	@echo "  make train / train-*   Sur le nœud GPU (après make interactive)"
	@echo "  make reserve           Batch depuis la frontale (oarsub sans -I)"
	@echo "  make reserve-fresh     Batch, from scratch"
	@echo "  make reserve-resume    Batch, reprise explicite"
	@echo ""
	@echo "Suivi:"
	@echo "  make logs               tail -f save/train.log"
	@echo "  make metrics            tail -f save/metrics.jsonl"
	@echo "  make job-status         oarstat (jobs en cours)"
	@echo ""
	@echo "Environnement:"
	@echo "  make check-deps         Vérifie torch pour PYTHON courant"
	@echo "  make print-python       Affiche l'interpréteur sélectionné"
	@echo ""
	@echo "Variables: DATA_DIR=$(DATA_DIR)  DATASET=$(DATASET)"
	@echo "           SAVE_DIR=$(SAVE_DIR)"
	@echo "           PYTHON=$(PYTHON)"
	@echo "           WALLTIME=$(WALLTIME)  OAR_GPU=$(OAR_GPU)"

check-deps:
	@$(PYTHON) -c "\
import sys; \
assert sys.version_info >= (3, 10), ( \
    f'Python 3.10+ requis (trouvé {sys.version.split()[0]}). '\
    'Grid5000: module load python/3.11 puis make setup' \
); \
import torch; \
print('OK: Python', sys.version.split()[0], '| torch', torch.__version__, '→', '$(PYTHON)')" 2>/dev/null || ( \
		echo "Erreur: environnement incomplet pour $(PYTHON)"; \
		echo "  Grid5000 : make setup"; \
		echo "  Ou       : module load $(G5K_PYTHON_MODULE) && make setup"; \
		echo "  Modules  : make list-python-modules"; \
		exit 1)

print-python:
	@echo "PYTHON=$(PYTHON)"
	@$(PYTHON) --version

train: check-deps
	@mkdir -p $(SAVE_DIR)
	$(TRAIN_CMD)

train-fresh: check-deps
	@mkdir -p $(SAVE_DIR)
	$(TRAIN_CMD) --fresh

train-resume: check-deps
	@mkdir -p $(SAVE_DIR)
	$(TRAIN_CMD) --resume auto

train-from-last: check-deps
	@mkdir -p $(SAVE_DIR)
	$(TRAIN_CMD) --resume-path $(SAVE_DIR)/last.pt

reserve:
	$(call write_oar_job_script,train)
	$(call oarsub_batch_job,train,$(OAR_JOB_NAME))

reserve-fresh:
	$(call write_oar_job_script,train-fresh)
	$(call oarsub_batch_job,train-fresh,$(OAR_JOB_NAME)_fresh)

reserve-resume:
	$(call write_oar_job_script,train-resume)
	$(call oarsub_batch_job,train-resume,$(OAR_JOB_NAME)_resume)

logs:
	@tail -f $(SAVE_DIR)/train.log

metrics:
	@tail -f $(SAVE_DIR)/metrics.jsonl

job-status:
	@oarstat -u $$USER 2>/dev/null || oarstat 2>/dev/null || echo "oarstat indisponible (hors Grid5000 ?)"

smoke:
	$(PYTHON) scripts/smoke_test.py
