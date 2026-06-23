# Génère un script batch OAR (évite les conflits -- avec oarsub)

define write_oar_job_script
	@mkdir -p $(SAVE_DIR)
	@printf '%s\n' \
		'#!/bin/bash' \
		'set -euo pipefail' \
		'$(G5K_MODULE_LOAD)' \
		'cd "$(PROJECT_DIR)"' \
		'exec $(MAKE) $(1) DATA_DIR="$(DATA_DIR)" SAVE_DIR="$(SAVE_DIR)"' \
		> "$(SAVE_DIR)/run_job.sh"
	@chmod +x "$(SAVE_DIR)/run_job.sh"
endef

define oarsub_batch_job
	$(OARSUB_BATCH) \
		-O "$(SAVE_DIR)/run.out" \
		-E "$(SAVE_DIR)/run.err" \
		-n "$(2)" \
		-- "$(SAVE_DIR)/run_job.sh"
endef
