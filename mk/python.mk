# Détection Python 3.10+
# Grid5000 (Lille, etc.) : module Spack `python/3.10.8` — détecté automatiquement.

PYTHON_MIN_MAJOR := 3
PYTHON_MIN_MINOR := 10
G5K_PYTHON_MODULE ?= python/3.10.8

# 1) binaires PATH  2) module Grid5000  3) fallback python3
DETECTED_PYTHON := $(shell \
	for c in python3.12 python3.11 python3.10 python3; do \
		if command -v $$c >/dev/null 2>&1 && \
		   $$c -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; then \
			echo $$c; exit 0; \
		fi; \
	done; \
	g5k=$$(bash -lc 'module load $(G5K_PYTHON_MODULE) 2>/dev/null && command -v python3' 2>/dev/null); \
	if [ -n "$$g5k" ] && $$g5k -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; then \
		echo $$g5k; exit 0; \
	fi; \
	command -v python3 2>/dev/null || echo python3)

G5K_MODULE_LOAD := module load $(G5K_PYTHON_MODULE) 2>/dev/null;

define _python_has_torch
$(shell $(1) -c "import torch" >/dev/null 2>&1 && echo yes)
endef

define _python_version_ok
$(shell $(1) -c 'import sys; sys.exit(0 if sys.version_info >= ($(PYTHON_MIN_MAJOR), $(PYTHON_MIN_MINOR)) else 1)' 2>/dev/null && echo yes)
endef
