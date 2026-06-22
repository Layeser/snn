# Variables partagées (override : make train DATA_DIR=/chemin/data)

include $(dir $(lastword $(MAKEFILE_LIST)))python.mk
include $(dir $(lastword $(MAKEFILE_LIST)))grid5000.mk

PROJECT ?=
SNN_ROOT := $(abspath $(CURDIR)/..)
VENV_PYTHON := $(SNN_ROOT)/.venv/bin/python
CONFIG ?= config/train.yml
DATA_DIR ?= $(HOME)/internship/snn/data
SAVE_DIR ?= $(CURDIR)/save

VENV_TORCH := $(call _python_has_torch,$(VENV_PYTHON))
VENV_VERSION_OK := $(call _python_version_ok,$(VENV_PYTHON))
_DEFAULT_PYTHON := $(if $(and $(wildcard $(VENV_PYTHON)),$(filter yes,$(VENV_TORCH)),$(filter yes,$(VENV_VERSION_OK))),$(VENV_PYTHON),$(DETECTED_PYTHON))
PYTHON ?= $(_DEFAULT_PYTHON)

# Grid5000 / OAR (voir mk/grid5000.mk)
OAR_JOB_NAME ?= $(PROJECT)_train

PROJECT_DIR := $(abspath $(CURDIR))
TRAIN_ARGS := --config $(CONFIG) --save-dir $(SAVE_DIR) --data-dir $(DATA_DIR)
TRAIN_CMD := $(PYTHON) -m scripts.train $(TRAIN_ARGS)

OAR_RUN := bash -lc '$(G5K_MODULE_LOAD) cd $(PROJECT_DIR) && $(MAKE) --no-print-directory'
