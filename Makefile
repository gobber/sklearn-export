BASH := /bin/bash

PYTHON_FILES := $(shell find -s ./sklearn_export -name '*.py' | tr '\n' ' ')

TEST_N_RANDOM_FEATURE_SETS=25
TEST_N_EXISTING_FEATURE_SETS=25

#
# Requirements
#

install.environment:
	$(info Start [install.environment] ...)
	$(BASH) .scripts/install.environment.sh

install.requirements:
	$(info Start [install.requirements] ...)
	$(BASH) .scripts/install.requirements.sh