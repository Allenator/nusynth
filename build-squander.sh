#!/bin/bash

# SQUANDER build helper script
# Tested on Arch Linux
# Refer to https://github.com/rakytap/sequential-quantum-gate-decomposer

SCRIPT_PATH="$( cd -- "$(dirname "$0")" > /dev/null 2>&1 ; pwd -P )"
SQUANDER_PATH="${SCRIPT_PATH}/sequential-quantum-gate-decomposer"

git submodule deinit -f sequential-quantum-gate-decomposer/
git submodule update --init

cd ${SQUANDER_PATH}

export GSL_LIB_DIR="/usr/lib"
export GSL_INC_DIR="/usr/include/gsl"
export TBB_LIB_DIR="/usr/lib"
export TBB_INC_DIR="/usr/include/tbb/"

python3 setup.py bdist_wheel

pip3 uninstall qgd -y
pip3 install dist/qgd-*.whl
