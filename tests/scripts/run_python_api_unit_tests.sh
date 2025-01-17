#/bin/bash

set -eo pipefail

if [[ -z "$TT_METAL_HOME" ]]; then
  echo "Must provide TT_METAL_HOME in environment" 1>&2
  exit 1
fi

# Execute TT Eager unit and sweep tests here
./tests/scripts/run_python_unit_tests.sh
./tests/scripts/run_python_sweep_tests.sh

./tests/scripts/run_python_ttnn_tests.sh

# Execute python model tests here
./tests/scripts/run_python_model_tests.sh
