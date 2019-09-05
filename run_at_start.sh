#!/bin/bash
if [ "${BASH_SOURCE[0]}" -ef "$0" ]
then
    echo "ERROR: Please source this script (via 'source ./run_at_start.sh'), not only execute it!"
    exit 1
fi
source env/bin/activate                         # Activate environment
export OLD_PYTHONPATH="$PYTHONPATH"
export PYTHONPATH=${PWD}/pckg                   # Make packages available

