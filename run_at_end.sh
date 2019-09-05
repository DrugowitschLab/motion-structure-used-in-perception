#!/bin/bash
if [ "${BASH_SOURCE[0]}" -ef "$0" ]
then
    echo "ERROR: Please source this script (via 'source ./run_at_start.sh'), not only execute it!"
    exit 1
fi
export PYTHONPATH="$OLD_PYTHONPATH"             # Restore python path
deactivate                                      # Deactivate environment

