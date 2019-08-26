#!/bin/bash
source env/bin/activate                         # Activate environment
export OLD_PYTHONPATH="$PYTHONPATH"
export PYTHONPATH=${PWD}/pckg                   # Make packages available

