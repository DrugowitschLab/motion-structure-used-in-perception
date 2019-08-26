#!/bin/bash
set -e                                          # Stop if anything goes wrong
source env/bin/activate                         # Activate environment
export PYTHONPATH=${PYTHONPATH}:${PWD}/pckg     # Make packages available



