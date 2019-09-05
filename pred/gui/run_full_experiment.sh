#!/bin/bash
set -e      # Exit if fails (e.g., aborted with ESC)
while getopts u: option
# while getopts u:T: option
do
case "${option}"
in
u) USERID=${OPTARG};;
# T) TRIALS=${OPTARG};;
esac
done

if [ ! "$USERID" ]; then
  echo "ERROR: No -u 123 user id provided!"
  exit 1
fi

TRIALS=100
REPS=2

echo "run_full_experiment.sh: Starting experiment for user $USERID with $TRIALS trial(s) per block."

# GLOBAL
python3 play.py presets/example_trials/GLO.py -f -g "Next block has training trials<br>with motion structure GLOBAL."
python3 play.py presets/real_trials/GLO.py -fT $TRIALS -R $REPS -u $USERID -g "Next block has $TRIALS real trials<br>with motion structure GLOBAL."

# CLUSTER
python3 play.py presets/example_trials/CLU.py -f -g "Next block has training trials<br>with motion structure CLUSTER."
python3 play.py presets/real_trials/CLU.py -fT $TRIALS -R $REPS -u $USERID -g "Next block has $TRIALS real trials<br>with motion structure CLUSTER."

# HIERARCHY
python3 play.py presets/example_trials/CDH_67.py -f -g "Next block has training trials<br>with motion structure COUNTER-ROTATING HIERARCHY."
python3 play.py presets/real_trials/CDH_67.py -fT $TRIALS -R $REPS -u $USERID -g "Next block has $TRIALS real trials<br>with motion structure COUNTER-ROTATING HIERARCHY."

echo "run_full_experiment.sh: Completed successfully."
