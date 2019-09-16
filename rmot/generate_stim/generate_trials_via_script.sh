#! /bin/bash

TMPCFGFILE="speed_and_seed.py"
MOSTCFGFILE="config_load_speed_and_seed_23.py"
MAXLOAD=6  # must be int
# nSubjects=1
nSubjects=20

conditions=("independent_thresh" "independent_test" "global" "counter" "hierarchy_124" "hierarchy_127")
# speedFactors=(2.25 )
speedFactors=(1.00 1.25 1.50 1.75 2.00 2.25 2.50 2.75 3.00)

for (( SJ=1; SJ<=$nSubjects; SJ++ ))
do
  for CD in "${conditions[@]}"
  do
    for SF in "${speedFactors[@]}"
    do
       echo "> Subject $SJ, condition $CD, speed factor $SF."
       echo "  > Generate config."
       cat << END |
speed_factor = $SF
PARTICIPANT = $SJ
CONDITION = "$CD"

seedbase = {
    "independent_thresh" : 207,
    "independent_test"   : 307,
    "hierarchy_124"      : 224,
    "hierarchy_127"      : 227,
    "global"             : 800,
    "counter"            : 530,
    }[CONDITION]

wldseed = 100000*int(100*speed_factor) + 100*seedbase + 1*PARTICIPANT

human_readable_dsl = "participant_%02d_%s" % (PARTICIPANT, CONDITION)
END
      awk '{print}' > $TMPCFGFILE
      sleep 1
      echo "  > Start python3 run.py"
      tmux new-session -d python3 run.py $MOSTCFGFILE
      sleep 5
      loadavg=`cat "/proc/loadavg" | awk '{print $0+0}'`
      thisloadavg=`echo $loadavg|awk -F \. '{print $1}'`
      echo "  > Load average is $loadavg."
      while [ "$thisloadavg" -ge $MAXLOAD ]
      do
        echo "  > Waiting for load average ($loadavg) to drop."
        sleep 10
        loadavg=`cat "/proc/loadavg" | awk '{print $0+0}'`
        thisloadavg=`echo $loadavg|awk -F \. '{print $1}'`
      done
      echo "  > Continue."
    done
  done
done

echo "Done."
