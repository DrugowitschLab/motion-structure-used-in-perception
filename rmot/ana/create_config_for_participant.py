import numpy as np
import pylab as pl
from glob import glob

# PARAMS
subj = 1
cond = "IND"  # in ("IND", "GLO", "CNT", "CDH1", "CDH2")
from DSLs_rmot_MarApr_2019 import DSL, subjects, conditions
stimdir = "./data/paper/trials/"
stimfile_start_pattern = "2019-03-21*_"
base_config_fname = "config_datarun_multikal_MarApr_2019.py"

marker_1 = "# # # MARKER 1 # # #"
fill_str = lambda dsl: "stimdata_dsl = '%s'" % dsl
marker_2 = "# # # MARKER 2 # # #"
outfile_fname = "tmp_config.py"
# END OF PARAMS

condstr = dict(IND="independent_test",
               GLO="global",
               CNT="counter",
               CDH1="hierarchy_124",
               CDH2="hierarchy_127",
            )[cond]

assert subj in subjects, "Subject not in DSL file."
assert condstr in conditions, "Condition not in DSL file."


# Find right stimulus file
speed = DSL[subj]["speed"]
matchstr = stimdir + stimfile_start_pattern + "participant_%02d_%s_speed_%.2f_*" % (subj, condstr, speed)
dirlist = glob(matchstr)

assert len(dirlist) == 1, "Error in found stimulus directories!"
stimdsl = dirlist[0].replace(stimdir, "")
print(" > Found uniquely matching stimulus: %s" % stimdsl)

# Write config file
with open(base_config_fname) as f:
    C = f.readlines()

if outfile_fname:
    print(" > Writing config file to: %s" % outfile_fname)
    with open(outfile_fname, "w") as f:
        WRITING = True
        for l in C:
            # l = l.replace("\n", "")
            if marker_1 in l:
                f.write(fill_str(stimdsl))
                WRITING = False
            elif marker_2 in l:
                WRITING = True
                f.write("\n")
            elif WRITING:
                f.write(l)
else:
    print(" > Skip writing to file.")