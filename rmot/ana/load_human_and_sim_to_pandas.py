import numpy as np
import pylab as pl
from os import path
from pandas import DataFrame

# # #  LOAD DATA  # # #
from DSLs_rmot_MarApr_2019 import title, exppath, simpath, subjects, conditions, trackers, DSL

# # #  PARAMETERS  # # #
numCorrect_label = "Accuracy_NumItems_Correct"
targets_label = "targets"
reponse_i_label = lambda i: "Response_%d_stim" % i
trial_label = "trial_number_name"

dataoutfname = "dataframe_" + title.replace(" ", "_") + "_all_responses_human_and_sim.pkl.zip"
SAVE = True

# # # # \PARAMETERS # # # #

# # # # AUX FUNCTIONS # # # #
def df_empty(columns, dtypes, index=None):
    import pandas as pd
    assert len(columns)==len(dtypes)
    df = pd.DataFrame(index=index)
    for c,d in zip(columns, dtypes):
        df[c] = pd.Series(dtype=d)
    return df

numTarget = None
def load_human_performance_by_condition(fname, cond):
    global numTarget
    X = np.load(fname)
    STRUCT = X["STRUCT"]
    if numTarget is None:
        numTarget = len(X[targets_label][0])
    else:
        assert numTarget == len(X[targets_label][0])
    assert numCorrect_label in STRUCT[:,0], "ERROR: 'Accuracy_NumItems_Correct' not found in data."
    trialidx = X['condition'] == cond
    trialnumber = X[trial_label][trialidx]
    perf = X[numCorrect_label][trialidx]
    targets = X[targets_label][trialidx] - 1  # to python indexing
    choiceset = np.transpose( [X[reponse_i_label(i+1)][trialidx] for i in range(numTarget)]  ) - 1    # to python indexing
    return trialnumber, perf, targets, choiceset

def load_kalman_performance_by_tracker(fname, trac):
    import pickle, gzip
    with gzip.open(fname) as f:
        X = pickle.load(f)
    return X[trac]["performance"], X[trac]["kal_gam"]  # This assumes "minimal save" was set

# # # # LOAD DATA # # # #

nReps = None

cols = ("subject", "condition", "targets", "tracker", "trial", "repetition", "numCorrect", "choiceset")
dtypes = (np.int, str, object, str, np.int, np.int, np.int, object)
df = df_empty(columns=cols, dtypes=dtypes)
for subj in subjects:
    for cond in conditions:
        print(" > Loading data for subject %s, condition %s:" % (str(subj), cond))
        # Load human data
        if DSL[subj]["exp"] is None:
            print(" [WARNING] Skipping: No human data found!")
            continue
        print("  > Loading human data...", flush=True, end="")
        trac = "Human"
        rep = 0
        fname = path.join(exppath, DSL[subj]["exp"])
        trialnumber, perf, targets, choiceset = load_human_performance_by_condition(fname, cond)
        d = DataFrame({"subject" : subj, "condition" : cond, "targets" : targets.tolist(), "tracker" : trac, "trial" : trialnumber, "repetition" :rep, "numCorrect" : perf, "choiceset" : choiceset.tolist()})
        df = df.append(d, ignore_index=True)
        print(" [loaded %d trials]" % (len(perf)))
        # Load simulation data
        if DSL[subj]["sim"][cond] is None:
            print("  > [WARNING] Skipping simulation data: No simulation data specified!")
            continue
        fname = path.join(simpath, DSL[subj]["sim"][cond], "simdata.pickle.zip")
        for trac in trackers:
            print("  > Loading tracker '%s'..." % trac, flush=True, end="")
            Perf, Gam = load_kalman_performance_by_tracker(fname, trac)
            for n, (perf, gam) in enumerate(zip(Perf, Gam)):
                R = len(perf)
                cs = [ gam[r,0][targets[n]] for r in range(R) ]     # Here we REUSE the human targets since we are in the same "cond"
                if nReps is not None:
                    assert R == nReps
                nReps = R
                d = DataFrame({"subject" : subj, "condition" : cond, "targets" : [targets[n]]*R, "tracker" : trac, "trial" : n+1, "repetition" : np.arange(1, R+1), "numCorrect" : perf, "choiceset" : cs})
                df = df.append(d, ignore_index=True)
            print(" [loaded %d trials with %d repetitions]" % (Perf.shape[0], Perf.shape[1]))

if SAVE is True:
    print(" > Save dataframe to file: %s" % dataoutfname)
    df.to_pickle(dataoutfname, compression="gzip")
    # LOAD VIA: df = pandas.read_pickle(dataoutfname, compression='gzip')
print(" > Done.")


