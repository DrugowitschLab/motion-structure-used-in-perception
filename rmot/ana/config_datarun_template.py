import numpy as np
from datetime import datetime
from motionstruct.classes import PhiWorldDataHose, PhiObservationGeneratorMOTFromFiles, PhiKalmanFilterPermutation, DPVI_Particle_Filter, eqf_unique_snowflake
from sympy.combinatorics import Permutation as SympyPermutation
from motionstruct.classPermutation import Permutation

from DSLs_rmot_MarApr_2019 import DSL as DSL_human

# # # # # # # # # # # # # #
# # # CORE PARAMETERS # # #
# # # # # # # # # # # # # #

# # # MARKER 1 # # #
stimdata_dsl = "2019-03-21-12-27-02-096006_participant_20_hierarchy_127_speed_2.25_targets_127"
# # # MARKER 2 # # #

conditions = ["independent", "global", "counter", "hierarchy_124", "hierarchy_127", "None"]
# Determine condition
for CONDITION in conditions:
    if CONDITION in stimdata_dsl:
        break
assert CONDITION != "None"
conditions.pop(-1)  # remove None dummy

# Determine subject
idx = stimdata_dsl.find("participant_") + len("participant_")
assert idx != -1
subj = int(stimdata_dsl[idx:idx+2])

# Determine speed
idx = stimdata_dsl.find("speed_") + len("speed_")
assert idx != -1
s = stimdata_dsl[idx:idx+4]
speed_factor = float(s)
assert speed_factor == DSL_human[subj]["speed"], "Speed of simdata does not match human speed! "


human_readable_dsl = "datarun_for_" + stimdata_dsl

fps = 60                # frames per second of MOT display
N = 7                   # number of objects
if CONDITION == "hierarchy_127":
    targets = [0,1,6]
else:
    targets = [0,1,3]       # Target indices (for performance evaluation)

volatility_factor = 1/8 #  * np.array( [1., 1.] + [1]*7 )           # Heterogeneous volatility
tau_vphi = 8.           # OU time constant of angular velocity
sigma_obs_phi = 0.050

whitespace = False
trackers = ("IND", "GLO", "GLW", "CNT", "CLU", "CLW", "CLI", "CDH", "SDH")

# # # # # # # # # # # # # #
# # # AUTO PARAMETERS # # #
# # # # # # # # # # # # # #

# Create dataset label
dsl = datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f") + "_" + human_readable_dsl

def make_L_and_tau_vphi(CONDITION, tracker):
    "CONDITION is the ground truth; tracker is the Kalman filter"
    global tau_vphi
    assert tracker in trackers, "Tracker '%s' unknown" % tracker
    M = 3+N                 # number of motion sources
    # # # BUILD MOTION STRUCTURE MATRIX # # #
    B = np.zeros((N, M))    # Motion structure component matrix
    lam = np.zeros(M)       # Motion component strengths
    # Some B matrices need to be changed to give the strongest possible comparison to the ground truth.
    # trackers IND and GLO are always identical
    # tracker CNT needs adaptation to the stimulus condition:
    if CONDITION in ("global", "counter") and tracker == "CNT":
        B[:] = np.array([
            [ 0, 1, 0, 1,0,0,0,0,0,0],
            [ 0, 1, 0, 0,1,0,0,0,0,0],
            [ 0, 1, 0, 0,0,1,0,0,0,0],
            [ 0, 1, 0, 0,0,0,1,0,0,0],
            [ 0,-1, 0, 0,0,0,0,1,0,0],
            [ 0,-1, 0, 0,0,0,0,0,1,0],
            [ 0,-1, 0, 0,0,0,0,0,0,1],
            ], dtype=np.float64)
    elif CONDITION in ("hierarchy_124", "hierarchy_127") and tracker == "CNT":
        B[:] = np.array([
            [ 0, 1, 0, 1,0,0,0,0,0,0],
            [ 0, 1, 0, 0,1,0,0,0,0,0],
            [ 0, 1, 0, 0,0,1,0,0,0,0],
            [ 0,-1, 0, 0,0,0,1,0,0,0],
            [ 0,-1, 0, 0,0,0,0,1,0,0],
            [ 0,-1, 0, 0,0,0,0,0,1,0],
            [ 0, 1, 0, 0,0,0,0,0,0,1],
            ], dtype=np.float64)
    elif tracker == "CNT":
        B[:] = np.array([
            [ 0, 1, 0, 1,0,0,0,0,0,0],
            [ 0, 1, 0, 0,1,0,0,0,0,0],
            [ 0, 1, 0, 0,0,1,0,0,0,0],
            [ 0,-1, 0, 0,0,0,1,0,0,0],
            [ 0,-1, 0, 0,0,0,0,1,0,0],
            [ 0,-1, 0, 0,0,0,0,0,1,0],
            [ 0,-1, 0, 0,0,0,0,0,0,1],
            ], dtype=np.float64)
    # tracker CLU needs adaptation:
    elif CONDITION in ("hierarchy_124", "hierarchy_127") and tracker in ("CLU", "CLW"):
        B[:] = np.array([
            [ 0, 1, 0, 1,0,0,0,0,0,0],
            [ 0, 1, 0, 0,1,0,0,0,0,0],
            [ 0, 1, 0, 0,0,1,0,0,0,0],
            [ 0, 0, 1, 0,0,0,1,0,0,0],
            [ 0, 0, 1, 0,0,0,0,1,0,0],
            [ 0, 0, 1, 0,0,0,0,0,1,0],
            [ 0, 1, 0, 0,0,0,0,0,0,1],
            ], dtype=np.float64)
    elif tracker in ("CLU", "CLW"):
        B[:] = np.array([
            [ 0, 1, 0, 1,0,0,0,0,0,0],
            [ 0, 1, 0, 0,1,0,0,0,0,0],
            [ 0, 1, 0, 0,0,1,0,0,0,0],
            [ 0, 1, 0, 0,0,0,1,0,0,0],
            [ 0, 0, 1, 0,0,0,0,1,0,0],
            [ 0, 0, 1, 0,0,0,0,0,1,0],
            [ 0, 0, 1, 0,0,0,0,0,0,1],
            ], dtype=np.float64)
    # Tracker CLI needs adaptation
    elif CONDITION in ("counter", ) and tracker == "CLI":
        B[:] = np.array([
            [ 0, 1, 0, 1,0,0,0,0,0,0],
            [ 0, 1, 0, 0,1,0,0,0,0,0],
            [ 0, 1, 0, 0,0,1,0,0,0,0],
            [ 0, 0, 0, 0,0,0,1,0,0,0],
            [ 0, 0, 1, 0,0,0,0,1,0,0],
            [ 0, 0, 1, 0,0,0,0,0,1,0],
            [ 0, 0, 1, 0,0,0,0,0,0,1],
            ], dtype=np.float64)
    elif tracker == "CLI":
        B[:] = np.array([
            [ 0, 1, 0, 1,0,0,0,0,0,0],
            [ 0, 1, 0, 0,1,0,0,0,0,0],
            [ 0, 1, 0, 0,0,1,0,0,0,0],
            [ 0, 0, 1, 0,0,0,1,0,0,0],
            [ 0, 0, 1, 0,0,0,0,1,0,0],
            [ 0, 0, 1, 0,0,0,0,0,1,0],
            [ 0, 0, 0, 0,0,0,0,0,0,1],
            ], dtype=np.float64)
    # tracker SDH needs adaptation
    elif CONDITION in ("counter",) and tracker == "SDH":
        B[:] = np.array([
            [ 1, 1, 0, 1,0,0,0,0,0,0],
            [ 1, 1, 0, 0,1,0,0,0,0,0],
            [ 1, 1, 0, 0,0,1,0,0,0,0],
            [ 1, 0, 0, 0,0,0,1,0,0,0],
            [ 1, 0, 1, 0,0,0,0,1,0,0],
            [ 1, 0, 1, 0,0,0,0,0,1,0],
            [ 1, 0, 1, 0,0,0,0,0,0,1],
            ], dtype=np.float64)
    elif tracker == "SDH":
        B[:] = np.array([
            [ 1, 1, 0, 1,0,0,0,0,0,0],
            [ 1, 1, 0, 0,1,0,0,0,0,0],
            [ 1, 1, 0, 0,0,1,0,0,0,0],
            [ 1, 0, 1, 0,0,0,1,0,0,0],
            [ 1, 0, 1, 0,0,0,0,1,0,0],
            [ 1, 0, 1, 0,0,0,0,0,1,0],
            [ 1, 0, 0, 0,0,0,0,0,0,1],
            ], dtype=np.float64)
    # tracker CDH needs adaptation
    elif CONDITION in ("counter",) and tracker == "CDH":
        B[:] = np.array([
            [ 1, 1, 0, 1,0,0,0,0,0,0],
            [ 1, 1, 0, 0,1,0,0,0,0,0],
            [ 1, 1, 0, 0,0,1,0,0,0,0],
            [ 1, 0, 0, 0,0,0,1,0,0,0],
            [ 1,-1, 0, 0,0,0,0,1,0,0],
            [ 1,-1, 0, 0,0,0,0,0,1,0],
            [ 1,-1, 0, 0,0,0,0,0,0,1],
            ], dtype=np.float64)
    else:
        B[:] = np.array([
            [ 1, 1, 0, 1,0,0,0,0,0,0],
            [ 1, 1, 0, 0,1,0,0,0,0,0],
            [ 1, 1, 0, 0,0,1,0,0,0,0],
            [ 1,-1, 0, 0,0,0,1,0,0,0],
            [ 1,-1, 0, 0,0,0,0,1,0,0],
            [ 1,-1, 0, 0,0,0,0,0,1,0],
            [ 1, 0, 0, 0,0,0,0,0,0,1],
            ], dtype=np.float64)

    lam_T = 0.5
    lam_I = 0.05
    lam_G = np.sqrt(lam_T**2 - lam_I**2) * np.sqrt(2/3)
    lam_C = np.sqrt(lam_T**2 - lam_G**2 - lam_I**2)
    lam_M = np.sqrt(lam_T**2 - lam_G**2)
    # strength of the components (columns of B)
    if tracker == "IND":
        lam = np.array([0, 0, 0] + [lam_T]*N)
    elif tracker == "GLO":
        lam_G = np.sqrt(lam_T**2 - lam_I**2)
        lam = np.array([lam_G, 0, 0] + [lam_I]*7 + [lam_M]*0 + [lam_T]*0)
    elif tracker == "GLW":
        lam_I =  np.sqrt(lam_T**2 - lam_G**2)
        lam = np.array([lam_G, 0, 0] + [lam_I]*7 + [lam_M]*0 + [lam_T]*0)
    elif tracker == "CNT":
        lam_C = np.sqrt(lam_T**2 - lam_I**2)
        lam = np.array([0, lam_C, 0] + [lam_I]*7 + [lam_M]*0 + [lam_T]*0)
    elif tracker == "CLU":
        lam_C = np.sqrt(lam_T**2 - lam_I**2)
        lam = np.array([0, lam_C, lam_C] + [lam_I]*7 + [lam_M]*0 + [lam_T]*0)
    elif tracker == "CLW":
        lam_C = lam_G
        lam_I = np.sqrt(lam_T**2 - lam_G**2)
        lam = np.array([0, lam_C, lam_C] + [lam_I]*7 + [lam_M]*0 + [lam_T]*0)
    elif CONDITION in ("counter", ) and tracker == "CLI":
        lam_C = np.sqrt(lam_T**2 - lam_I**2)
        lam = np.array([0, lam_C, lam_C] + [lam_I]*3 + [lam_T]*1 + [lam_I]*3)
    elif tracker == "CLI":
        lam_C = np.sqrt(lam_T**2 - lam_I**2)
        lam = np.array([0, lam_C, lam_C] + [lam_I]*6 + [lam_T]*1)
    elif CONDITION in ("counter",) and tracker == "SDH":
        lam = np.array([lam_G, lam_C, lam_C] + [lam_I]*3 + [lam_M]*1 + [lam_I]*3)
    elif tracker == "SDH":
        lam = np.array([lam_G, lam_C, lam_C] + [lam_I]*6 + [lam_M]*1)
    elif CONDITION in ("counter",) and tracker == "CDH":
        lam = np.array([lam_G, lam_C, 0] + [lam_I]*3 + [lam_M]*1 + [lam_I]*3)
    elif tracker == "CDH":
        lam = np.array([lam_G, lam_C, 0] + [lam_I]*6 + [lam_M]*1)
    else:
        raise Exception("Unknown combination ('%s', '%s')." % (CONDITION, tracker))

    lam *= np.sqrt(volatility_factor)
    lam /= np.sqrt(2)
    lam *= speed_factor
    tau = tau_vphi / volatility_factor
    L = B @ np.diag(lam)    # Complete motion matrix
    return L, tau



# # # COMPLETE MATRIX # # #

# Use sympy for construction, then cast to faster type
differential_confusion_candidates = [ SympyPermutation(size=N)] + [ SympyPermutation(size=N)(m,n) for n in range(N) for m in range(n) ]
differential_confusion_candidates = [ Permutation(p.list()) for p in differential_confusion_candidates ]

# The actual config dict
cfg = {
    # GLOABAL PARAMETERS
    "global" : dict(
        DRYRUN = False,         # If true, nothing will be saved to disk.
        loglevel = "INFO",     # level of logging ["DEBUG", "INFO", "WARNING", "ERROR"]
        dsl = dsl,              # dataset label
        outdir = "./data/sim/%s" % dsl,      # output directory
        T = 15.,                 # duration of one run
        T_burnin = 9.,           # duration during which NO confusion is possible (real tracking duration will be T - T_burnin)
        save_minimal = True,     # Save only minimal information (discard Kalman trajectory data; keep only last time point)
        ),
    # WORLD SIMULATION (dot movement)
    "world" : dict(
        cls = PhiWorldDataHose,         # World class
        ),
    # OBSERVATIONS (from world state)
    "observe" : dict(
        cls = PhiObservationGeneratorMOTFromFiles,
        datadir = "./data/paper/trials/" + stimdata_dsl,
        seed = 1002,                     # seed of the random number generator for observations (sensory noise)
        trials = np.arange(30, dtype=int),
        reps_per_trial = 25,            # HINT: REDUCE THIS NUMBER TO 1 FOR QUICKER (and less accurate) RESULTS
        gen_x = lambda rng, s: np.concatenate( ( rng.normal(s[:N], sigma_obs_phi) % (2*np.pi), np.zeros(N)) ) ,
        ),
    # KALMAN TRACKER (for given permutation)
    "tracker" : {
        "cls" : PhiKalmanFilterPermutation,
        "default_kwargs" : dict(
            sigma_obs_phi = sigma_obs_phi,  # assumed observation noise
            init_certain = False,            # The first state is not only correct, but also has high certainty
            whitespace = whitespace # whether to perform velocity integration in white space (allows heterogeneous tau_vphi)
            ),
        },
    # DATA ASSIGNMENT (permutation and confusion)
    "assign" : dict(
        cls = DPVI_Particle_Filter,
        equality_func = eqf_unique_snowflake,
        maxParticles = 1,
        perm_proposal = differential_confusion_candidates,
        ),
    # PERFORMANCE EVALUATION
    "evaluate" : dict(
        performance = lambda perm: np.intersect1d(perm[list(targets)], list(targets)).size
        )
    }


# ADD THE TRACKERS
tr = {"independent" : "IND",
      "global" : "GLO",
      "counter" : "CNT",
      "hierarchy_124" : "CDH",
      "hierarchy_127" : "CDH"}[CONDITION]
L, tau = make_L_and_tau_vphi(CONDITION, tr)
cfg["tracker"]["TRU"] = dict(L=L, tau_vphi=tau)

for tr in ("IND",):  #trackers:
    L, tau = make_L_and_tau_vphi(CONDITION, tr)
    cfg["tracker"][tr] = dict(L=L, tau_vphi=tau)


