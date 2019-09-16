import numpy as np
from datetime import datetime
from motionstruct.classes import ScreenXYObservationGenerator
from sympy.combinatorics import Permutation as SympyPermutation
from motionstruct.classPermutation import Permutation

from speed_and_seed import speed_factor, wldseed, human_readable_dsl, CONDITION, PARTICIPANT

# # # # # # # # # # # # # #
# # # CORE PARAMETERS # # #
# # # # # # # # # # # # # #

fps = 60                # frames per second of MOT display
N = 7                   # number of objects
# Different from the plotting in the paper, we use dots 1,2,4 (or 1,2,7 for CDH2) as targets.
# The difference, however, is only a re-ordering for the presentation.
if CONDITION == "hierarchy_127":
    targets = [0,1,6]
else:
    targets = [0,1,3]       # Target indices (for performance evaluation)

volatility_factor = 1/8 #  * np.array( [1., 1.] + [1]*7 )           # Heterogeneous volatility
tau_vphi = 8.           # OU time constant of angular velocity
whitespace = False

# # # # # # # # # # # # # #
# # # AUTO PARAMETERS # # #
# # # # # # # # # # # # # #

# Create dataset label
tstr = str([t+1 for t in targets]).replace(", ", "").replace("]","").replace("[",'')
dsl = datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f") + "_" + human_readable_dsl + "_speed_%.2f" % speed_factor +  "_targets_" + tstr

M = 2+N                 # number of motion sources
# # # BUILD MOTION STRUCTURE MATRIX # # #
B = np.zeros((N, M))    # Motion structure component matrix
lam = np.zeros(M)       # Motion component strengths

if CONDITION == "counter":
    B[:] = np.array([
        [ 0, 1, 1,0,0,0,0,0,0],
        [ 0, 1, 0,1,0,0,0,0,0],
        [ 0, 1, 0,0,1,0,0,0,0],
        [ 0, 1, 0,0,0,1,0,0,0],
        [ 0,-1, 0,0,0,0,1,0,0],
        [ 0,-1, 0,0,0,0,0,1,0],
        [ 0,-1, 0,0,0,0,0,0,1],
        ], dtype=np.float64)
else:
    B[:] = np.array([
        [ 1, 1, 1,0,0,0,0,0,0],
        [ 1, 1, 0,1,0,0,0,0,0],
        [ 1, 1, 0,0,1,0,0,0,0],
        [ 1,-1, 0,0,0,1,0,0,0],
        [ 1,-1, 0,0,0,0,1,0,0],
        [ 1,-1, 0,0,0,0,0,1,0],
        [ 1, 0, 0,0,0,0,0,0,1],
        ], dtype=np.float64)

lam_tot = 0.5
lam_ind_obj = 0.05
lam_glo = np.sqrt(lam_tot**2 - lam_ind_obj**2) * np.sqrt(2/3)
lam_obj = np.sqrt(lam_tot**2 - lam_glo**2 - lam_ind_obj**2)
lam_ind_moon = np.sqrt(lam_tot**2 - lam_glo**2)
lam_noise = lam_tot
# strength of the components (columns of B)
if CONDITION in ("independent_thresh", "independent_test"):
    lam = np.sqrt(volatility_factor) * np.array([0, 0] + [lam_noise]*N)
elif CONDITION in ("hierarchy_124", "hierarchy_127"):
    lam = np.sqrt(volatility_factor) * np.array([lam_glo, lam_obj] + [lam_ind_obj]*6 + [lam_ind_moon]*1 + [lam_noise]*0)
elif CONDITION == "global":
    lam_glo = np.sqrt(lam_tot**2 - lam_ind_obj**2)
    lam = np.sqrt(volatility_factor) * np.array([lam_glo, 0] + [lam_ind_obj]*7 + [lam_ind_moon]*0 + [lam_noise]*0)
elif CONDITION == "counter":
    lam_obj = np.sqrt(lam_tot**2 - lam_ind_obj**2)
    lam = np.sqrt(volatility_factor) * np.array([0, lam_obj] + [lam_ind_obj]*7 + [lam_ind_moon]*0 + [lam_noise]*0)
else:
    raise Exception("Unknown condition.")

lam /= np.sqrt(2)
lam *= speed_factor
tau_vphi_wld = tau_vphi / volatility_factor
L = B @ np.diag(lam)    # Complete motion matrix

# # # COMPLETE MATRIX # # #

# The actual config dict
cfg = {
    # GLOABAL PARAMETERS
    "global" : dict(
        DRYRUN = False,         # If true, nothing will be saved to disk.
        loglevel = "INFO",     # level of logging ["DEBUG", "INFO", "WARNING", "ERROR"]
        dsl = dsl,              # dataset label
        outdir = "./data/myexp/trials/%s" % dsl,      # output directory
        f_outfname = lambda n: "./data/myexp/matlab_trials/participant_%d/%s/%.2f/trial_%05d.mat" % (PARTICIPANT, CONDITION, speed_factor, n),
        reps = 30,                  # number of experiment repetitions (trials)
        T = 15.,                 # duration of one run
        targets = targets,
        ),
    # WORLD SIMULATION (dot movement)
    "world" : dict(
        seed = wldseed,           # seed of the random number generator for the world (dot motion)
        dt = 1./fps / 10,       # world simulation time resolution
        L = L,                  # true motion matrix
        tau_vphi = tau_vphi_wld,     # OU time constant of angular velocity
        whitespace = whitespace # whether to perform velocity integration in white space (allows heterogeneous tau_vphi)
        ),
    # OBSERVATIONS (from world state)
    "observe" : dict(
        cls = ScreenXYObservationGenerator,
        dt = 1./fps,                          # interval between observations, i.e., 1/frames-per-second
        screen_resolution = (2048, 1152),     # pixel resolution
        screen_aspect = 16 / 9,               # physical aspect ratio (potentially correcting for non-square pixels)
        relative_radius = 0.4,                # circle size on the screen
        final_min_distance = 10/360*2*np.pi   # Redraw trajectory if final locations are overlapping
        ),
    }

