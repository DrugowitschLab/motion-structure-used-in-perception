import numpy as np
from datetime import datetime
from motionstruct.classes import PhiWorld, PhiWorldDataHose,\
                    PhiObservationGeneratorLocPredict,PhiObservationGeneratorLocPredictFromFiles,\
                    PhiKalmanFilterLocPredict
from motionstruct.functions import dist_mod2pi


# # # # # # # # # # # # # #
# # # CORE PARAMETERS # # #
# # # # # # # # # # # # # #

GROUNDTRUTH = "GLO"                                    # in ("GLO", "CLU", "CDH")
datadsl = "2019-03-26-10-47-59-579319_uid_00107_glo"

glo = 4/5

human_readable_dsl = "pred_datarun_for_" + datadsl

N = 7                   # number of objects
volatility_factor = 8. * np.array( [1., 1., 1.] + [1]*7 )           # Volatility
speed_factor = 1.5

tau_vphi = 8.              # OU time constant of angular velocity
sigma_obs_phi = 0.001      # observation noise of phi

whitespace = True

# # # # # # # # # # # # # #
# # # AUTO PARAMETERS # # #
# # # # # # # # # # # # # #

# Create dataset label
dsl = datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f") + "_" + human_readable_dsl
M = 3+N                 # number of motion sources


# # #  BUILD MOTION STRUCTURE MATRICES  # # #
Tau = {}
Lam = {}
Bs  = {}



# # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # #      O B S E R V E R   M O D E L S      # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # #

# # #  OBSERVER : independent  # # #

OBSERVER = "IND"
B = np.zeros((N, M))    # Motion structure component matrix
lam = np.zeros(M)       # Motion component strengths

B[:] = np.array([
        [ 1, 1, 0, 1,0,0,0,0,0,0],
        [ 1, 1, 0, 0,1,0,0,0,0,0],
        [ 1, 1, 0, 0,0,1,0,0,0,0],
        [ 1,-1, 0, 0,0,0,1,0,0,0],
        [ 1,-1, 0, 0,0,0,0,1,0,0],
        [ 1,-1, 0, 0,0,0,0,0,1,0],
        [ 1, 0, 0, 0,0,0,0,0,0,1],
        ], dtype=np.float64)

lam_tot = 1/2
lam_I = lam_tot
lam_G = 0.
lam_C = 0.

lam[:] = np.sqrt(volatility_factor) * np.array([lam_G, lam_C, 0.] + [lam_I]*7)
lam *= speed_factor

Tau[OBSERVER] = tau_vphi / volatility_factor   # adapt by volatility
Lam[OBSERVER] = lam
Bs[OBSERVER] = B



# # #  OBSERVER : global  # # #

OBSERVER = "GLO"
B = np.zeros((N, M))    # Motion structure component matrix
lam = np.zeros(M)       # Motion component strengths

B[:] = np.array([
        [ 1, 1, 0, 1,0,0,0,0,0,0],
        [ 1, 1, 0, 0,1,0,0,0,0,0],
        [ 1, 1, 0, 0,0,1,0,0,0,0],
        [ 1,-1, 0, 0,0,0,1,0,0,0],
        [ 1,-1, 0, 0,0,0,0,1,0,0],
        [ 1,-1, 0, 0,0,0,0,0,1,0],
        [ 1, 0, 0, 0,0,0,0,0,0,1],
        ], dtype=np.float64)

lam_tot = 1/2
lam_I = 1/12
lam_G = np.sqrt(lam_tot**2 - lam_I**2)
lam_C = 0.

lam[:] = np.sqrt(volatility_factor) * np.array([lam_G, lam_C, 0.] + [lam_I]*7)
lam *= speed_factor

Tau[OBSERVER] = tau_vphi / volatility_factor   # adapt by volatility
Lam[OBSERVER] = lam
Bs[OBSERVER] = B


# # #  OBSERVER : weak global (to match global component of CDH) # # #

OBSERVER = "GLW"
B = np.zeros((N, M))    # Motion structure component matrix
lam = np.zeros(M)       # Motion component strengths

B[:] = np.array([
        [ 1, 1, 0, 1,0,0,0,0,0,0],
        [ 1, 1, 0, 0,1,0,0,0,0,0],
        [ 1, 1, 0, 0,0,1,0,0,0,0],
        [ 1,-1, 0, 0,0,0,1,0,0,0],
        [ 1,-1, 0, 0,0,0,0,1,0,0],
        [ 1,-1, 0, 0,0,0,0,0,1,0],
        [ 1, 0, 0, 0,0,0,0,0,0,1],
        ], dtype=np.float64)

lam_tot = 1/2
lam_I = 1/12
lam_G = np.sqrt(glo) * np.sqrt(lam_tot**2 - lam_I**2)
lam_C = 0.
lam_M = np.sqrt(lam_tot**2 - lam_G**2)

lam[:] = np.sqrt(volatility_factor) * np.array([lam_G, lam_C, 0.] + [lam_M]*7)
lam *= speed_factor

Tau[OBSERVER] = tau_vphi / volatility_factor   # adapt by volatility
Lam[OBSERVER] = lam
Bs[OBSERVER] = B


# # #  OBSERVER : counter-rotating  # # #

OBSERVER = "CNT"
B = np.zeros((N, M))    # Motion structure component matrix
lam = np.zeros(M)       # Motion component strengths

B[:] = np.array([
        [ 1, 1, 0, 1,0,0,0,0,0,0],
        [ 1, 1, 0, 0,1,0,0,0,0,0],
        [ 1, 1, 0, 0,0,1,0,0,0,0],
        [ 1,-1, 0, 0,0,0,1,0,0,0],
        [ 1,-1, 0, 0,0,0,0,1,0,0],
        [ 1,-1, 0, 0,0,0,0,0,1,0],
        [ 1,-1, 0, 0,0,0,0,0,0,1],
        ], dtype=np.float64)

lam_tot = 1/2
lam_I = 1/12
lam_G = 0.
lam_C = np.sqrt(lam_tot**2 - lam_I**2)

lam[:] = np.sqrt(volatility_factor) * np.array([lam_G, lam_C, 0.] + [lam_I]*7)
lam *= speed_factor

Tau[OBSERVER] = tau_vphi / volatility_factor   # adapt by volatility
Lam[OBSERVER] = lam
Bs[OBSERVER] = B

# # #  OBSERVER : clusters  # # #

OBSERVER = "CLU"
B = np.zeros((N, M))    # Motion structure component matrix
lam = np.zeros(M)       # Motion component strengths

B[:] = np.array([
        [ 1, 0, 0, 1,0,0,0,0,0,0],
        [ 1, 0, 0, 0,1,0,0,0,0,0],
        [ 1, 0, 0, 0,0,1,0,0,0,0],
        [ 0, 1, 0, 0,0,0,1,0,0,0],
        [ 0, 1, 0, 0,0,0,0,1,0,0],
        [ 0, 1, 0, 0,0,0,0,0,1,0],
        [ 0, 1, 0, 0,0,0,0,0,0,1],
        ], dtype=np.float64)

lam_tot = 1/2
lam_I = 1/12
lam_G = 0.
lam_C = np.sqrt(lam_tot**2 - lam_I**2)
lam_M = lam_tot

lam[:] = np.sqrt(volatility_factor) * np.array([lam_C, lam_C, 0.] + [lam_I]*7 +[lam_M]*0 )
lam *= speed_factor

Tau[OBSERVER] = tau_vphi / volatility_factor   # adapt by volatility
Lam[OBSERVER] = lam
Bs[OBSERVER] = B


# # #  OBSERVER : weak clusters (To match the Green Cluster <-> Maverick correlation in CDH)  # # #

OBSERVER = "CLW"
B = np.zeros((N, M))    # Motion structure component matrix
lam = np.zeros(M)       # Motion component strengths

B[:] = np.array([
        [ 1, 0, 0, 1,0,0,0,0,0,0],
        [ 1, 0, 0, 0,1,0,0,0,0,0],
        [ 1, 0, 0, 0,0,1,0,0,0,0],
        [ 0, 1, 0, 0,0,0,1,0,0,0],
        [ 0, 1, 0, 0,0,0,0,1,0,0],
        [ 0, 1, 0, 0,0,0,0,0,1,0],
        [ 0, 1, 0, 0,0,0,0,0,0,1],
        ], dtype=np.float64)

lam_tot = 1/2
lam_I = 1/12
lam_G = 0.
lam_C = np.sqrt(glo) * np.sqrt(lam_tot**2 - lam_I**2)
lam_M = np.sqrt(lam_tot**2 - lam_C**2)

lam[:] = np.sqrt(volatility_factor) * np.array([lam_C, lam_C, 0.] + [lam_M]*7 )
lam *= speed_factor

Tau[OBSERVER] = tau_vphi / volatility_factor   # adapt by volatility
Lam[OBSERVER] = lam
Bs[OBSERVER] = B


# # #  OBSERVER : clusters + independent # # #

OBSERVER = "CLI"
B = np.zeros((N, M))    # Motion structure component matrix
lam = np.zeros(M)       # Motion component strengths

B[:] = np.array([
        [ 1, 0, 0, 1,0,0,0,0,0,0],
        [ 1, 0, 0, 0,1,0,0,0,0,0],
        [ 1, 0, 0, 0,0,1,0,0,0,0],
        [ 0, 1, 0, 0,0,0,1,0,0,0],
        [ 0, 1, 0, 0,0,0,0,1,0,0],
        [ 0, 1, 0, 0,0,0,0,0,1,0],
        [ 0, 0, 0, 0,0,0,0,0,0,1],
        ], dtype=np.float64)

lam_tot = 1/2
lam_I = 1/12
lam_G = 0.
lam_C = np.sqrt(lam_tot**2 - lam_I**2)
lam_M = lam_tot

lam[:] = np.sqrt(volatility_factor) * np.array([lam_C, lam_C, 0] + [lam_I]*6 +[lam_M]*1 )
lam *= speed_factor

Tau[OBSERVER] = tau_vphi / volatility_factor   # adapt by volatility
Lam[OBSERVER] = lam
Bs[OBSERVER] = B


# # #  OBSERVER : mag7 = Counter-rotating deep hierarchy # # #

OBSERVER = "CDH"
B = np.zeros((N, M))    # Motion structure component matrix
lam = np.zeros(M)       # Motion component strengths

B[:] = np.array([
        [ 1, 1, 0, 1,0,0,0,0,0,0],
        [ 1, 1, 0, 0,1,0,0,0,0,0],
        [ 1, 1, 0, 0,0,1,0,0,0,0],
        [ 1,-1, 0, 0,0,0,1,0,0,0],
        [ 1,-1, 0, 0,0,0,0,1,0,0],
        [ 1,-1, 0, 0,0,0,0,0,1,0],
        [ 1, 0, 0, 0,0,0,0,0,0,1],
        ], dtype=np.float64)

lam_tot = 1/2
lam_I = 1/12
lam_G = np.sqrt(glo) * np.sqrt(lam_tot**2 - lam_I**2)
lam_C = np.sqrt(lam_tot**2 - lam_G**2 - lam_I**2)
lam_M = np.sqrt(lam_tot**2 - lam_G**2)

lam[:] = np.sqrt(volatility_factor) * np.array([lam_G, lam_C, 0.] + [lam_I]*6 + [lam_M]*1)
lam *= speed_factor

Tau[OBSERVER] = tau_vphi / volatility_factor   # adapt by volatility
Lam[OBSERVER] = lam
Bs[OBSERVER] = B


# # #  OBSERVER : Standard deep hierarchy  # # #

OBSERVER = "SDH"
B = np.zeros((N, M))    # Motion structure component matrix
lam = np.zeros(M)       # Motion component strengths

B[:] = np.array([
        [1, 1, 0, 1,0,0,0,0,0,0],
        [1, 1, 0, 0,1,0,0,0,0,0],
        [1, 1, 0, 0,0,1,0,0,0,0],
        [1, 0, 1, 0,0,0,1,0,0,0],
        [1, 0, 1, 0,0,0,0,1,0,0],
        [1, 0, 1, 0,0,0,0,0,1,0],
        [1, 0, 0, 0,0,0,0,0,0,1],
        ], dtype=np.float64)

lam_tot = 1/2
lam_I = 1/12
lam_G = np.sqrt(glo) * np.sqrt(lam_tot**2 - lam_I**2)
lam_C = np.sqrt(lam_tot**2 - lam_G**2 - lam_I**2)
lam_M = np.sqrt(lam_tot**2 - lam_G**2)

lam[:] = np.sqrt(volatility_factor) * np.array([lam_G, lam_C, lam_C] + [lam_I]*6 + [lam_M]*1)
lam *= speed_factor

Tau[OBSERVER] = tau_vphi / volatility_factor   # adapt by volatility
Lam[OBSERVER] = lam
Bs[OBSERVER] = B


# # #  OBSERVER : Half deep hierarchy  # # #

OBSERVER = "HDH"
B = np.zeros((N, M))    # Motion structure component matrix
lam = np.zeros(M)       # Motion component strengths

B[:] = np.array([
        [0, 1, 0, 1,0,0,0,0,0,0],
        [0, 1, 0, 0,1,0,0,0,0,0],
        [0, 1, 0, 0,0,1,0,0,0,0],
        [1, 0, 1, 0,0,0,1,0,0,0],
        [1, 0, 1, 0,0,0,0,1,0,0],
        [1, 0, 1, 0,0,0,0,0,1,0],
        [1, 0, 0, 0,0,0,0,0,0,1],
        ], dtype=np.float64)

lam_tot = 1/2
lam_I = 1/12
lam_G = np.sqrt(glo) * np.sqrt(lam_tot**2 - lam_I**2)
lam_C1 = np.sqrt(lam_tot**2 - lam_I**2)
lam_C2 = np.sqrt(lam_tot**2 - lam_G**2 - lam_I**2)
lam_M = np.sqrt(lam_tot**2 - lam_G**2)

lam[:] = np.sqrt(volatility_factor) * np.array([lam_G, lam_C1, lam_C2] + [lam_I]*6 + [lam_M]*1)
lam *= speed_factor

Tau[OBSERVER] = tau_vphi / volatility_factor   # adapt by volatility
Lam[OBSERVER] = lam
Bs[OBSERVER] = B

# # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # #    BUILD THE ACTUAL CONFIG DICTIONARY   # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # #

# The actual config dict
cfg = {
    # GLOABAL PARAMETERS
    "global" : dict(
        DRYRUN = False,           # If true, nothing will be saved to disk.
        loglevel = "INFO",      # level of logging ["DEBUG", "INFO", "WARNING", "ERROR"]
        dsl = dsl,               # dataset label
        outdir = "./data/sim/%s" % dsl,      # output directory
        save_minimal = True,     # Save only minimal information (discard trajectory data; keep only last time point)
        ),
    # WORLD SIMULATION (dot movement)
    "world" : dict(
        cls = PhiWorldDataHose,         # World class
        ),
    # OBSERVATIONS (from world state)
    "observe" : dict(
        cls = PhiObservationGeneratorLocPredictFromFiles,
        datadir = "./data/paper/" + datadsl,
        seed = 1002,                     # seed of the random number generator for observations (sensory noise)
        sigma_obs_phi = 0.,              # Observation noise for visible dots
        ),
    # KALMAN TRACKER (for given permutation)
    "tracker" : {
        "cls" : PhiKalmanFilterLocPredict,
        "default_kwargs" : dict(              # These will be used for all expedients and updated with individual dicts
            sigma_obs_phi = sigma_obs_phi,  # assumed observation noise of visible dots
            whitespace = whitespace,         # perform velocity integration in white space (allows multiple tau_vphi)
            valid_max_variance = (np.pi/2)**2,   # Throw a warning if estimated variance exceeds this value (cf. Gauss vs vanMises)
            ),
        "TRU" : dict(
            tau_vphi = Tau[GROUNDTRUTH],            # OU time constant of angular velocity used by the tracker
            B = Bs[GROUNDTRUTH],                      # motion component matrix of the tracker
            lam = Lam[GROUNDTRUTH],                  # motion strengths of the tracker
            ),
        },
    # PERFORMANCE EVALUATION
    "evaluate" : dict(
        error = lambda s_kal, s_wld: dist_mod2pi(s_kal[:N], s_wld[:N]),
        performance = lambda s_kal, s_wld: np.linalg.norm( dist_mod2pi(s_kal[:N], s_wld[:N]) ),
        points = lambda phi_choice, phi_true: np.sum(np.round(10 * np.clip(1 - np.abs(dist_mod2pi(phi_choice, phi_true)/(np.pi/2)), 0, 1)))
        )
    }


trackers = ("IND", "GLO", "GLW", "CNT", "CLU", "CLW", "CLI", "CDH", "SDH", "HDH")
for tr in trackers:
    cfg["tracker"][tr] = dict(
            tau_vphi = Tau[tr],            # OU time constant of angular velocity used by the tracker
            B = Bs[tr],                      # motion component matrix of the tracker
            lam = Lam[tr],                  # motion strengths of the tracker
        )




