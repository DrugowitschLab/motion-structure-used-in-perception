import numpy as np
import pylab as pl
from time import time
import logging
from motionstruct.functions import init_logging, asciiL, recursive_dict_update
from motionstruct.classes import PhiWorld, PhiObservationGenerator, PhiKalmanFilterPermutation, DPVI_Particle, DPVI_Particle_Filter
from motionstruct.classPermutation import Permutation


# Help string and argument parsing
from argparse import ArgumentParser, RawTextHelpFormatter
parser = ArgumentParser(formatter_class=RawTextHelpFormatter,
                        description="Discrete-time Kalman tracker for rotational MOT",
                        epilog="If using ipython3, indicate end of ipython arg parser via '--':\n   $ ipython3 run.py -- <args>")
parser.add_argument(dest="cfgfile", metavar="filename", default="config.py", nargs='?', type=str,
                    help="config.py file holding dictionary 'cfg' (same directory, default: config.py)")
parser.add_argument("-u", "--update", dest="updatefile", metavar="filename", default=[], nargs='*', type=str,
                    help="file holding dictionary 'cfg' for recursively updating the main cfg (same directory, default: None)")
args = parser.parse_args()

# Import config from specified config file
cfgmodulename = args.cfgfile.split(".py")[0]
cmd = "from " + cfgmodulename + " import cfg"
exec(cmd)

msg = []
for udf in args.updatefile:
    updatemodulename = udf.split(".py")[0]
    cmd = "from " + updatemodulename + " import cfg as ucfg"
    exec(cmd)
    _,msg = recursive_dict_update(cfg, ucfg, msg=msg, s="[%s] cfg." % updatemodulename)

# Dryrun?
DRYRUN = cfg["global"]["DRYRUN"]


# # # # # # # # # # # # # # # # # # # # # #
# # #     M A I N   R O U T I N E     # # #
# # # # # # # # # # # # # # # # # # # # # #

# Create the output directory
outdir = cfg["global"]["outdir"]
if not DRYRUN:
    import os
    if outdir[-1] != "/":
        outdir += "/"
    if not os.path.exists(outdir):
        os.makedirs(outdir)


# Create the logger
logger = init_logging(cfg, outdir)
logger.info("Simulation started.")
logger.info("Loading config from file: %s.py" % cfgmodulename)
if len(args.updatefile) > 0:
    logger.info("Updating config from files: %s" % str(args.updatefile))
    logger.debug("Number of entry updates: %d. Details follow." % len(msg))
    for m in msg:
        logger.debug("Updated key: %s" % m)
logger.info("DSL: '%s'" % cfg["global"]["dsl"])


if not DRYRUN:
    logger.debug("Output directory: %s" % outdir)
    # copy the config file to outdir
    from shutil import copyfile
    logger.debug("Copy file '%s.py' to '%s'." % (cfgmodulename, outdir))
    copyfile("%s.py" % cfgmodulename, outdir+"config.py")
    if len(args.updatefile) > 0:
        for i,udf in enumerate(args.updatefile):
            logger.debug("Copy file '%s' to '%s'." % (udf, outdir))
            copyfile(udf, outdir+"uconfig_%d.py" % (i+1))

if ("save_minimal" in cfg["global"]) and cfg["global"]["save_minimal"] is True:
    logger.warning("Warning: Only minimal data will be stored!")
    SAVE_MINIMAL = True
else:
    SAVE_MINIMAL = False


if "L" in cfg["world"]:
    L = cfg["world"]["L"]
    logger.info("The data's motion structure matrix L looks as follows:\n" + asciiL(L, indent=5))
else:
    logger.info("The data's motion structure matrix is determined by data.")

tracker_names = list(cfg["tracker"].keys())
tracker_names.remove("cls")
tracker_names.remove("default_kwargs")
logger.info("There will be %d trackers." % len(tracker_names))
for tr in tracker_names:
    Lt = cfg["tracker"][tr]["L"]
    taut = cfg["tracker"][tr]["tau_vphi"]
    logger.info("The motion structure matrix L of tracker '%s' looks as follows:\n" % (tr,) + asciiL(Lt, indent=5))
    sigdiag = (Lt @ Lt.T * taut / 2).diagonal()
    logger.info("leading to stationary diag(velo-covariance): " + str(sigdiag))


T = cfg["global"]["T"]
# Generate World (can be reused in repetitions)
kwargs = cfg["world"]
wldcls = kwargs.pop("cls")
wld = wldcls(**kwargs)

# Generate Observation Generator (can be reused in repetitions)
kwargs = cfg["observe"]
obscls = kwargs.pop("cls")
obs = obscls(**kwargs)

Lt = cfg["tracker"]["TRU"]["L"]
N, M = Lt.shape

try:
    L_DATA = obs.get_L_groundtruth()
except:
    L_DATA = None

if L_DATA is not None: # We use the squared version here since empty columns in L can be ignored.
    assert np.allclose(Lt@Lt.T, L_DATA@L_DATA.T), "ERROR: Tracker does not match ground truth L!"
    logger.info("Verified: Ground truth L (data) and tracker L ('TRU') match.")
else:
    logger.warn("Could not load ground truth L from data; skipping check.")

# Kalman filter class
kalcls = cfg["tracker"].pop("cls", PhiKalmanFilterPermutation)   # Defaults to PhiKalmanFilterPermutation
kal_default_kwargs = cfg["tracker"].pop("default_kwargs")

# Particle Filter class
pfcls = cfg["assign"].pop("cls")

# Performance measuring function
performance_func = cfg["evaluate"]["performance"]

# Data storage
archive = dict(
    obs_t = [],         # observation (visible data) time points
    obs_X = [],         # observation (visible data) values
    )

for tr in tracker_names:
    archive[tr] = dict(kal_t = [],         # the best particle's observation time points
                       kal_mu = [],        # the best particle's estimated mean
                       kal_Sig = [],       # the best particle's estimated covariance matrix
                       kal_gam = [],       # the best particle's permutation over time
                       performance = []    # performance of each trial
                       )

# Take start time
t_start = time()
nTrial = len(obs.trials)
reps_per_trial = obs.reps_per_trial


# HERE COMES THE OUTER MAIN LOOP
logger.info("Enter simulation main loop.")
for trial in range(nTrial):
    for rep in range(reps_per_trial):
        logger.info("*** Trial %d of %d : Repetition %d of %d ***" % (trial+1, nTrial, rep+1, reps_per_trial))
        # Draw the data using the Observation Generator which calls  World Generator
        obs.run_sim_and_generate_observations(T, wld, trial, rep)
        # Generate the initial (seeding) Kalman filters / particle filters
        Pf = dict()
        for tr in tracker_names:
            logger.debug("Adding particle filter and Kalman filter '%s'." % tr)
            kwargs = dict(kal_default_kwargs)   # Kalman filter kwargs
            kwargs.update(cfg["tracker"][tr])   # L and tau_vphi
            kal = kalcls(**kwargs)              # create
            kal.init_filter(wld.S[0])           # initialize it with the correct estimates and assignment
            kwargs = cfg["assign"]              # particle filter kwargs
            pf = pfcls(**kwargs)                # create
            pf.set_initial_kal(kal)             # and charge it with the initial Kalman filter
            Pf[tr] = pf
        # Check for burnin time
        isBurnin = False
        if ( "T_burnin" in cfg["global"] ) and ( cfg["global"]["T_burnin"] > 0. ):
            perm_proposal = [ Permutation(size=N) ]
            for tr in Pf:
                Pf[tr].set_proposal_permutations(perm_proposal)
            isBurnin = True
        # HERE IS THE INNER MAIN LOOP
        for x,t in zip(obs.X, obs.get_times()):
            # Check for end of burnin period, and allow for confusion
            if ( isBurnin is True ) and ( t >= cfg["global"]["T_burnin"] ):
                isBurnin = False
                perm_proposal = cfg['assign']['perm_proposal']
                for tr in Pf:
                    Pf[tr].set_proposal_permutations(perm_proposal)
            for tr in Pf:
                pf = Pf[tr]
                smax = pf.propagate_evaluate_select_and_integrate_observation(x, t)
                logger.debug("t=%.3f: Highest ranked permutations for tracker '%s': %s \t %s \t %s" % (t, tr, str(pf.P[0].kal.gam), str(pf.P[:2][-1].kal.gam), str(pf.P[:3][-1].kal.gam) ))
                logger.debug("         Highest relative score: %.2e" % smax)
        logger.info("Highest ranked permutations at trial end:")
        for tr in Pf:
            pf = Pf[tr]
            logger.info("  '%s' : %s" % (tr, str(pf.P[0].kal.gam)))
        logger.debug("Store data to archive.")
        ts = slice(-1,None) if SAVE_MINIMAL else slice(None, None)
        # Evaluate the performance
        archive["obs_t"].append(obs.get_times()[ts])
        archive["obs_X"].append(obs.X[ts])
        perf_opt = cfg['evaluate']['performance'](cfg['assign']['perm_proposal'][0].perm)  # perfect performance evaluated from identity permutation
        for tr in Pf:
            pf = Pf[tr]
            perf = performance_func(pf.P[0].kal.gam.perm)       # performance of the best particle
            logger.info("Tracking performance of '%s': %d / %d correct" % (tr, perf, perf_opt))
            # Store data
            ka = pf.P[0].kal.archive
            archive[tr]["kal_t"].append(ka["t"][ts])
            archive[tr]["kal_mu"].append(ka["mu"][ts])
            archive[tr]["kal_Sig"].append(ka["Sig"][ts])
            archive[tr]["kal_gam"].append(ka["gam"][ts])
            archive[tr]["performance"].append(perf)

Tn = 1 if SAVE_MINIMAL else len(obs.get_times())
Tnplus = 1 if SAVE_MINIMAL else (len(obs.get_times()) + 1)
archive["obs_t"] = np.array( archive["obs_t"] ).reshape(nTrial, reps_per_trial, Tn)
archive["obs_X"] = np.array( archive["obs_X"] ).reshape(nTrial, reps_per_trial, Tn, obs.X.shape[-1])
for tr in Pf:
    archive[tr]["kal_t"] = np.array( archive[tr]["kal_t"] ).reshape(nTrial, reps_per_trial, Tnplus)       # +1 --> init + obs
    archive[tr]["kal_mu"] = np.array( archive[tr]["kal_mu"] ).reshape(nTrial, reps_per_trial, Tnplus, obs.X.shape[-1])
    archive[tr]["kal_Sig"] = np.array( archive[tr]["kal_Sig"] ).reshape(nTrial, reps_per_trial, Tnplus, obs.X.shape[-1], obs.X.shape[-1])
    archive[tr]["kal_gam"] = np.array( archive[tr]["kal_gam"] ).reshape(nTrial, reps_per_trial, Tnplus, N)
    archive[tr]["performance"] = np.array( archive[tr]["performance"] ).reshape(nTrial, reps_per_trial)

t_end = time()
logger.info("Simulation main loop completed. Main loop runtime: %5.3fs." % (t_end - t_start))
from scipy.stats import sem
for tr in tracker_names:
    performance = np.array(archive[tr]["performance"])
    logger.info("Average performance of '%s': %f ± %f (sem)." % (tr, performance.mean(), sem(performance.mean(1))))

if not DRYRUN:
    fname = outdir + "simdata.pickle.zip"
    logger.info("Save results to file '%s'." % fname)
    import pickle, gzip
    pickle.dump( archive, gzip.open( fname, 'wb' ), protocol=-1)

logger.info("Simulation completed successfully.")


