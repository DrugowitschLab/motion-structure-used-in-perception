import numpy as np
import pylab as pl
from time import time
import logging
from motionstruct.functions import init_logging, asciiL, recursive_dict_update

# Help string and argument parsing
from argparse import ArgumentParser, RawTextHelpFormatter
parser = ArgumentParser(formatter_class=RawTextHelpFormatter,
                        description="Discrete-time Kalman tracker for rotational location prediction tasks",
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

if cfg["global"]["save_minimal"]:
    logger.warning("Warning: Only minimal data will be stored!")

if "T" in cfg["global"]:
    T = cfg["global"]["T"]
else:
    T = None

if "B" in cfg["world"]:
    L = cfg["world"]["B"] @ np.diag(cfg["world"]["lam"])
    N, M = L.shape
    logger.info("The data's motion structure matrix L looks as follows:\n" + asciiL(L, indent=5))
else:
    logger.info("The data's motion structure matrix L is not provided in cfg.")

tracker_names = list(cfg["tracker"].keys())
tracker_names.remove("cls")
tracker_names.remove("default_kwargs")
logger.info("There will be %d trackers." % len(tracker_names))
for tname in tracker_names:
    Lt = cfg["tracker"][tname]["B"] @ np.diag(cfg["tracker"][tname]["lam"])
    taut = cfg["tracker"][tname]["tau_vphi"]
    logger.info("The motion structure matrix L of tracker '%s' looks as follows:\n" % (tname,) + asciiL(Lt, indent=5))
    sigdiag = (Lt @ np.diag(taut) @ Lt.T / 2).diagonal()
    logger.info("leading to stationary diag(velo-covariance): " + str(sigdiag))

# Generate World (can be reused in repetitions)
kwargs = cfg["world"]
wldcls = cfg["world"].pop("cls")
wld = wldcls(**kwargs)

# Generate Observation Generator (can be reused in repetitions)
kwargs = cfg["observe"]
obscls = cfg["observe"].pop("cls")
obs = obscls(**kwargs)

# Kalman filter class
kalcls = cfg["tracker"].pop("cls")
kal_default_kwargs = cfg["tracker"].pop("default_kwargs")

# Choice agent
if ("agent" in cfg) and (cfg["agent"]["cls"] is not None):
    agtcls = cfg["agent"].pop("cls")
    agt_trackers = cfg["agent"].pop("trackers")
    agt = agtcls(**cfg["agent"])
else:
    agt = None

# Performance measuring function
performance_func = cfg["evaluate"]["performance"]
error_func = cfg["evaluate"]["error"]
points_func = cfg["evaluate"]["points"] if "points" in cfg["evaluate"] else None

# Data storage
archive = dict(
    wld_t = [],         # world (ground truth) simulation times
    wld_S = [],         # world (ground truth) states
    obs_t = [],         # observation (visible data) time points
    obs_X = [],         # observation (visible data) values
    )

for tname in tracker_names:
    archive[tname] = dict(kal_t = [],         # the trackers's observation time points
                          kal_mu = [],        # the trackers's estimated mean
                          kal_Sig = [],       # the trackers's estimated covariance matrix
                          error = [],         # Distance of world state (mod 2pi)
                          performance = []    # performance of each trial
                          )
if agt is not None:                       # Agent dependent results
    for tname in agt_trackers:
        archive[tname]["choice"] = []         # The agent's choices
        if "points" in cfg["evaluate"]:
            archive[tname]["points"] = []     # and the points earned


# Take start time
t_start = time()
from motionstruct.classes import PhiObservationGeneratorLocPredictFromFiles
if obscls == PhiObservationGeneratorLocPredictFromFiles:
    reps = obs.reps
    logger.info("Loaded number of repetitions (%d) from files." % reps)
else:
    reps = cfg["global"]["reps"]

# HERE COMES THE OUTER MAIN LOOP
logger.info("Enter simulation main loop.")
for rep in range(reps):
    logger.info("*** Trial %d of %d ***" % (rep+1, reps))
    # Draw the data using the Observation Generator which calls  World Generator
    obs.run_sim_and_generate_observations(T, wld)
    # Generate the Kalman filters
    kals = dict()
    for tname in tracker_names:
        logger.debug("Adding Kalman filter '%s'." % tname)
        kwargs = dict(kal_default_kwargs)
        kwargs.update(cfg["tracker"][tname])
        kwargs['logger'] = logger
        kal = kalcls(**kwargs)
        s0 = wld.S[0]
        kal.init_filter(s0)           # initialize them with the correct estimates and assignment
        kals[tname] = kal
    # HERE IS THE INNER MAIN LOOP
    for x,t in zip(obs.X, obs.get_times()):
        for tname, kal in kals.items():
            kal.propagate_and_integrate_observation(x,t)
    # If an agent is specified, now his time has come!
    if agt is not None:
        for tname in agt_trackers:
            N = kals[tname].N
            mu = kals[tname].archive["mu"][-1][:N]
            Sig = kals[tname].archive["Sig"][-1][:N,:N]
            choice = agt.draw_response(mu, Sig)
            archive[tname]["choice"].append(choice)
            if points_func:
                p = [points_func(c, wld.S[-1][obs.targets]) for c in choice]
                archive[tname]["points"].append(p)
    # Store data
    ts = slice(-1,None) if cfg["global"]["save_minimal"] else slice(None, None)
    logger.debug("Store data to archive.")
    archive["wld_t"].append(wld.get_times()[ts])
    archive["wld_S"].append(wld.S[ts])
    archive["obs_t"].append(obs.get_times()[ts])
    archive["obs_X"].append(obs.X[ts])
    for tname, kal in kals.items():
        ka = kal.archive
        archive[tname]["kal_t"].append(ka["t"][ts])
        archive[tname]["kal_mu"].append(ka["mu"][ts])
        archive[tname]["kal_Sig"].append(ka["Sig"][ts])
        archive[tname]["error"].append(error_func(ka["mu"][-1], wld.S[-1]))
        archive[tname]["performance"].append(performance_func(ka["mu"][-1], wld.S[-1]))
    logger.debug("Trial complete.")


t_end = time()
logger.info("Simulation main loop completed. Main loop runtime: %5.3fs." % (t_end - t_start))
from scipy.stats import sem
for tname in kals:
    p = archive[tname]["performance"]
    logger.info("Average performance of tracker '%s': %f ± %f (sem)." % (tname, np.mean(p), sem(p)))

if not DRYRUN:
    fname = outdir + "simdata.pickle.zip"
    logger.info("Save results to file '%s'." % fname)
    import pickle, gzip
    pickle.dump( archive, gzip.open( fname, 'wb' ), protocol=-1)
    #np.savez_compressed(fname, **archive)

logger.info("Simulation completed successfully.")

