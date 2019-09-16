import numpy as np
import pylab as pl
from time import time
import logging
from motionstruct.functions import init_logging, asciiL, recursive_dict_update
from motionstruct.classes import PhiWorld
import scipy.io as sio
import os

# Help string and argument parsing
from argparse import ArgumentParser, RawTextHelpFormatter
parser = ArgumentParser(formatter_class=RawTextHelpFormatter,
                        description="Stimulus generator for rotational MOT",
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
logger.info("Stimulus generator started.")
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
    import os
    if os.path.isfile("speed_and_seed.py"):
        logger.debug("Copy file 'speed_and_seed.py' to '%s'." % (outdir,))
        copyfile("speed_and_seed.py", outdir+"speed_and_seed.py")


L = cfg["world"]["L"]
N, M = L.shape
T = cfg["global"]["T"]

targets = cfg["global"]["targets"]

logger.info("The data's motion structure matrix L looks as follows:\n" + asciiL(L, indent=5))

# Generate World (can be reused in repetitions)
kwargs = cfg["world"]
wld = PhiWorld(**kwargs)

# Generate Observation Generator (can be reused in repetitions)
kwargs = cfg["observe"]
obscls = kwargs.pop("cls")
obs = obscls(**kwargs)

# Data storage
archive = dict(
    wld_t = [],         # world (ground truth) simulation times
    wld_S = [],         # world (ground truth) states
    obs_t = [],         # observation (visible data) time points
    obs_X = [],         # observation (visible data) values
    )

# Take start time
t_start = time()
reps = cfg["global"]["reps"]
# HERE COMES THE OUTER MAIN LOOP
logger.info("Enter simulation main loop.")
for rep in range(reps):
    logger.info("*** Trial %d of %d ***" % (rep+1, reps))
    # Draw the data using the Observation Generator which calls  World Generator
    obs.run_sim_and_generate_observations(T, wld)
    # Store data
    logger.debug("Store data to archive.")
    archive["wld_t"].append(wld.get_times())
    archive["wld_S"].append(wld.S)
    archive["obs_t"].append(obs.get_times())
    archive["obs_X"].append(obs.X)
    # Write matlab file
    if not DRYRUN:
        logger.debug("Write matlab file.")
        fname = cfg["global"]["f_outfname"](rep+1)     # make index matlab friendly
        if not os.path.exists(os.path.dirname(fname)):
            os.makedirs(os.path.dirname(fname))
        mdict = {'X':obs.X, 'T':obs.get_times(), 'targets':[t+1 for t in targets], 'dsl' : cfg["global"]["dsl"]}
        sio.savemat( fname, mdict )

t_end = time()
logger.info("Stimulus generation main loop completed. Main loop runtime: %5.3fs." % (t_end - t_start))

if not DRYRUN:
    fname = outdir + "simdata.npz"
    logger.info("Save results to file '%s'." % fname)
    np.savez_compressed(fname, **archive)

logger.info("Generation completed successfully.")



# TEST WITH
#fig = figure(figsize=(16,9)); ax = fig.add_axes((0,0,1,1), aspect='auto', xlim=(0,1920), ylim=(0,1080)); tn=0
#while tn < obs.X.shape[0]: ax.plot(obs.X[tn,:,0], obs.X[tn,:,1], 'o'); tn+=1


