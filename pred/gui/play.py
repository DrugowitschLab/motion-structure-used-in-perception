#! /usr/bin/python3

# # # # # #
#   Please excuse the chaotic code: it was first developed with a different application in mind,
#   and then got modified for this project.
# # #

import numpy as np
from time import time
from datetime import datetime
from functions import asciiL, StructuredMotionStimulus, connect_event_handlers, Cursor, PhaseChanger,\
                      create_dsl, create_outdir, fname_of_trial, write_trial_to_file, build_data_dict,\
                      calculate_points
from general_config import config as cfg

# # # # # # # # # # # # # # # # # # # # # # # # #
# # #          Argument parsing             # # #
# # # # # # # # # # # # # # # # # # # # # # # # #

from argparse import ArgumentParser, RawTextHelpFormatter
parser = ArgumentParser(formatter_class=RawTextHelpFormatter,
                        description="Structured Motion Stimuli for Chicken experiments",
                        epilog="If using ipython3, indicate end of ipython arg parser via '--':\n   $ ipython3 play.py -- <args>")
parser.add_argument(dest="stimfile", metavar="stimulus_file.py", type=str,
                    help="python file defining the motion structure (current working directory)")
parser.add_argument("-s", dest="rngseed", metavar="rngseed", default=None, type=int,
                    help="Seed for numpy's random number generator (default: None)")
parser.add_argument("-v", dest="vidfile", metavar="video_file.mp4", default=None, type=str,
                    help="Save video of stimulus to disk (default: None)")
parser.add_argument("-t", dest="tmax", metavar="seconds", default=None, type=float,
                    help="Stimulus duration in seconds, required for -v  (default: infinity)")
parser.add_argument("-f", dest="isFullscreen", action='store_true',
                    help="Run in full screen (press ESC to close; default: false)")
parser.add_argument("-T", dest="maxTrials", metavar="num trials", default=None, type=int,
                    help="Maximum number of trials (default: infinity)")
parser.add_argument("-R", dest="repTrials", metavar="num reps", default=None, type=int,
                    help="Trial repetitions (requires -T; leads to T/R unique trials; default: 1)")
parser.add_argument("-g", dest="greeter", metavar="string", default=None, type=str,
                    help="Greeter displayed before first trial")
parser.add_argument("-u", dest="userID", metavar="ID", default=None, type=int,
                    help="Integer-valued ID of the participant")

args = parser.parse_args()


# # #  Import motion structure from config file  # # #
import os
import sys
stimpath, stimfile = os.path.split(args.stimfile)
sys.path.append(stimpath)                                               # make file accessible to import
stimfilebase = stimfile.split(".py")[0]
cmd = "from " + stimfilebase + " import B, lam, tau_vphi"               # import relevant stimulus parameters
exec(cmd)
# Optional variables (backward comaptible)
varlist = ["targets", "f_dW", "phi0", "human_readable_dsl", "disc_color"]
for varname in varlist:
    cmd = "from " + stimfilebase + " import " + varname
    try:
        exec(cmd)
    except:
        globals()[varname] = None

if args.userID is not None:
    hdsl = "uid_%05d" % args.userID
    if human_readable_dsl is not None:
        hdsl += "_" + human_readable_dsl
    human_readable_dsl = hdsl

DRYRUN = human_readable_dsl is None
if DRYRUN:
    print("\n\n   # # #  D R Y R U N : No data will be saved!  # # #\n\n")

print(" > Motion structure loaded from '%s.py'." % args.stimfile)

if disc_color is not None:
    cfg['display']['disc_color'] = disc_color


# # #  Select matplotlib backend  # # #
import matplotlib as mpl
if args.vidfile is not None:
    assert args.tmax is not None, "Error: For video rendering, a duration < infinity is required."
    assert not os.path.exists(args.vidfile), "Error: Video output file '%s' already exists." % args.vidfile
    mpl.use(cfg['display']['backend_noninteractive'])
    mpl.interactive(False)
else:
    mpl.use(cfg['display']['backend_interactive'])
    mpl.interactive(False)
    print(" > Used backend:", mpl.get_backend())

# Assertions on trials and repetitions
if args.repTrials is None:
    args.repTrials = 1
else:
    assert args.maxTrials is not None, "Option -R requires -T (which was not given)."
    assert args.maxTrials % args.repTrials == 0, "Option -R must divide -T."


# # # # # # # # # # # # # # # # # # # # # # # #
# # #    Import and process parameters    # # #
# # # # # # # # # # # # # # # # # # # # # # # #

DEV = cfg['DEV']
if DEV:
    print(" > DEVELOPER mode turned ON!")

# # #  RNG seeds (np and trial reps)
np.random.seed(args.rngseed)                                            # random seed
if args.maxTrials is None:
    seedlist = np.array([], dtype=np.uint32)
else:
    uniqueTrials = args.maxTrials // args.repTrials
    maxval = np.iinfo(np.uint32).max
    seedlist = np.tile( np.random.randint(0, high=maxval, size=uniqueTrials, dtype=np.uint32), args.repTrials)
    np.random.shuffle(seedlist)

seedlist = np.insert(seedlist, 0, [0])               # Seed for the initial fake trial
seedgenerator = (i for i in seedlist)     # Yields the next seed

# # #  Max time
tmax = args.tmax                                                        # max duration
if tmax is None:
    INFRUN = True
    tmax = 1e10     # 300 years
else:
    INFRUN = False

# # #  Import motion related parameters  # # #
N,M = B.shape                                                           # N dots, M motion components
L = B @ np.diag(lam)                                                    # The motion structure matrix
dt = cfg['sim']['dt']
tau_vr = cfg['sim']['tau_vr']
tau_r = cfg['sim']['tau_r']
radial_sigma = cfg['sim']['radial_sigma']
radial_mean= cfg['sim']['radial_mean']

# # #  Import display related parameters  # # #
fps = cfg['display']['fps']
show_labels = cfg['display']['show_labels']
mpl.rc("figure", dpi=cfg['display']['monitor_dpi'])                     # set monitor dpi

# # #  Print a preview of the motion structure  # # #
print(" > The motion structure matrix L looks as follows:")
print(asciiL(L, 3))
print(" > This leads to the following velocity covariance matrix:")
if isinstance(tau_vphi, float):
    tau_vphi = np.array( [tau_vphi]*M )
print(asciiL(1/2. * L@np.diag(tau_vphi)@L.T, 3))


# # # # # # # # # # # # # # # # # # # # # # # #
# # #  Initialize the stimulus generator  # # #
# # # # # # # # # # # # # # # # # # # # # # # #

# # #  See also class StructuredMotionStimulus in functions.py  # # #
kwargs = dict(L=L, tau_vphi=tau_vphi, tau_r=tau_r, tau_vr=tau_vr, radial_sigma=radial_sigma, radial_mean=radial_mean,
              dt=dt, fps=fps, f_dW=f_dW, phi0=phi0, rngseed=args.rngseed, DEV=DEV)
stim = StructuredMotionStimulus(**kwargs)

frame_wct = []                  # wall clock times of rendered frames

archive = dict(                 # Store stimulus history (unless INFRUN)
    t = [0.],                   # time points of frames (in sim time)
    Phi = [stim.Phi.copy()],    # Angular locations and velocities for all N dots
    R = [stim.R.copy()],         # Radial locations and velocities for all N dots
    visible = [np.arange(N)]    # Which dots are visible at the time frame
    )

# # # INITIALIZE DATA STORAGE  # # #

dsl = create_dsl(human_readable_dsl)
print(" > DSL is: %s" % dsl)

# Create output path and copy config data
if not DRYRUN:
    outdir = create_outdir(dsl)
    from shutil import copyfile
    from os import path
    copyfile(args.stimfile, path.join(outdir, "config.py"))
    copyfile("general_config.py", path.join(outdir, "general_config.py"))



# # # # # # # # # # # # # # # # # # # # # # # # #
# # #    Initialize Figure and Plotting     # # #
# # # # # # # # # # # # # # # # # # # # # # # # #

import pylab as pl

# # #  First plot, called only once  # # #
def init_plot():
    # # #  Axes setup  # # #
    fig.set_facecolor(cfg['display']['bg_color'])
    rect = 0.01, 0.01, 0.98, 0.98
    ax = fig.add_axes(rect, projection='polar')
    ax.set_facecolor(cfg['display']['bg_color'])
    ax.set_thetagrids(np.arange(0,360,45))
    # # #  Plot the dots  # # #
    x = archive['Phi'][-1][:N]
    y = archive['R'][-1][:N]
    norm = pl.matplotlib.colors.Normalize(vmin=0., vmax=1.)
    cmap = pl.cm.Paired
    kwargs = dict(marker='o', s=cfg['display']['disc_radius']**2, c=cfg['display']['disc_color'],
                  cmap=cmap, norm=norm, linewidths=0., zorder=2)
    #if (targets is not None) and globalVars.HIDETARGETS:
        #rgbcolors = [cmap(c) for c in cfg['display']['disc_color']]
        #for i in targets:
            #rgbcolors[i] = rgbcolors[i][:3] + (0.,)
    plottedDots = ax.scatter(x, y, animated=False, **kwargs)       # Test if animated and blit should be used in non-interactive backends
    plottedDots.set_visible(False)                                  # Initially dots are invisible
    # # #  Plot the labels  # # #
    labelkwargs = dict(fontsize=cfg['display']['label_fontsize'], color=cfg['display']['label_color'], weight='bold', ha='center', va='center')
    if not show_labels:
        labelkwargs['visible'] = False                    # no labels? set invisible.
    plottedLabels = []
    for n,(xn,yn) in enumerate(zip(x,y)):
        plottedLabels.append( ax.text(xn, yn, str(n+1), **labelkwargs) )
    # # #  Text instructions  # # #
    plottedText = ax.text(np.pi/2, 0.25, "", weight='bold', size='14', ha="center")
    # # #  Axes range and decoration  # # #
    ax.set_rmax(cfg['display']['axes_radius'])
    #ax.set_rticks(np.array([0.333, 0.666, 1.0, 1.333]) * radial_mean)
    ax.set_rticks(np.array([1.0,]) * radial_mean)
    ax.set_xticks([])
    if cfg['display']['show_grid']:
        ax.grid(True)
        ax.spines['polar'].set_visible(False)
    else:
        ax.grid(False)
        ax.spines['polar'].set_visible(False)
    if not DEV:
        ax.set_yticklabels([])
        ax.set_xticklabels([])
    # # #  Return a list of variable figure elements (required for blitting)  # # #
    return [plottedDots,] + plottedLabels + [plottedText]


# # #  The central routine to update each frame  # # #
def update_dots(count, archive, plottedObjects):
    global globalVars
    global pointbuffer
    # # #  matplotlib's FuncAnimation does some ghost calls in the beginning which we skip  # # #
    if count == 0:
        return plottedObjects
    # # #  Unpack variable elements  # # #
    plottedDots, plottedLabels, plottedText = plottedObjects[0], plottedObjects[1:-2], plottedObjects[-2]
    # # # Set Phase specific controls # # #
    frame_in_trial = globalVars.frame_in_trial
    phase = phaseChanger.getPhase(frame_in_trial)
    if globalVars.trial_number == 0:    # Start of experiment
        plottedDots.set_visible(False)
    else:
        plottedDots.set_visible(True)
    #print(frame_in_trial, phase)
    if (args.maxTrials is not None) and (globalVars.trial_number > args.maxTrials):
        globalVars.PAUSE = True
        globalVars.HIDETARGETS = False
        globalVars.fade_frame_state = 0
        globalVars.cursor.set_visible(False)
        plottedDots.set_visible(False)
        s = "%d points\n" % pointbuffer if pointbuffer is not None else ""
        plottedText.set_text(s + "%d trials completed.\nThank you very much!\nClose with <ESC>" % args.maxTrials)
        globalVars.COMPLETED = True
        return plottedObjects
    if phase == "still":
        globalVars.PAUSE = True
        globalVars.HIDETARGETS = False
        globalVars.MOUSEWASRESET = False
        globalVars.fade_frame_state = 0
        if frame_in_trial < 25:
            plottedText.set_text("")
        else:
            plottedText.set_text("")
        globalVars.cursor.set_visible(False)
    elif phase == "present":
        globalVars.PAUSE = False
        globalVars.HIDETARGETS = False
        globalVars.fade_frame_state = 0
        plottedText.set_text("")
        globalVars.cursor.set_visible(False)
    elif phase == "fade":
        globalVars.PAUSE = False
        globalVars.HIDETARGETS = True
        plottedText.set_text("")
        globalVars.cursor.set_visible(False)
    elif phase == "track":
        globalVars.PAUSE = False
        globalVars.HIDETARGETS = True
        globalVars.fade_frame_state = cfg['experiment']['fade']['numFrames']
        plottedText.set_text("")
        globalVars.cursor.set_visible(False)
        globalVars.cursor.reset_mouse_position()      # always reset mouse pos to prevent glitches
    elif phase == "predict":
        if not globalVars.MOUSEWASRESET:
            globalVars.cursor.reset_mouse_position()
            globalVars.MOUSEWASRESET = True
            if len(globalVars.choicetimes) == 1:    # Only trial start time
                globalVars.choicetimes.append(str(datetime.now()))
        globalVars.PAUSE = True
        globalVars.HIDETARGETS = True
        globalVars.fade_frame_state = cfg['experiment']['fade']['numFrames']
        plottedText.set_text("Make your predictions")
        globalVars.cursor.set_visible(True)
    elif phase == "after":
        pointbuffer = calculate_points(globalVars, archive)
        if (not DRYRUN) and (not globalVars.writtenToDisk):
            datadict = build_data_dict(globalVars, archive)
            fname = fname_of_trial(dsl, globalVars.trial_number)
            write_trial_to_file(fname, datadict)
            globalVars.writtenToDisk = True
        # If all trials complete, we automatically start a new trial to clean up.
        if (args.maxTrials is not None) and (globalVars.trial_number == args.maxTrials):
            plottedDots.set_visible(False)
            globalVars.start_new_trial()
        else:
            globalVars.PAUSE = True
            globalVars.HIDETARGETS = False
            globalVars.fade_frame_state = cfg['experiment']['fade']['numFrames']
            s = "%d points\n" % pointbuffer if pointbuffer is not None else ""
            if args.maxTrials is not None:
                nLeft = args.maxTrials - globalVars.trial_number
                s += "%d trials left\n" % nLeft if nLeft > 1 else "%d trial left\n" % nLeft
            plottedText.set_text(s + "<Mouse click> or <space>\nto proceed")
            globalVars.cursor.set_visible(False)
    # # #  Some necessary book keeping  # # #
    global SIMREADY, t, next_report_time
    assert SIMREADY, "Error: Plotting update called before sim was ready. Too high fps?"
    if (time() - t_start) > next_report_time:                    # Print progress?
        next_report_time += 1
        print("   > Wall-clock time: %7.3fs, simulation time: %7.3fs, frame number: %5d" % (time() - t_start, t, count))
    # # #  Update the figure with latest data  # # #
    x = archive['Phi'][-1][:N]
    y = archive['R'][-1][:N]
    plottedDots.set_offsets(np.vstack([x,y]).T)
    for n,(xn,yn) in enumerate(zip(x,y)):
        plottedLabels[n].set_position((xn, yn))
    cmap = plottedDots.get_cmap()
    rgbcolors = [cmap(c) for c in cfg['display']['disc_color']]
    if (targets is not None) and (globalVars.HIDETARGETS is True):
        for i in targets:
            globalVars.fade_frame_state = min(globalVars.fade_frame_state + 1, cfg['experiment']['fade']['numFrames'])
            f_alpha = lambda n: 1 - n/cfg['experiment']['fade']['numFrames']
            rgbcolors[i] = rgbcolors[i][:3] + (f_alpha(globalVars.fade_frame_state),)
    plottedDots.set_color(rgbcolors)
    frame_wct.append(time())                      # Store the time of frame drawing
    # # #  Integrate the stimulus until the next frame  # # #
    SIMREADY = False
    nSteps = 0 if globalVars.PAUSE else None
    t_in_trial, phi, r = stim.advance(nSteps)     # See class StructuredMotionStimulus in functions.py for dynamics
    t += 1/fps
    # # #  Store the new state  # # #
    if t_in_trial > archive['t'][-1]:
        archive['t'] += [t_in_trial]
        archive['Phi'] += [phi]
        archive['R'] += [r]
        visible_dots = np.arange(N).tolist()
        if phase in ("track", "predict", "after"):
            for i in targets:
                visible_dots.pop(visible_dots.index(i))
        archive['visible'] += [visible_dots]
    SIMREADY = True
    # # #  Test for end of stimulus presentation (-t option)  # # #
    if not INFRUN and (count >= ani_frames - 1):
        print("   > Wall-clock time: %7.3fs, simulation time: %7.3fs, frame number: %5d" % (time() - t_start, t, count))
        if mpl.get_backend() == "TkAgg":         # TkAgg has this nasty bug: https://github.com/matplotlib/matplotlib/issues/9856/
            print(" > Done. Please close the figure window.")
        else:
            print(" > Done. Figure window will be closed.")
            pl.close(fig)             # Close the figure and thus release the block
    # # #  Return the list of variable figure elements (required for blitting)  # # #
    if phase != "predict":
        globalVars.frame_in_trial += 1
    return plottedObjects

# # #  Initialize figure and 1st plot  # # #

if args.isFullscreen:
    pl.matplotlib.rcParams['toolbar'] = 'None'

fig = pl.figure(figsize=cfg['display']['figsize'])
fig.canvas.set_window_title("Structured Motion Stimulus")

# # #  Greeter  # # #
if (args.greeter is not None) and  mpl.get_backend() == "Qt5Agg":
    from PyQt5 import QtWidgets
    sizeObject = QtWidgets.QDesktopWidget().screenGeometry(-1)
    w, h = sizeObject.width(), sizeObject.height()
    from PyQt5.QtWidgets import QMessageBox
    mbox = QMessageBox()
    mbox.resize(w, h)
    # Use <br> in greeter string for new line
    mbox.information(mbox, "Prediction task", args.greeter)

# # #  Init actual plot  # # #
plottedObjects = init_plot()

if args.isFullscreen:
    manager = pl.get_current_fig_manager()
    if mpl.get_backend() == "TkAgg":
        manager.full_screen_toggle()
    elif mpl.get_backend() in ( "Qt4Agg", "Qt5Agg" ):
        manager.window.showFullScreen()



# # #  Init time domains  # # #
t = 0.                      # sim time
t_start = time()            # wall clock time
next_report_time = 0        # printing of progress (in wall clock time)
SIMREADY = True             # A "lock" for security

# # #  Number of frames (function calls) for the animation  # # #
ani_frames = None if INFRUN else int(round(tmax * fps)) + 1

# # #  Inter-frame interval  # # #
if args.vidfile is not None:
    interval = 1                # Render to video? As fast as possible
else:
    interval = 1/fps*1000       # Live preview? 1 / frames per second


phaseChanger = PhaseChanger(cfg['experiment'])




# # # # # # # # # # # # # # # # # # # # # # # # #
# # #               Main loop               # # #
# # # # # # # # # # # # # # # # # # # # # # # # #


class Foo:
    pass
globalVars = Foo()
globalVars.PAUSE = False
globalVars.HIDETARGETS = False
globalVars.COMPLETED = False
globalVars.fig = fig
globalVars.cursor = Cursor(ax=fig.get_axes()[0])
globalVars.phaseChanger = phaseChanger
globalVars.fade_frame_state = cfg['experiment']['fade']['numFrames']
globalVars.frame_in_trial = 0
globalVars.trial_number = 0
globalVars.trial_seed = None
# collect the predictions
globalVars.targets = np.copy(targets)
np.random.shuffle(globalVars.targets)
cmap = plottedObjects[0].get_cmap()
globalVars.targetColors = [cmap(c) for c in cfg['display']['disc_color'][globalVars.targets]]
globalVars.prediction = []
globalVars.choicetimes = [str(datetime.now())]         # Fmt: [start of trial, start of decision period (rounded to frame), time of 1st choice, time of 2nd choice]
globalVars.f_points = cfg['experiment']['f_points']

nextColor = globalVars.targetColors[0]
globalVars.cursor.set_dotkwargs(color=nextColor, size=cfg['display']['disc_radius'][targets[0]])

connect_event_handlers(globalVars=globalVars)
plottedObjects += [globalVars.cursor.dot]
fig.gca().set_rmax(cfg['display']['axes_radius'])

def start_new_trial():
    if (globalVars.trial_number > 0) and (pointbuffer is not None):
        allPoints.append(pointbuffer)
    globalVars.trial_number += 1
    print("\n               # # # #  New trial (%d)  # # # #\n" % globalVars.trial_number)
    try:
        globalVars.trial_seed = next(seedgenerator)
        stim.set_seed(globalVars.trial_seed)
    except StopIteration:
        globalVars.trial_seed = None
        print("No more seeds specified.")
    globalVars.phaseChanger.newTrial()
    globalVars.frame_in_trial = 0
    globalVars.targets = np.copy(targets)
    stim.rng.shuffle(globalVars.targets)                  # repetition trials have identical order
    cmap = plottedObjects[0].get_cmap()
    globalVars.targetColors = [cmap(c) for c in cfg['display']['disc_color'][globalVars.targets]]
    globalVars.cursor.set_dotkwargs(color=globalVars.targetColors[0])
    globalVars.prediction = []
    globalVars.choicetimes = [str(datetime.now())]
    globalVars.writtenToDisk = False
    stim.reset_states()
    global archive
    archive['t'] = [stim.t_in_trial]                   # time points of frames (in sim time)
    archive['Phi'] = [stim.Phi.copy()]                 # Angular locations and velocities for all N dots
    archive['R'] = [stim.R.copy()]                     # Radial locations and velocities for all N dots
    archive['visible'] = [np.arange(N)]                # Which dots are visible at the time framw


globalVars.start_new_trial = start_new_trial
start_new_trial()

pointbuffer = None
allPoints = []

# FAKE END OF TRIAL
globalVars.frame_in_trial = 10000
globalVars.phaseChanger.setPredictionMade()
globalVars.writtenToDisk = True
globalVars.trial_number = 0

print(" > Animation starts.")
import matplotlib.animation as animation

# # #  This is our diligent worker  # # #
ani = animation.FuncAnimation(fig, update_dots, ani_frames, init_func=None,
                fargs=(archive, plottedObjects), interval=interval, blit=True, repeat=False)


# # #  Render the video or live preview  # # #
if args.vidfile is not None:   # Video? Use ani.save with external encoding library
    Writer = animation.writers[cfg['video']['renderer']]
    writer = Writer(metadata=dict(title="Structured Motion Stimulus", artist="Johannes Bill"),
                    fps=fps, codec=cfg['video']['codec'], bitrate=cfg['video']['bitrate'])
    ani.save(args.vidfile, dpi=cfg['video']['dpi'], writer=writer)
    print(" > Video saved to file '%s'." %  args.vidfile)
else:                          # Life preview? Display figure and block further execution
    pl.show(block=True)
    # Frame rate and frame times
    frame_wct = np.array(frame_wct)
    dfwct = frame_wct[1:] - frame_wct[:-1]          # Evaluate inter-frame intervals. Did the preview run at correct speed?
    print(" > Avg frame interval was %.4fs with std deviation ±%.4fs (target was %.4fs)." % (dfwct.mean(), dfwct.std(), 1/fps))
    if not DRYRUN:
        fname = path.join(outdir, "frametimes.npy")
        np.save(fname, frame_wct)
    # Points
    print(" > Total points: %d. Average: %.2f ± %.2f" % (np.sum(allPoints), np.mean(allPoints), np.std(allPoints)) )
    if not DRYRUN:
        fname = path.join(outdir, "points.npy")
        np.save(fname, allPoints)

# # # # # # # # # # # # # # # # # # # # # # # # #
# # #             Debriefing                # # #
# # # # # # # # # # # # # # # # # # # # # # # # #

t_end = time()
print(" > Stimulus presentation complete (wall-clock duration incl overhead: %.3fs)" % (t_end-t_start))


