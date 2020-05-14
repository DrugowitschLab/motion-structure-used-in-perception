from mpl_settings import panellabel_fontkwargs, fig_width
import numpy as np
import pylab as pl
from scipy.stats import sem
from scipy.stats import linregress

# # #  PARAMETERS  # # #

# A : Motion trees
# B : Bar plots
# C : Observer model
# D : Stacked bar plots

ZOOM = 2.
PLOT = ("A","B","C","D")
SAVEFIG = True

fname_data = "../data/dataframe_MOT_MarchApril_2019_human_and_sim.pkl.zip"

perf_level = dict(chance=3*3/7, thresh=2.15, perfect=3.)

trackers = ("Human", "TRU", "IND")
visible_trackers = trackers
conditions = ("independent_test", "global", "counter", "hierarchy_124", "hierarchy_127")

condition_labels = { "independent_test" : "IND",
                     "global" : "GLO",
                     "counter" : "CNT",
                     "hierarchy_124" : r"CDH$_\mathrm{1}$",
                     "hierarchy_127" : r"CDH$_\mathrm{2}$"
                   }

inv_condition_labels = { "IND" : "independent_test",
                          "GLO" : "global",
                          "CNT" : "counter",
                          "CDH" : "hierarchy_124",
                          "CDH1" : "hierarchy_124",
                          "CDH2" : "hierarchy_127"
                     }

perfcolors = {"independent_test" : "0.2",
              "global" : "limegreen",
              "counter" : "cornflowerblue",
              "hierarchy_124" : "orange",
              "hierarchy_127" : "tomato"
              }

tracker_hatches = {
    "Human" : None,
    "TRU" : None,
    "IND" : "////",
    }

stacked_perf_color = { i : pl.cm.RdYlGn(i/3.) for i in range(4) }

# # #  \PARAMETERS  # # #


# # #  AUX FUNCTIONS  # # #
def light_color(colname, weight=0.3):
    c4 = pl.matplotlib.colors.to_rgba_array(colname)
    w4 = pl.matplotlib.colors.to_rgba_array("white")
    weights = weight, 1-weight
    avg4 = (np.vstack([c4,w4]).T * weights).sum(1) / np.sum(weights)
    return avg4

def auto_yscale(ax=None, data=None, tol=0.15):
    if ax is None:
        ax = pl.gca()
    if data is not None:
        ymin, ymax = data.min(), data.max()
    else:
        ymin, ymax = ax.yaxis.get_data_interval()
    d = tol * (ymax - ymin)
    ax.set_ylim(ymin - d, ymax + d)
    pl.draw()

def print_panel_label(letter, ax, dx=-0.03, dy=0.005, abs_x=None, transform=None):
    if not transform:
        transform = fig.transFigure
    x,y = ax.bbox.transformed(transform.inverted()).corners()[1]  # Top left in fig coords
    if abs_x is not None:
        x = abs_x
        dx = 0.
    t = ax.text(x+dx, y+dy, letter, transform=transform, **panellabel_fontkwargs)
    return t

# # #  \AUX FUNCTIONS  # # #


# # #  MAIN   # # #
pl.matplotlib.rc("figure", dpi=ZOOM*pl.matplotlib.rcParams['figure.dpi'])

import pandas
print("> Load data from file: %s" % fname_data)
df = pandas.read_pickle(fname_data, compression='gzip')

subjects = np.unique(df.subject)
nReps = np.max(df["repetition"])
nTrials = max(df['trial'])

pc = perf_level["chance"]
pp = perf_level["perfect"]

# # #  Setup Figure and axes  # # #
w,h = fig_width["onecol"], 1.2*fig_width["onecol"]
fig = pl.figure(figsize=(w, h))
ar = w/h
axes = dict()

# trees
condlist = ("IND", "GLO", "CNT", "CDH")
hi = 0.48/fig.get_size_inches()[1]
wi = hi/ar * 432/288         # <-- enter pixel dims of PNG panels here
for i,s in enumerate( condlist ):
    li  = 0.06 + i*(wi + 0.025)
    ri = li + wi      # for alignment of perf. and title
    rect = li, 0.97-hi, wi, hi
    ax = fig.add_axes(rect, aspect="auto", xticks=[], yticks=[])
    axes["tree_" + s] = ax

# bars
rect = 0.12, 0.82-0.30, ri-0.12, 0.30
ax = fig.add_axes(rect, aspect="auto")
axes["bars"] = ax

# observer model
hi = 0.55/fig.get_size_inches()[1]
wi = hi/ar * 510/330            # <-- enter pixel dims of PNG panels here
nPanel = 3
for i in range(nPanel):
    li = 0.06 + i * (ri - 0.06)/nPanel + ((ri - 0.06)/nPanel - wi)/2
    rect = li, 0.31, wi, hi
    ax = fig.add_axes(rect, aspect="auto", xticks=[], yticks=[])
    axes["obsmodel_%d" % i] = ax

# stacked
wtot = ri-0.12
wi = wtot / ( len(conditions) * (1 + 0.15) - 0.15 )
for i,c in enumerate(conditions):
    rect = 0.12 + i * wi * 1.15, 0.075, wi, 0.15
    ax = fig.add_axes(rect, aspect="auto")
    axes["stacked_%s" % c] = ax


# # #  Set panel labels
print_panel_label(letter="A", ax=axes["tree_IND"], abs_x=0.025, dy=0.)
print_panel_label(letter="B", ax=axes["bars"], abs_x=0.025)
print_panel_label(letter="C", ax=axes["obsmodel_0"], abs_x=0.025)
print_panel_label(letter="D", ax=axes["stacked_independent_test"], abs_x=0.025, dy=0.035)


# # #  PLOT TREES  # # #
if "A" in PLOT:
    for s in condlist:
        ax = axes["tree_" + s]
        ax.imshow(pl.imread(f"./panel/mot_struct_{s}.png"))
        for name in ('bottom', 'top', 'right', 'left'):
            c = perfcolors[inv_condition_labels[s]]
            if s=="CDH" and name in ('bottom', 'right'):
                c = perfcolors[inv_condition_labels["CDH2"]]
            ax.spines[name].set_color(light_color(c, weight=0.7))
            ax.spines[name].set_lw(1.5)
            ax.set_xticks([])
            ax.set_yticks([])


# # #  BARS  # # #
if "B" in PLOT:
    ax = axes["bars"]
    ymin, ymax = 10.,0.
    nTracker = len(trackers)
    nCond = len(conditions)
    eps = 0.016
    for tn, trac in enumerate(trackers):
        hatch=tracker_hatches[trac]
        w = 0.8/nCond - eps
        xbias = -0.4 + tn
        x = xbias + np.linspace(0, 0.8, nCond+1)[:-1] + eps/2
        Y = [] # condition, subj
        for cond in conditions:
            y = []
            for subj in subjects:
                idx = (df.subject == subj) & (df.condition == cond) & (df.tracker == trac)
                if idx.sum() == 0:
                    raise Exception("> Error: No data for %s:%s:%s!" % (subj, cond, trac))
                d = df.loc[idx]
                y.append(np.mean(d["numCorrect"]))
            Y.append(y)
        # PLOT BARS
        colors = [perfcolors[cond] for cond in conditions]
        if trac == "Human":
            fc = [light_color(c, weight=0.6) for c in colors]
        else:
            fc = [light_color(c, weight=0.3) for c in colors]
        ec = [light_color(c, weight=0.7) for c in colors]
        l = "Ground truth" if trac == "TRU" else trac
        kwargs = dict(align='edge', color=fc, hatch=hatch, width=w, edgecolor=ec,
                      linewidth=0.75, ecolor="0.5", capsize=1.0, zorder=1, error_kw=dict(lw=0.75))
        ymean = np.mean(Y, axis=1)
        yerr = sem(Y, axis=1)
        if trac in visible_trackers:
            ax.bar(x, ymean, yerr=yerr, **kwargs)
        # TRACKER NAMES
        if trac=="TRU":
            for xi, cond in zip(x, conditions):
                s = condition_labels[cond]
                c = perfcolors[cond]
                kwargs = dict(color=c, rotation="vertical", ha="center", va="bottom")
                delta = 0.015 if "hierarchy" in cond else 0.005
                ax.text(float(xi+w/2) + delta, perf_level["chance"]+0.025, s, **kwargs)
        # PLOT DOTS
        xsubject = np.linspace(0, w, len(subjects) + 4)[2:-2]
        x = (x[:,None] + xsubject).flatten()
        y = np.array(Y).flatten()
        weight = 1. if trac == "Human" else 0.8
        c = [light_color(perfcolors[cond], weight=weight) for cond in conditions for subj in subjects]
        kwargs = dict(s=1., c=c, marker='.', zorder=2)
        if trac in visible_trackers:
            ax.scatter(x, y, **kwargs)
    # # #  Beautify
    xmin, xmax = -0.5, nTracker-0.5
    ax.set_xlim(xmin, xmax)

    # dotted lines
    for y in perf_level.values():
        ymax = max(y, ymax)
        ymin = min(y, ymin)
        kwargs = dict(dashes=[2, 2], c="0.3", lw=0.5, zorder=0)
        ax.plot([xmin,xmax], [y]*2, **kwargs)

    # yticks
    auto_yscale(ax=ax, data=np.array((ymin, ymax)), tol=0.03)
    ax.set_ylabel("Performance\navg ± sem", labelpad=-24)
    ax.set_yticks([perf_level["chance"], perf_level["perfect"]])
    ax.set_yticklabels([f"\n{pc:.2f}\n(chance)", f"\n{pp:.2f}\n(perfect)"])

    # xticks
    ax.set_xticks(np.arange(nTracker))
    ax.set_xticklabels([ dict(Human="Human", TRU="Correct prior", IND="IND prior")[tr] for tr in trackers], fontsize=8)

    patch = pl.Rectangle((0.5, 0.), 5, 5, fc="0.9", ec="0.8", lw=0.5, zorder=-1)
    ax.add_patch(patch)
    t = ax.text(1.8, 3-0.2, "Bayesian observer\nmodel predictions", fontsize=6, va="center", ha="center", color="0.15")


if "C" in PLOT:
    for i,s in enumerate(("graph", "assignment", "confusion")):
        ax = axes["obsmodel_%d" % i]
        ax.imshow(pl.imread(f"./panel/observer_{s}.png"))
        for name in ('bottom', 'top', 'right', 'left'):
            ax.spines[name].set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])


# # #  STACKED
if "D" in PLOT:
    nT = nTrials * len(subjects)
    for cn,cond in enumerate(conditions):
        # count
        F = np.zeros((len(trackers),4))
        for trn, tr in enumerate(trackers):
            count = np.bincount(df.loc[(df.condition==cond) & (df.tracker==tr)].numCorrect).astype(float)
            assert count.sum() == nT * (1 if tr=="Human" else nReps)
            count /= nT * (1 if tr=="Human" else nReps)
            F[trn] = count
        # plot
        ax = axes["stacked_%s" % cond]
        bottom = np.zeros(len(trackers))
        for ncorr in range(4):
            y = F[:,ncorr]
            kwargs = dict(width=0.8, color=stacked_perf_color[ncorr])
            ax.bar(np.arange(len(trackers)), y, bottom=bottom, **kwargs)
            bottom += y
        ax.set_title(condition_labels[cond], color=perfcolors[cond], pad=6.)
        ax.set_xticks(np.arange(len(trackers)))
        labels = [ dict(Human="Human", TRU="%s pr." % condition_labels[cond][:3], IND="IND pr.")[tr] for tr in trackers]
        ax.set_xticklabels(labels, rotation=50, ha="right")
        ax.xaxis.set_tick_params(pad=0.)
        # beautify
        ax.set_ylim(0,1)
        if cn==0:
            ax.set_yticks([0,1])
            ax.set_ylabel("Fraction of\n 0,1,2,3 corr.", labelpad=3)
            for s in ("right", "top"):
                ax.spines[s].set_visible(False)
        else:
            ax.set_yticks([])
            for s in ("left", "right", "top"):
                ax.spines[s].set_visible(False)


# # #  SAVE  # # #

if SAVEFIG:
    fname = "./fig/Figure_2.png"
    print("> Save figure to file: %s" % fname)
    fig.savefig(fname)
    fname = "./fig/Figure_2.pdf"
    print("> Save figure to file: %s" % fname)
    fig.savefig(fname)
else:
    print("> Figure NOT saved!")
    pl.show()
