from mpl_settings import panellabel_fontkwargs, fig_width
import numpy as np
import pylab as pl
import pandas as pd
from scipy.stats import ttest_1samp

# # # # PARAMETERS # # # #

# A : Observer model sketches
# B : Momentum-free observer and Weber's law observer performance like Fig 2B
# C : Log-likelihood for comp. obs.

ZOOM = 2.0
SAVEFIG = True
RANDOMDATA = False           # For fast plotting during dev.
PLOT = ("A","B","C")
PRINTPSTARS = True

tracker_labels = dict(Human="Human", TRU="Correct prior", IND="IND prior")

# Panel B
visible_trackers = ("TRU", "IND")
fname_data_perf = {
    "momentumfree" : "../data/dataframe_MOT_MarchApril_2019_noInertia_noisefree_human_and_sim.pkl.zip",
    "humaninspired": "../data/dataframe_MOT_MarchApril_2019_velodep_with_stochastic_QR_beta_4_rho_5_human_and_sim.pkl.zip",
    }
perf_level = dict(chance=3*3/7, thresh=2.15, perfect=3.)

# Panel C
ll_observer = "computational"
ll_column = "p(data|m)"   # Select LL estimator
fname_data_logL = {
    "computational" : "../data/MOT_MarchApril_2019_model_loglikelihoods.pkl.zip",
    }

alpha = 1.0
kals = ("IND", "GLO", "CNT", "CLU", "CLI", "CDH")
conds = ("IND", "GLO", "CNT", "CDH1", "CDH2")


# # # # \PARAMETERS # # # #


# # #  AUX FUNCTIONS  # # #
# For for the significance stars (arial has none)
from matplotlib.font_manager import FontProperties
font0 = FontProperties()
font0.set_family(["FreeSans"])

observers = ("momentumfree", "humaninspired")  # for panel B
observernames = { "momentumfree" : "Momentum-free observer",
                  "computational" : "Computational observer",
                  "humaninspired" : "Weber's law observer"
                }

obscolor = {"momentumfree" : "#6ce700ff",
            "computational": "#00c3d7ff",
            "humaninspired": "#c600d0ff"}

condition_label = { "IND" : r"IND",
                     "GLO" : r"GLO",
                     "CNT" : r"CNT",
                     "CDH1" : r"CDH$_\mathrm{1}$",
                     "CDH2" : r"CDH$_\mathrm{2}$"
                   }

kalnamedict = {"IND" : "independent_test",
               "GLO" : "global",
               "CNT" : "counter",
               "CDH1" : "hierarchy_124",
               "CDH2" : "hierarchy_127" }

inv_condition_labels = { "IND" : "independent_test",
                          "GLO" : "global",
                          "CNT" : "counter",
                          "CDH" : "hierarchy_124",
                          "CDH1" : "hierarchy_124",
                          "CDH2" : "hierarchy_127"
                     }

perfcolors = {"IND" : "0.2",
              "GLO" : "limegreen",
              "CNT" : "cornflowerblue",
              "CDH1" : "orange",
              "CDH2" : "tomato"
              }

tracker_hatches = {
    "Human" : None,
    "TRU" : None,
    "IND" : "////",
    }

def calc_xycoords(condition, kaltracker, observer):
    df = Df["ll"][observer]
    c = condition
    k = kaltracker
    l0 = df.loc[(df.condition == kalnamedict[c]) & (df.tracker == "TRU")][ll_column].reset_index(drop=True, inplace=False)
    l = df.loc[(df.condition == kalnamedict[c]) & (df.tracker == k)][ll_column].reset_index(drop=True, inplace=False)
    y = l - l0
    xbias = 0.0
    xm = 1 * kals.index(k[:3]) + xbias
    x = xm + np.linspace(-0.30, 0.30, len(y))
    return x, y, l0

def p_string(p):
    thresh = np.array( (1e-4, 1e-3, 0.01, 0.05, 1.01) )
    pstr = ("∗"*4, "∗"*3, "∗"*2, "∗", "ns")
    if np.isnan(p):
        return "ns"
    else:
        return pstr[ (p < thresh).argmax() ]

def print_panel_label(letter, ax, dx=-0.03, dy=0.005, abs_x=None, abs_y=None, transform=None):
    if transform is None:
        transform = fig.transFigure
    x,y = ax.bbox.transformed(transform.inverted()).corners()[1]  # Top left in fig coords
    if abs_x is not None:
        x, dx = abs_x, 0.
    if abs_y is not None:
        y, dy = abs_y, 0.
    t = ax.text(x+dx, y+dy, letter, transform=transform, **panellabel_fontkwargs)
    return t

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

def light_color(colname, weight=0.3):
    c4 = pl.matplotlib.colors.to_rgba_array(colname)
    w4 = pl.matplotlib.colors.to_rgba_array("white")
    weights = weight, 1-weight
    avg4 = (np.vstack([c4,w4]).T * weights).sum(1) / np.sum(weights)
    return avg4

def grayed_color(colname, weight=0.3):
    c4 = pl.matplotlib.colors.to_rgba_array(colname)
    w4 = pl.matplotlib.colors.to_rgba_array("0.9")
    weights = weight, 1-weight
    avg4 = (np.vstack([c4,w4]).T * weights).sum(1) / np.sum(weights)
    return avg4

# # # # LOAD AND PROCESS DATA # # # #
Df = dict()

if "B" in PLOT:
    for obsname,fname in fname_data_perf.items():
        if fname:
            print("> Load data from file: %s" % fname)
            Df[obsname] = pd.read_pickle(fname, compression='gzip')

if "C" in PLOT:
    Df["ll"] = dict()
    for obs,fname_data in fname_data_logL.items():
        print("> Load data from file: %s" % fname_data)
        df = pd.read_pickle(fname_data, compression='gzip')
        Df["ll"][obs] = df.loc[df.alpha==alpha]

# # # # SETUP FIG # # # #
# For for the significance stars (arial has none)
from matplotlib.font_manager import FontProperties
font0 = FontProperties()
font0.set_family(["FreeSans"])
font0.set_size(5)

pl.matplotlib.rc("figure", dpi=ZOOM*pl.matplotlib.rcParams['figure.dpi'])

# # #  Setup Figure and axes  # # #
w,h = fig_width["onecol"], 5.25
fig = pl.figure(figsize=(w, h))
ar = w/h
axes = dict()

# observer models
l0 = 0.25/w
hi = 0.65/h
wi = 3.0/w
bi = 1 - hi - 0.100/h
rect = l0, bi, wi, hi
ax = fig.add_axes(rect, aspect="auto", xticks=[], yticks=[])
axes["observers"] = ax

# performance bars
l1 = l0 + 0.16 / fig.get_size_inches()[0]
hi = 0.9 / fig.get_size_inches()[1]
bi = bi - hi - 0.3875 / fig.get_size_inches()[1]
rect = l1, bi, 0.98-l1, hi
ax = fig.add_axes(rect, aspect="auto")
axes["bars"] = ax

# Observer trees
l1 = l0 + 0.32 / fig.get_size_inches()[0]
hi = 0.50/fig.get_size_inches()[1]
wi = hi/ar * 240/300
bi = bi - hi - 0.58/h
for i,kal in enumerate(kals):
    li = l1 + i *(wi * 1.05) + 0.025 * wi
    rect = li, bi, wi, hi
    r0 = li + wi +  0.025 * wi               # Store for the log likelihood axes
    ax = fig.add_axes(rect, aspect="auto", xticks=[], yticks=[])
    axes["obs_%s" % kal] = ax

axes["obs_CLU"].set_title(r"simpler  $\longleftarrow$   Motion structure prior   $\longrightarrow$  more complex" + " "*20, fontsize=6, pad=4)

# Log-likelihood main axes
b = 0.0625/fig.get_size_inches()[1]
wi = r0 - l1
hall = bi - b # - 0.05/fig.get_size_inches()[1]
hi = hall / len(conds)
d = 0.05/fig.get_size_inches()[1]
for ci,c in enumerate(reversed(conds)):
    rect = l1, b + ci * hi + 0.7*hi - d, wi, 0.3 * hi
    ax = fig.add_axes(rect, aspect="auto", xticks=[], yticks=[])
    axes["sig_%s" % c] = ax
    rect = l1, b + ci * hi, wi, 0.7 * hi-d
    ax = fig.add_axes(rect, aspect="auto", xticks=[], yticks=[])
    axes["logL_%s" % c] = ax
    ax.set_ylabel(c, rotation="horizontal", ha="right", va="center", color=perfcolors[c], fontsize=8)

# Comp obs. title
linex = np.array([l1+0.015, l1+wi-0.015]),  np.array([l1+wi/2]*2)
liney = np.array([bi+0.13]*2), np.array([bi+0.13, bi+0.1350])
kwargs = dict(lw=0.5, color=obscolor["computational"], zorder=5, clip_on=False, transform=fig.transFigure)
for lx, ly in zip(linex,liney):
    ax.plot( lx, ly, **kwargs)
kwargs = dict(fontsize=6, va="baseline", ha="center", color=obscolor["computational"], transform=fig.transFigure)
ax.text( l1+wi/2, bi+0.1415, observernames["computational"], **kwargs)

# # #  PLOT OBSERVER MODEL SKETCHES  # # #
if "A" in PLOT:
    ax = axes["observers"]
    ax.imshow(pl.imread(f"./panel/observermodelspictogram_wide.png"))
    ax.set_frame_on(False)
    print_panel_label("A", ax, abs_x=0.025, abs_y=0.97)


# # #  PLOT ALT. OBSERVERS' PERFORMANCES  # # #
if "B" in PLOT:
    print_panel_label("B", ax, abs_x=0.025, abs_y=0.795)
    # PERPARE GENERAL VARS
    pc = perf_level["chance"]
    pp = perf_level["perfect"]
    nObs = len(observers)
    nCond = len(conds)
    nTracker = len(visible_trackers)
    ymin, ymax = 10.,0.
    from scipy.stats import sem
    ax = axes["bars"]
    # ITERATE OVER OBSERVER MODELS
    for oi, obsname in enumerate(observers):
        df = Df[obsname]
        subjects = np.unique(df.subject)
        nReps = np.max(df["repetition"])
        nTrials = max(df['trial'])
        eps = 0.016
        for tn, trac in enumerate(visible_trackers):
            hatch=tracker_hatches[trac]
            w = 0.8/nCond - eps
            xbias = -0.4 + tn + oi * nTracker
            x = xbias + np.linspace(0, 0.8, nCond+1)[:-1] + eps/2
            Y = [] # condition, subj
            for cond in conds:
                y = []
                for subj in subjects:
                    if RANDOMDATA:
                        y.append(np.random.uniform(1.35, 2.95))
                    else:
                        idx = (df.subject == subj) & (df.condition == inv_condition_labels[cond]) & (df.tracker == trac)
                        if idx.sum() == 0:
                            raise Exception("> Error: No data for %s:%s:%s!" % (subj, cond, trac))
                        d = df.loc[idx]
                        y.append(np.mean(d["numCorrect"]))
                Y.append(y)
            # PLOT BARS
            colors = [perfcolors[cond] for cond in conds]
            if trac == "Human":
                fc = [light_color(c, weight=0.6) for c in colors]
            else:
                fc = [light_color(c, weight=0.3) for c in colors]
            ec = [light_color(c, weight=0.7) for c in colors]
            kwargs = dict(align='edge', color=fc, hatch=hatch, width=w, edgecolor=ec,
                          linewidth=0.75, ecolor="0.5", capsize=1.0, zorder=1, error_kw=dict(lw=0.75))
            ymean = np.mean(Y, axis=1)
            yerr = sem(Y, axis=1)
            if trac in visible_trackers:
                ax.bar(x, ymean, yerr=yerr, **kwargs)
            # TRACKER NAMES
            if trac == "TRU" and obsname == "humaninspired":
                for xi, cond in zip(x, conds):
                    s = cond
                    c = perfcolors[cond]
                    kwargs = dict(color=c, rotation="vertical", ha="center", va="bottom")
                    delta = 0.015 if "hierarchy" in cond else 0.005
                    ax.text(float(xi+w/2) + delta, perf_level["chance"]+0.025, s, **kwargs)
            # PLOT DOTS
            xsubject = np.linspace(0, w, len(subjects) + 4)[2:-2]
            x = (x[:,None] + xsubject).flatten()
            y = np.array(Y).flatten()
            weight = 1. if trac == "Human" else 0.8
            c = [light_color(perfcolors[cond], weight=weight) for cond in conds for subj in subjects]
            kwargs = dict(s=1., c=c, marker='.', zorder=2)
            if trac in visible_trackers:
                ax.scatter(x, y, **kwargs)
            # background patches and titles
            fc = light_color(obscolor[obsname], weight=0.07)
            patch = pl.Rectangle( ( (oi * nTracker)-0.5, 0.), nTracker, 5, fc=fc, ec="0.8", lw=0.5, zorder=-1)
            ax.add_patch(patch)
            linex = np.array([-0.43, nTracker-0.57]), np.array([nTracker/2-0.5, nTracker/2-0.5])
            liney = np.array([3.1, 3.1]), np.array([3.1, 3.15])
            for lx,ly in zip(linex, liney):
                kwargs = dict(lw=0.5, color=obscolor[obsname], zorder=5, clip_on=False)
                ax.plot( (oi*nTracker) + lx, ly, **kwargs)
            ax.text((oi*nTracker) + nTracker/2 - 0.5 , 3.2, observernames[obsname], fontsize=6, va="baseline", ha="center", color=obscolor[obsname])

    # # #  Beautify
    xmin, xmax = -0.5, (nObs * nTracker)-0.5
    ax.set_xlim(xmin, xmax)

    # dotted lines
    for y in perf_level.values():
        ymax = max(y, ymax)
        ymin = min(y, ymin)
        kwargs = dict(dashes=[2, 2], c="0.3", lw=0.5, zorder=0)
        ax.plot([xmin,xmax], [y]*2, **kwargs)

    # yticks
    auto_yscale(ax=ax, data=np.array((ymin, ymax)), tol=0.03)
    ax.set_ylabel("Performance\navg ± sem", labelpad=-24, fontsize=7)
    ax.set_yticks([perf_level["chance"], perf_level["perfect"]])
    ax.set_yticklabels([f"\n{pc:.2f}\n(chance)", f"\n{pp:.2f}\n(perfect)"])

    # xticks
    ax.set_xticks(np.arange(nObs * nTracker))
    ax.set_xticklabels([ dict(Human="Human", TRU="Correct prior", IND="IND prior")[tr] for tr in visible_trackers] * nObs, fontsize=6)


# # #  PLOT LOG LIKELIHOODS  # # #
if "C" in PLOT:
    # Observer models
    fname = lambda kal: "./panel/mot_struct_%s_f3n.png" % kal
    for k in kals:
        ax = axes["obs_%s" % k]
        for name in ('bottom', 'top', 'right', 'left'):
            ax.spines[name].set_color("0.7")
            ax.spines[name].set_lw(0.5)
        try:
            ax.imshow(pl.imread(fname(k)))
        except:
            print("Error plotting '%s'."  % fname(k))
            ax.text(0.5, 0.5, k, va='center', ha='center')
    # plot ll
    for nc, c in enumerate(conds):
        nKal = len(kals)
        ax = axes["logL_%s" % c]
        axs = axes["sig_%s" % c]
        axs.set_xlim(-0.5, nKal-0.5)
        axs.set_frame_on(False)
        ax.hlines(0., -0.48, 1*nKal-0.58, '0.3', zorder=1, lw=0.5)
        ymin, ymax = 100, -100
        for oi,o in enumerate((ll_observer,)):
            for k in kals:
                if k == c[:3]:
                    x,y,l0 = calc_xycoords(c, kaltracker=c, observer=o)
                    print("> L0(%s) = %.3f ± %.3f (std)" % (c, np.mean(l0), np.std(l0)))
                    d = 0.15
                    kwargs = dict(zorder=0, fc='0.9', ec='0.9')
                    patch = pl.Rectangle( (x[0] - d, -1000), x[-1]-x[0] + 2*d, 1100, **kwargs)
                    ax.add_patch(patch)
                x,y,l0 = calc_xycoords(c, k, o)
                kwargs = dict(ms=1.25, c=obscolor[o], mfc=obscolor[o], mew=0., zorder=2)
                ax.plot(x, y, "o", **kwargs)
                ym = np.mean(y)
                xm = np.mean(x)
                ax.hlines(ym, xm-0.35, xm+0.35, 'k', zorder=2, lw=0.5)
                ymin, ymax = min(ymin, np.min(y)), max(ymax, np.max(y))
                # T test
                t, p = ttest_1samp(-np.array(y), 0.)
                p = p/2 if t > 0 else (1 - p/2)  # Correct for one-sided
                s = "" if p < 0.05 else "NOT "
                print(" %d.%d) %s: %s fits %ssignificantly better than %s on structure %s (p = %.3e)." % (nc+1, oi+1, o, c, s, k, c, p))
                ceff = c if c != "GLM" else "GLO"
                if (k != ceff[:3]) and PRINTPSTARS:
                    pstr = p_string(p)
                    fs = 6 if pstr=="ns" else 6
                    font0.set_size(fs)
                    axs.text(xm, 0.35, pstr, fontsize=fs, va='center', ha='center', fontproperties=font0)
                    # print mean value
                    s = "(%.1f)" % ym
                    axs.text(xm, -0.25, s, fontsize=4, color="0.15", va='center', ha='center')
        # Plot the *** grid
        kwargs = dict(lw=0.5, color="0.15")
        axs.hlines(0.9, -0.003, nKal-1+0.003, 'k', **kwargs)
        xg = np.arange(nKal).tolist()
        xtru = xg.pop(kals.index(c[:3]))
        axs.vlines(xg, 0.65, 0.91, 'k', **kwargs)
        axs.vlines(xtru, 0.1, 0.91, 'k', **kwargs)
        axs.set_ylim(-0.20,0.95)
        # Scale
        d = 0.15 * (ymax - ymin)
        ytick_candidates = np.array([-300, -250, -200, -150, -100, -50, -40, -30, -20, -10])
        ytick = ytick_candidates[np.argmin(np.abs(0.75*ymin - ytick_candidates))]
        ax.set_ylim(ymin-d, ymax+d)
        ax.set_yticks([])
        ax.set_frame_on(False)
        ax.plot([nKal-1+0.49,nKal-1+0.45,nKal-1+0.45,nKal-1+0.49],[0,0,ytick,ytick], lw=0.5, color='k', zorder=2)
        ax.text(nKal-1+0.53, 0, "0", fontsize=6, ha='left', va='center')
        ax.text(nKal-1+0.53, ytick, "%.0f" % ytick, fontsize=6, ha='left', va='center')
        ax.set_xlim(-0.5, 1*nKal-0.5)

    # General labels ( in fig coords):
    print_panel_label("C", ax, abs_x=0.025, abs_y=0.505)
    ax.text(0.035, 0.185, "Stimulus condition", fontsize=8, rotation="vertical", va="center", ha="center", transform=fig.transFigure)
    ax.text(0.97, 0.185, "Log-likelihood ratio to ground truth", fontsize=7, rotation="vertical", va="center", ha="center", transform=fig.transFigure)


# # #  DONE  # # #
if SAVEFIG:
    import os
    figfname = os.path.join("./fig/", "Figure_3.png")
    fig.savefig(figfname, dpi=600)
    print(" > Figure saved to file: %s" % figfname)
    figfname = os.path.join("./fig/", "Figure_3.pdf")
    fig.savefig(figfname, dpi=600)
    print(" > Figure saved to file: %s" % figfname)
else:
    print(" > Not saving figure.")
    pl.show()
