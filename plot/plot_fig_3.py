from mpl_settings import panellabel_fontkwargs, fig_width
import numpy as np
import pylab as pl
from scipy.stats import ttest_1samp


# # #  PARAMETERS  # # #

# A : Trial example
# B : Motion trees: stimulus
# C : Motion trees: observer
# D : Log-likelihood and significance

ZOOM = 2.
PLOT = ("A","B","C","D")
SAVEFIG = True

fname_data = "../data/fitResults_predict_MarApr2019.pkl.zip"

kals = ("IND", "GLO", "CNT", "CLU", "CLI", "CDH", "SDH")
conds = ("GLO", "CLU", "CDH")
# Remark: The 'GLM' (Global monochrome) condition in the dataset was a
#         control that we appended for 4 participants, which turned out
#         not to be relevant (and was not pre-registered).



# # #  \PARAMETERS  # # #

nKal = len(kals)
nCond = len(conds)


# # #  AUX FUNCTIONS  # # #

def calc_xycoords(condition, kaltracker):
    c = condition
    k = kaltracker
    l0 = df.loc[(df.cond == c) & (df.kal == "TRU")].ll.reset_index(drop=True, inplace=False)
    l = df.loc[(df.cond == c) & (df.kal == k)].ll.reset_index(drop=True, inplace=False)
    y = l - l0
    xm = kals.index(k)
    x = xm + np.linspace(-0.25, 0.25, len(y))
    return x, y

def p_string(p):
    thresh = np.array( (1e-4, 1e-3, 0.01, 0.05, 1.01) )
    pstr = ("∗"*4, "∗"*3, "∗"*2, "∗", "ns")
    return pstr[ (p < thresh).argmax() ]

def print_panel_label(letter, ax, dx=-0.03, dy=0.005, abs_x=None, abs_y=None):
    x,y = ax.bbox.transformed(fig.transFigure.inverted()).corners()[1]  # Top left in fig coords
    if abs_x is not None:
        x, dx = abs_x, 0.
    if abs_y is not None:
        y, dy = abs_y, 0.
    t = ax.text(x+dx, y+dy, letter, transform=fig.transFigure, **panellabel_fontkwargs)
    return t


# # #  \AUX FUNCTIONS  # # #

# For for the significance stars (arial has none)
from matplotlib.font_manager import FontProperties
font0 = FontProperties()
font0.set_family(["FreeSans"])

# # #  MAIN   # # #


pl.matplotlib.rc("figure", dpi=ZOOM*pl.rcParams['figure.dpi'])

import pandas
print("> Load data from file: %s" % fname_data)
df = pandas.read_pickle(fname_data, compression='gzip')


# # #  Setup Figure and axes  # # #
w,h = fig_width["twocol"], 0.385*fig_width["twocol"]
fig = pl.figure(figsize=(w, h))
ar = w/h
axes = dict()

# stimulus example
hi = 0.7/fig.get_size_inches()[1]
wi = hi/ar
b = 0.99-hi                   # Store to align the observer trees
rect = 0.035, b, wi, hi
ax = fig.add_axes(rect, aspect="equal", xticks=[], yticks=[])
axes["stim"] = ax

# Observer trees
l0 = 0.035 + wi + 0.03
hi = 0.48/fig.get_size_inches()[1]
wi = hi/ar * 432/288
for i,kal in enumerate(kals):
    li = l0 + i *(wi * 1.05)
    rect = li, b, wi, hi
    r0 = li + wi                # Store for the log likelihood axes
    ax = fig.add_axes(rect, aspect="auto", xticks=[], yticks=[])
    axes["obs_%s" % kal] = ax

axes["obs_CLU"].set_title(r"            simpler  $\longleftarrow$   Bayesian observer model   $\longrightarrow$  more complex", fontsize=8)

# Stimulus condition trees
top = b-0.03
for i,cond in enumerate(conds):
    bi = top - (i+1) * hi - i * 0.07
    rect = l0 - wi - 0.01, bi, wi, hi
    ax = fig.add_axes(rect, aspect="auto", xticks=[], yticks=[])
    axes["cond_%s" % cond] = ax

axes["cond_CLU"].set_ylabel("Stimulus structure", labelpad=4)

# Loglikelihood
delta = 0. # (r0-l0) * (0.1/nKal)    # left right extra space for the vertical scale
wi = r0 - l0 + 2 * delta
frac = 0.25  # vertical fraction of significance axes
hl, hs = (1-frac) * hi, frac * hi
for i,cond in enumerate(conds):
    bi = top - i * hi - i * 0.07 - hs
    rect = l0-delta, bi, wi, hs
    ax = fig.add_axes(rect, aspect="auto", xticks=[], yticks=[])
    axes["sig_%s" % cond] = ax
    bi = top - (i+1) * hi - i * 0.07
    rect = l0-delta, bi, wi, hl
    ax = fig.add_axes(rect, aspect="auto", xticks=[], yticks=[])
    axes["ll_%s" % cond] = ax


# # #  Set panel labels
print_panel_label(letter="A", ax=axes["stim"], abs_x=0.015, abs_y=0.94)
print_panel_label(letter="C", ax=axes["obs_IND"], dx=-0.015, abs_y=0.94)
print_panel_label(letter="B", ax=axes["cond_GLO"], dx=-0.025, dy=-0.01)


# # #  PLOT EXAMPLE MOTION  # # #
if "A" in PLOT:
    ax = axes["stim"]
    ax.imshow(pl.imread(f"./panel/global_motion_pred.png"))
    ax.set_frame_on(False)

# # #  PLOT TREES  # # #
if "C" in PLOT:
    for s in kals:
        ax = axes["obs_" + s]
        ax.imshow(pl.imread(f"./panel/pred_struct_{s}.png"))
        for name in ('bottom', 'top', 'right', 'left'):
            ax.spines[name].set_color("0.7")
            ax.spines[name].set_lw(0.5)
            ax.set_xticks([])
            ax.set_yticks([])

if "B" in PLOT:
    for s in conds:
        ax = axes["cond_" + s]
        ax.imshow(pl.imread(f"./panel/pred_struct_{s}.png"))
        for name in ('bottom', 'top', 'right', 'left'):
            ax.spines[name].set_color("0.7")
            ax.spines[name].set_lw(0.5)
            ax.set_xticks([])
            ax.set_yticks([])


# # #  PLOT LOG LIKELIHOOD AND SIGNIFICANCE  # # #
if "D" in PLOT:
    for nc, c in enumerate(conds):
        ax = axes["ll_%s" % c]
        axs = axes["sig_%s" % c]
        ax.hlines(0., -0.50, nKal-0.6, '0.3', zorder=1, lw=0.5)
        if c == "CLU":
            ax.yaxis.set_label_position("right")
            ax.set_ylabel(" "*4 + "Log-likelihood ratio \n" + " "*4 + "Bayesian model vs. stimulus structure", labelpad=18., linespacing=1.4)
        ax.set_xlim(-0.5, nKal-0.5)
        axs.set_xlim(-0.5, nKal-0.5)
        axs.set_frame_on(False)
        ymin, ymax = 100, -100
        for k in kals:
            if k == c:
                x,y = calc_xycoords(c, kaltracker=c if c != "GLM" else "GLO")
                d = 0.15
                kwargs = dict(zorder=0, fc='0.9', ec='0.9')
                patch = pl.Rectangle( (x[0] - d, -1000), x[-1]-x[0] + 2*d, 1100, **kwargs)
                ax.add_patch(patch)

            x,y = calc_xycoords(c, k)
            kwargs = dict(ms=0.8, c='k', zorder=2)
            ax.plot(x, y, "o", **kwargs)
            ym = np.mean(y)
            ymed = np.median(y)
            xm = np.mean(x)
            ax.hlines(ym, xm-0.35, xm+0.35, 'k', zorder=2, lw=0.5)
            ymin, ymax = min(ymin, np.min(y)), max(ymax, np.max(y))
            # T test
            t, p = ttest_1samp(-np.array(y), 0.)
            p = p/2 if t > 0 else (1 - p/2)  # Correct for one-sided
            s = "" if p < 0.05 else "NOT "
            print("  %d) %s fits %ssignificantly better than %s on structure %s (p = %.3e)." % (nc+1, c, s, k, c, p))
            ceff = c if c != "GLM" else "GLO"
            if k != ceff:
                pstr = p_string(p)
                fs = 5 if pstr=="ns" else 6
                axs.text(xm, 0.35, pstr, fontsize=fs, va='center', ha='center', fontproperties=font0)
        # Plot the *** grid
        kwargs = dict(lw=0.5)
        axs.hlines(0.9, 0., nKal-1, 'k', **kwargs)
        xg = np.arange(nKal).tolist()
        xtru = xg.pop(kals.index(c if c != "GLM" else "GLO"))
        axs.vlines(xg, 0.65, 0.91, 'k', **kwargs)
        axs.vlines(xtru, 0.1, 0.91, 'k', **kwargs)
        axs.set_ylim(0,1)
        # Scale
        d = 0.15 * (ymax - ymin)
        ytick_candidates = np.array([-300, -250, -200, -150, -100, -50])
        ytick = ytick_candidates[np.argmin(np.abs(0.75*ymin - ytick_candidates))]
        ax.set_ylim(ymin-d, ymax+d)
        ax.set_yticks([])
        ax.set_frame_on(False)
        # ytick
        ax.plot([nKal-1+0.45,nKal-1+0.41,nKal-1+0.41,nKal-1+0.45],[0,0,ytick,ytick], lw=0.5, color='k', zorder=2)
        ax.text(nKal-1+0.5, 0, "0", fontsize=6, ha='left', va='center')
        ax.text(nKal-1+0.5, ytick, "%.0f" % ytick, fontsize=6, ha='left', va='center')
        ax.set_xlabel("")
        ax.set_xticks([])
        axs.set_yticks([])


# # #  SAVE  # # #

if SAVEFIG:
    fname = "./fig/Fig3.png"
    print("> Save figure to file: %s" % fname)
    fig.savefig(fname)
    fname = "./fig/Fig3.pdf"
    print("> Save figure to file: %s" % fname)
    fig.savefig(fname)
else:
    print("> Figure NOT saved!")
    pl.show()
