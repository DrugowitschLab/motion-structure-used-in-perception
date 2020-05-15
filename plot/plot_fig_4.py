from mpl_settings import panellabel_fontkwargs, fig_width
import numpy as np
import pylab as pl
from scipy.stats import ttest_1samp
pi = np.pi


# # #  PARAMETERS  # # #

# A : Trial example
# B : Human performance GLO
# C : Example model fit with GLO prior
# D : Example model fit with IND prior
# E : Log-likelihood and significance

ZOOM = 1.5
PLOT = ("A","B","C","D","E")

SAVEFIG = True

fname_data = "../data/fitResults_predict_MarApr2019_SLSQP_grad.pkl.zip"
fname_panelBCD_data_TRU = "../data/predict_model_performance_plottingdata_cond_GLO_model_TRU.npz"
fname_panelBCD_data_IND = "../data/predict_model_performance_plottingdata_cond_GLO_model_IND.npz"

exSubjIdx = 4       # example subject for panels C and D (python indexing)

kals = ("IND", "GLO", "CNT", "CLU", "CLI", "CDH", "SDH")
conds = ("GLO", "CLU", "CDH")

std_chance = 2 * np.pi / np.sqrt(12)

# Kalman filter covariance matrices at end of trial
Sig_TRU = {"IND" : np.array([[1.94829133, 0.        ],
                             [0.        , 1.94829133]]),
           "GLO" : np.array([[0.06588445, 0.01092615],
                             [0.01092615, 0.06588445]]),
           "CLU" : np.array([[0.08203185, 0.02707355],
                             [0.02707355, 0.08203185]]),
           "CDH" : np.array([[0.08181467, 0.01354736],
                             [0.01354736, 0.44583885]])
          }


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

# Ellipse plotting from: https://stackoverflow.com/a/25022642
from matplotlib.patches import Ellipse
def eigsorted(cov):
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    return vals[order], vecs[:,order]

# # #  MAIN   # # #


pl.matplotlib.rc("figure", dpi=ZOOM*pl.rcParams['figure.dpi'])

# Data for E
import pandas
print("> Load data from file: %s" % fname_data)
df = pandas.read_pickle(fname_data, compression='gzip')

# Data for B,C,D
if fname_panelBCD_data_TRU is not None:
    print("> Load performance and model data for example subject: %d" % exSubjIdx)
    exCond = str(np.load(fname_panelBCD_data_TRU)["cond"])
    Sig_g, Sig_r = np.load(fname_panelBCD_data_TRU)["sigma_err"]
    exDgTRU, exDrTRU = np.load(fname_panelBCD_data_TRU)["delta_to_kal"][exSubjIdx]
    exParamTRU = np.load(fname_panelBCD_data_TRU)["fitparams_a_b_pL"][exSubjIdx]
    exDgIND, exDrIND = np.load(fname_panelBCD_data_IND)["delta_to_kal"][exSubjIdx]
    exParamIND = np.load(fname_panelBCD_data_IND)["fitparams_a_b_pL"][exSubjIdx]
    exSubj = np.load(fname_panelBCD_data_TRU)["subjects"][exSubjIdx]

# # # # # # # # # # # # # # # # # #
# # #  Setup Figure and axes  # # #
# # # # # # # # # # # # # # # # # #

w,h = fig_width["twocol"], 0.385 * 1.55 * fig_width["twocol"]
fig = pl.figure(figsize=(w, h))
ar = w/h
axes = dict()

# stimulus example
hi = 0.9/fig.get_size_inches()[1]
wi = hi/ar
b = 0.95-hi                   # Store to align the observer trees
li = 0.05
rect = li, b, wi, hi
ax = fig.add_axes(rect, aspect="equal", xticks=[], yticks=[])
axes["stim"] = ax

# Performance
for i,axname in enumerate(("perf","model_TRU","model_IND")):
    li += wi * 1.77
    hi = 1.05/fig.get_size_inches()[1]
    wi = hi/ar
    b = 0.965-hi                   # Store to align the observer trees
    rect = li, b, wi, hi
    ax = fig.add_axes(rect, aspect="equal", xticks=[-pi,pi], yticks=[-pi,pi])
    axes[axname] = ax


# Observer trees
wiold = 0.8/fig.get_size_inches()[1]/ar
l0 = 0.035 + wiold + 0.015
hi = 0.48/fig.get_size_inches()[1]
wi = hi/ar * 432/288
b = 0.99-hi-0.40                   # Store to align the observer trees
for i,kal in enumerate(kals):
    li = l0 + i *(wi * 1.05)
    rect = li, b, wi, hi
    r0 = li + wi                # Store for the log likelihood axes
    ax = fig.add_axes(rect, aspect="auto", xticks=[], yticks=[])
    axes["obs_%s" % kal] = ax

axes["obs_CLU"].set_title(r"            simpler  $\longleftarrow$    Motion structure prior   $\longrightarrow$  more complex", fontsize=8)

# Stimulus condition trees
top = b-0.03
for i,cond in enumerate(conds):
    bi = top - (i+1) * hi - i * 0.07/1.55
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
    bi = top - i * hi - i * 0.07/1.55 - hs
    rect = l0-delta, bi, wi, hs
    ax = fig.add_axes(rect, aspect="auto", xticks=[], yticks=[])
    axes["sig_%s" % cond] = ax
    bi = top - (i+1) * hi - i * 0.07/1.55
    rect = l0-delta, bi, wi, hl
    ax = fig.add_axes(rect, aspect="auto", xticks=[], yticks=[])
    axes["ll_%s" % cond] = ax


# # #  Set panel labels
print_panel_label(letter="A", ax=axes["stim"], abs_x=0.015, abs_y=0.97)
print_panel_label(letter="B", ax=axes["stim"], abs_x=0.21, abs_y=0.97)
print_panel_label(letter="C", ax=axes["stim"], abs_x=0.465, abs_y=0.97)
print_panel_label(letter="D", ax=axes["stim"], abs_x=0.74, abs_y=0.97)
print_panel_label(letter="E", ax=axes["cond_GLO"], abs_x=0.015, abs_y=0.63)


# # #  PLOT EXAMPLE MOTION  # # #
if "A" in PLOT:
    ax = axes["stim"]
    ax.imshow(pl.imread(f"./panel/global_motion_pred.png"))
    ax.set_frame_on(False)


# # #  PLOT HUMAN PERF TO OPT AND CHANCE # # #
if "B" in PLOT:
    ax = axes["perf"]
    kwargs = dict(s=1, c='k', edgecolors=None, zorder=1)
    ax.scatter(Sig_g, Sig_r , **kwargs)
    ax.plot(Sig_g[exSubjIdx], Sig_r[exSubjIdx], 'o', color="orange", ms=2.2, mew=0.0, zorder=2)
    # reference box and id-line
    cg = cr = std_chance
    optg = np.sqrt(Sig_TRU[exCond][0,0])
    optr = np.sqrt(Sig_TRU[exCond][1,1])
    ax.plot([cg, optg], [cr, optr], lw=0.5, linestyle=":", color="0.3", zorder=0)
    kwargs = dict(lw=0.5, linestyle=":", edgecolor="0.3", facecolor=None, fill=False, zorder=-1)
    patch = pl.Rectangle((optg, optr), cg-optg, cr-optr, **kwargs)
    ax.add_patch(patch)
    annokwargs = dict(fontsize=6, color="0.0",
                      arrowprops=dict(edgecolor="0.0", facecolor="0.0", shrink=0.0, width=0.5, lw=0.0, headwidth=3, headlength=5))
    ax.annotate('Bayes opt.', xy=(optg+0.01, optr+0.03), xytext=(0.35, 1.20), ha="left", va="baseline", **annokwargs)
    ax.annotate('Chance', xy=(cg-0.03, cr-0.01), xytext=(1.4, 1.5), ha="right", va="baseline", **annokwargs)
    ax.annotate('Subj. in C & D', xy=(Sig_g[exSubjIdx]+0.01, Sig_r[exSubjIdx]), xytext=(2.0, 0.4), ha="right", va="baseline", **annokwargs)
    # beautify
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    ax.set_xlabel("Pred. error: green dot", labelpad=1, color="forestgreen", fontsize=7)
    ax.set_ylabel("Pred. error: red dot", labelpad=1, color="crimson", fontsize=7)
    ax.set_xticks([0.0, 0.5, 1.0, 1.5, 2.0])
    ax.set_yticks([0.0, 0.5, 1.0, 1.5, 2.0])
    ax.set_title("Performance in %s" % exCond, pad=2, fontsize=6)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)



if "C" in PLOT:
    ax = axes["model_TRU"]
    # data
    x, y = exDgTRU, exDrTRU
    a, b, pL = exParamTRU
    # trial dots
    kwargs = dict(s=0.25, c='k', edgecolors=None, zorder=1)
    ax.scatter(x, y , **kwargs)
    # ellipse
    S = Sig_TRU[exCond]
    cov = a * np.eye(2) + b * S / np.mean(S.diagonal())
    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))
    for nstd in (1.0, 2.0, 3.0):
        w, h = 2 * nstd * np.sqrt(vals)
        kwargs = dict(fill=False, facecolor="none", edgecolor='%.2f' % ((nstd-0.75)/3.), linewidth=0.5, zorder=0)
        ell = Ellipse(xy=(np.mean(x), np.mean(y)), width=w, height=h, angle=theta, **kwargs)
        ax.add_artist(ell)
    # dotted lines
    kwargs = dict(lw=0.5, linestyles="dotted", edgecolor="0.3", zorder=-1)
    ax.hlines(0, -pi, pi, **kwargs)
    ax.vlines(0, -pi, pi, **kwargs)
    # beautify
    ax.set_xlim(-pi, pi)
    ax.set_ylim(-pi, pi)
    ax.set_xticks([-pi, 0, +pi])
    ax.set_xticklabels([r"${-}\pi$", "0", r"$\pi$"])
    ax.set_yticks([-pi, 0, +pi])
    ax.set_yticklabels([r"${-}\pi$", "0", r"$\pi$"])
    ax.set_title("Subj. #%d: %s obs. model" % (int(exSubjIdx)+1, exCond), pad=2, fontsize=6)
    ax.set_xlabel(r"$\varphi_\mathrm{human} - \mu_\mathrm{kal}$: green", labelpad=1, color="forestgreen", fontsize=7)
    ax.set_ylabel(r"$\varphi_\mathrm{human} - \mu_\mathrm{kal}$: red", labelpad=1, color="crimson", fontsize=7)


if "D" in PLOT:
    ax = axes["model_IND"]
    # data
    x, y = exDgIND, exDrIND
    a, b, pL = exParamIND
    # trial dots
    kwargs = dict(s=0.25, c='k', edgecolors=None, zorder=1)
    ax.scatter(x, y , **kwargs)
    # ellipse
    S = Sig_TRU["IND"]
    cov = a * np.eye(2) + b * S / np.mean(S.diagonal())
    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))
    for nstd in (1.0, 2.0, 3.0):
        w, h = 2 * nstd * np.sqrt(vals)
        kwargs = dict(fill=False, facecolor="none", edgecolor='%.2f' % ((nstd-0.75)/3.), linewidth=0.5, zorder=0)
        ell = Ellipse(xy=(np.mean(x), np.mean(y)), width=w, height=h, angle=theta, **kwargs)
        ax.add_artist(ell)
    # dotted lines
    kwargs = dict(lw=0.5, linestyles="dotted", edgecolor="0.3", zorder=-1)
    ax.hlines(0, -pi, pi, **kwargs)
    ax.vlines(0, -pi, pi, **kwargs)
    # beautify
    ax.set_xlim(-pi, pi)
    ax.set_ylim(-pi, pi)
    ax.set_xticks([-pi, 0, +pi])
    ax.set_xticklabels([r"${-}\pi$", "0", r"$\pi$"])
    ax.set_yticks([-pi, 0, +pi])
    ax.set_yticklabels([r"${-}\pi$", "0", r"$\pi$"])
    ax.set_title("Subj. #%d: %s obs. model" % (int(exSubjIdx)+1, "IND",), pad=2, fontsize=6)
    ax.set_xlabel(r"$\varphi_\mathrm{human} - \mu_\mathrm{kal}$: green", labelpad=1, color="forestgreen", fontsize=7)
    ax.set_ylabel(r"$\varphi_\mathrm{human} - \mu_\mathrm{kal}$: red", labelpad=1, color="crimson", fontsize=7)



# # #  PLOT TREES  # # #
if "E" in PLOT:
    for s in kals:
        ax = axes["obs_" + s]
        ax.imshow(pl.imread(f"./panel/pred_struct_{s}.png"))
        for name in ('bottom', 'top', 'right', 'left'):
            ax.spines[name].set_color("0.7")
            ax.spines[name].set_lw(0.5)
            ax.set_xticks([])
            ax.set_yticks([])

if "E" in PLOT:
    for s in conds:
        ax = axes["cond_" + s]
        ax.imshow(pl.imread(f"./panel/pred_struct_{s}.png"))
        for name in ('bottom', 'top', 'right', 'left'):
            ax.spines[name].set_color("0.7")
            ax.spines[name].set_lw(0.5)
            ax.set_xticks([])
            ax.set_yticks([])



# # #  PLOT LOG LIKELIHOOD AND SIGNIFICANCE  # # #
if "E" in PLOT:
    for nc, c in enumerate(conds):
        ax = axes["ll_%s" % c]
        axs = axes["sig_%s" % c]
        ax.hlines(0., -0.50, nKal-0.6, '0.5', zorder=1, lw=0.5)
        if c == "CLU":
            ax.yaxis.set_label_position("right")
            ax.set_ylabel(" "*4 + "Log-likelihood ratio to ground truth", labelpad=20., linespacing=1.4)
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
            # highlight example participant
            if c=="GLO" and k=="IND":
                ax.plot(x[exSubjIdx], y[exSubjIdx], 'o', color="orange", ms=2.0, mew=0.0, zorder=3)
            # T test
            t, p = ttest_1samp(-np.array(y), 0.)
            p = p/2 if t > 0 else (1 - p/2)  # Correct for one-sided
            s = "" if p < 0.05 else "NOT "
            print("  %d) %s fits %ssignificantly better than %s on structure %s (p = %.3e)." % (nc+1, c, s, k, c, p))
            ceff = c if c != "GLM" else "GLO"
            if k != ceff:
                pstr = p_string(p)
                fs = 6 if pstr=="ns" else 6
                font0.set_size(fs)
                axs.text(xm, 0.35, pstr, fontsize=fs, va='center', ha='center', fontproperties=font0)
                # print mean value
                s = "(%.1f)" % ym
                axs.text(xm, -0.25, s, fontsize=4, color="0.15", va='center', ha='center')
        # Plot the *** grid
        kwargs = dict(lw=0.5, color="0.15")
        axs.hlines(0.9, -0.002, nKal-1+0.002, 'k', **kwargs)
        xg = np.arange(nKal).tolist()
        xtru = xg.pop(kals.index(c if c != "GLM" else "GLO"))
        axs.vlines(xg, 0.65, 0.91, 'k', **kwargs)
        # axs.vlines(xg, 0.1, 0.25, 'k', **kwargs)
        axs.vlines(xtru, 0.1, 0.91, 'k', **kwargs)
        axs.set_ylim(-0.20,0.95)
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
    fname = "./fig/Figure_4.png"
    print("> Save figure to file: %s" % fname)
    fig.savefig(fname)
    fname = "./fig/Figure_4.pdf"
    print("> Save figure to file: %s" % fname)
    fig.savefig(fname)
else:
    print("> Figure NOT saved!")
    pl.show()
