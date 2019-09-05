from mpl_settings import panellabel_fontkwargs, fig_width, cm2inch
import numpy as np
import pylab as pl

# # #  PARAMETERS  # # #

# A : Cartoon
# B : Bias variance decompositon data

ZOOM = 2.
PLOT = ("A","B")

conditions = ['GLO', 'CLU', 'CDH']
cmap = pl.cm.gist_rainbow

SAVEFIG = True
fname_data = "../data/bias_variance_analysis_prediction_MarApr2019.pkl.zip"

# # #  AUX FUNCTIONS  # # #

def print_panel_label(letter, ax, dx=-0.03, dy=0.005, abs_x=None, abs_y=None):
    x,y = ax.bbox.transformed(fig.transFigure.inverted()).corners()[1]  # Top left in fig coords
    if abs_x is not None:
        x, dx = abs_x, 0.
    if abs_y is not None:
        y, dy = abs_y, 0.
    t = ax.text(x+dx, y+dy, letter, transform=fig.transFigure, **panellabel_fontkwargs)
    return t

def dark_color(col, weight=0.7):
    c4 = col # pl.matplotlib.colors.to_rgba_array(colname)
    k4 = pl.matplotlib.colors.to_rgba_array("black")
    weights = weight, 1-weight
    avg4 = (np.vstack([c4,k4]).T * weights).sum(1) / np.sum(weights)
    return avg4

# # #  \AUX FUNCTIONS  # # #

# # #  MAIN   # # #
pl.matplotlib.rc("figure", dpi=ZOOM*pl.rcParams['figure.dpi'])

if fname_data is not None:
    import pandas
    print("> Load data from file: %s" % fname_data)
    df = pandas.read_pickle(fname_data, compression='gzip')


# # #  Setup Figure and axes  # # #
w,h = fig_width["onecol"], 0.45*fig_width["onecol"]
fig = pl.figure(figsize=(w, h))
ar = w/h
axes = dict()

# Cartoon
wi = 0.3*fig_width["onecol"] / fig.get_size_inches()[0]
hi = cm2inch(3.70000) / fig.get_size_inches()[1]
b = 0.95-hi
rect = 0.05, b, wi, hi
ax = fig.add_axes(rect, aspect="auto", xticks=[], yticks=[])
axes["sketch"] = ax

# Data
wi = 0.47
hi = 0.77
b = 0.95-hi
rect = 0.98-wi, b, wi, hi
ax = fig.add_axes(rect, aspect="auto", xticks=[0,0.5,1], yticks=[0,0.5,1])
ax.set_xlabel("Bla", labelpad=2)
ax.set_ylabel("Blub", labelpad=2)
axes["data"] = ax

print_panel_label(letter="A", ax=axes["sketch"], abs_x=0.02, abs_y=0.92)
print_panel_label(letter="B", ax=axes["data"], dx=-0.10, abs_y=0.92)

# # #  PLOT SKETCH  # # #
if "A" in PLOT:
    ax = axes["sketch"]
    ax.imshow(pl.imread(f"./panel/bias_variance.png"))
    ax.set_frame_on(False)


# # #  PLOT DATA  # # #

sg, sr = df["score_green"], df["score_red"]
ax = axes["data"]
for ci, c in enumerate(conditions):
    mar = dict(GLO="o", CLU="d", CDH="s")[c]
    ms =  dict(GLO=3.6, CLU=3.35, CDH=3.35)[c]
    x, y = sg[df['cond'] == c], sr[df['cond'] == c]
    for si, (xi,yi) in enumerate(zip(x,y)):
        col = cmap((1 - 0.05) * si/11. + 0.05)
        ec, mew = dark_color(col), 0.15
        if si % 2 == 1:
            col = "none"
            mew = 0.5
        kwargs = dict(ms=ms, mfc=col, marker=mar, zorder=2, ls="None", label=c if si==8 else None,
                      mew=mew, mec=ec)
        ax.plot(xi, yi, **kwargs)

ax.plot([0,1], [0,1], ":k", lw=0.5, zorder=0)
ax.plot([np.mean(sg)], [np.mean(sg)], "kx", ms=4 )

# Ellipse plotting from: https://stackoverflow.com/a/25022642
from matplotlib.patches import Ellipse
def eigsorted(cov):
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    return vals[order], vecs[:,order]

sg, sr = df["score_green"], df["score_red"]
cov = np.cov(sg, sr)
vals, vecs = eigsorted(cov)
theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))
for nstd in (1.0, 2.0, 3.0):
    w, h = 2 * nstd * np.sqrt(vals)
    kwargs = dict(fill=False, facecolor="none", edgecolor='%.2f' % ((nstd-0.5)/4.), linewidth=0.5, zorder=0)
    ell = Ellipse(xy=(np.mean(sg), np.mean(sr)),
                  width=w, height=h, angle=theta, **kwargs)
    ax.add_artist(ell)

leg = ax.legend(loc='lower right', fontsize=6, borderpad=0.3)

ax.set_xlim(0,1)
ax.set_ylim(0,1)
ax.set_xlabel(r"$f_\mathrm{noise}\,$(green dot)", labelpad=2)
ax.set_ylabel(r"$f_\mathrm{noise}\,$(red dot)", labelpad=2)


# # #  SAVE  # # #
if SAVEFIG:
    fname = "./fig/Fig4.png"
    print("> Save figure to file: %s" % fname)
    fig.savefig(fname)
    fname = "./fig/Fig4.pdf"
    print("> Save figure to file: %s" % fname)
    fig.savefig(fname)
else:
    print("> Figure NOT saved!")
    pl.show()
