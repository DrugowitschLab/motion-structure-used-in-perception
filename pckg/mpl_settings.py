import matplotlib as mpl

name = "Matplotlib settings for 'Bill et al. (2019)'"

cm2inch = lambda x: 0.393700787 * x

fig_width = dict(
    onecol = cm2inch(8.7),
    oneandhalfcol = cm2inch(11.4),
    twocol = cm2inch(17.8)
    )

panellabel_fontkwargs = dict(fontweight="normal", fontsize=10, ha="center", va="baseline")

config = {
    'axes' : dict(labelsize=8, titlesize=8, linewidth=0.5),
    'figure' : dict(dpi=114., figsize=[fig_width['onecol'], 1.2941*fig_width['onecol']], facecolor='white'),
    'figure.subplot' : dict(left=0.14, bottom=0.08, right=0.98, top=0.98, wspace=0.4, hspace=0.4),
    'font' : {'family' : 'sans-serif', 'size' : 8, 'weight' : 'normal',
              'sans-serif' : ['Arial', 'LiberationSans-Regular', 'FreeSans'],
              'serif' : ['Times New Roman', 'DejaVu Serif', 'serif']},
    'image' : dict(cmap='RdBu_r' , interpolation='nearest'),
    'legend' : dict(fontsize=6, borderaxespad=0.5, borderpad=0.5, labelspacing=0.3),
    'lines' : dict(linewidth=0.5),
    'xtick' : dict(labelsize=6),
    'xtick.major' : dict(size=1., pad=2, width=0.5),
    'ytick' : dict(labelsize=6),
    'ytick.major' : dict(size=1., pad=2, width=0.5),
    'errorbar' : dict(capsize=1.5),      # combine with errorbar(..., capthick=0.5) <-- or whatever line width
    'hatch' : dict(linewidth=0.5),
    'savefig' : dict(dpi=600)
    }

print ("\n\t * * * Importing '%s' * * *\n" % name)

for key,val in config.items():
    s = ""
    for k,v in val.items():
        s += k + "=%s, " % str(v)
    print ("  > Set '%s' to %s" % (key, s[:-2]) )
    mpl.rc(key, **val)

print ('\n\n')
