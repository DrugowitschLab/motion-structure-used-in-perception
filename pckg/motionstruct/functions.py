
# # # COLOR FORMATTER from https://stackoverflow.com/a/384125 # # #
BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE = range(8)
#The background is set with 40 plus the number of the color, and the foreground with 30
#These are the sequences need to get colored ouput
RESET_SEQ = "\033[0m"
COLOR_SEQ = "\033[1;%dm"
BOLD_SEQ = "\033[1m"

def formatter_message(message, use_color = True):
    if use_color:
        message = message.replace("$RESET", RESET_SEQ).replace("$BOLD", BOLD_SEQ)
    else:
        message = message.replace("$RESET", "").replace("$BOLD", "")
    return message

COLORS = {
    'WARNING': RED,
    'INFO': WHITE,
    'DEBUG': BLUE,
    'CRITICAL': RED,
    'ERROR': RED
}

import logging
class ColoredFormatter(logging.Formatter):
    def __init__(self, msg, use_color = True):
        logging.Formatter.__init__(self, msg)
        self.use_color = use_color

    def format(self, record):
        from copy import copy
        record = copy(record)
        levelname = record.levelname
        if self.use_color and levelname in COLORS:
            levelname_color = COLOR_SEQ % (30 + COLORS[levelname]) + levelname[0] + RESET_SEQ
            record.levelname = levelname_color
        return logging.Formatter.format(self, record)


def init_logging(cfg, outdir):
    import logging
    logger = logging.Logger("SimLogger")
    if "loglevel" in cfg["global"]:
        logger.setLevel(cfg["global"]["loglevel"])
    else:
        logger.setLevel(logging.DEBUG)
    consoleHandler = logging.StreamHandler()
    #logFormatter = logging.Formatter("[%(levelname)-8.8s]  %(message)s")
    logFormatter = ColoredFormatter(formatter_message("[$BOLD%(levelname)-12.12s$RESET]  %(message)s"))
    consoleHandler.setFormatter(logFormatter)
    logger.addHandler(consoleHandler)

    if not cfg["global"]["DRYRUN"]:
        logging_fname = outdir+"logfile.log"
        fileHandler = logging.FileHandler(logging_fname)
        logFormatter = logging.Formatter("%(asctime)s [%(levelname)-8.8s]  %(message)s")
        fileHandler.setFormatter(logFormatter)
        logger.addHandler(fileHandler)
    else:
        logger.warning("Dryrun. No data will be saved!\n\n \
                        \t\t * * * * * * * * * * * * * * * * * * *\n \
                        \t\t * * * * * *  D R Y R U N  * * * * * *\n \
                        \t\t * * * * * * * * * * * * * * * * * * *\n")
    return logger


import numpy as np
def asciiL(L, indent=0):
    indent = " "*indent
    theta = np.array([0.05, 0.500, 0.999, 1.001])
    chars = {
        'zero' : " ",
        3 : "█",
        2 : "▓",
        1 : "▒",
        0 : "░"
        }
    vmax, vmin = L.max(), L.min()
    char = lambda val: chars['zero'] if val == 0. else chars[( (val - vmin)/(vmax-vmin) < theta ).argmax()]
    s = indent + "┌" + "─"*(2*L.shape[1]) + "┐\n"
    for line in L:
        s += indent + "│" + "".join( [ char(v)*2 for v in line] ) + "│\n"
    s += indent + "└" + "─"*(2*L.shape[1]) + "┘"
    return s


import collections
def recursive_dict_update(d, u, s="cfg.", msg=[]):
    for k, v in u.items():
        if isinstance(v, collections.Mapping):
            d[k], msg = recursive_dict_update(d.get(k, {}), v, s=s + str(k) + ".", msg=msg)
        else:
            d[k] = v
            s += str(k)
            msg.append(s)
    return d, msg


def dist_mod2pi(x, y=None):
        from numpy import pi
        if y is not None:
            d = x - y
        else:
            d = x.copy()
        d += pi
        d %= (2*pi)
        d -= pi
        return d


# # # # # # # # # # # # # # # # # # # # # #
# # #  S I M P L E   P L O T T I N G  # # #
# # # # # # # # # # # # # # # # # # # # # #

def plot_kal(kal, rep=None):
    N = wld.N
    fig = pl.figure(figsize=(12,5))
    t = kal.archive["t"]
    mu = np.array(kal.archive["mu"])
    yerr = np.array([ np.sqrt(sig.diagonal()) for sig in kal.archive["Sig"] ])
    for i, (mu_i, yerr_i) in enumerate(zip(mu.T, yerr.T)):
        if i == 0:
            pl.subplot(121)
            pl.xlabel("Time [s]")
            pl.ylabel("Phi")
            perf = performance_func(kal.gam.perm)
            title = "Angular location (perf: %d, trial: %d)" % (perf,rep+1) if (rep is not None) else "Angular location (perf: %d)" % perf
            pl.title(title)
            pl.hlines((0, 2*np.pi), 0, T)
            pl.plot(wld.get_times(), wld.S[:,:N], lw=0.75)
            pl.gca().set_prop_cycle(None)
            pl.plot(obs.get_times(), obs.X[:,:N], '.', ms=2)
            pl.gca().set_prop_cycle(None)
            pl.ylim(-0.25,2*np.pi+0.25)
        elif i == N:
            pl.subplot(122)
            pl.xlabel("Time [s]")
            pl.ylabel("v_Phi")
            pl.title("Angular velocity")
            pl.gca().set_prop_cycle(None)
            pl.plot(wld.get_times(), wld.S[:,N:], lw=0.75)
            pl.gca().set_prop_cycle(None)
            pl.ylim(-2,2)
            #pl.plot(obs.get_times(), X[:,N:], '.', ms=2)
            #pl.gca().set_prop_cycle(None)
        pl.errorbar(x=t, y=mu_i, yerr=yerr_i, marker='x', ms=4, lw=0., elinewidth=0.5, alpha=0.8)
        true_err = wld.S[::10,i] - mu_i[1:]
        true_err += np.pi
        true_err %= 2 * np.pi
        true_err -= np.pi
        true_err = np.std(true_err)
        #print("True Std (i=%d):" % i, true_err)
        #print("Est. Std (i=%d):" % i, (yerr_i).mean(), "\n")
    pl.subplots_adjust(0.05, 0.1, 0.98, 0.93)
    pl.show()
    return fig

