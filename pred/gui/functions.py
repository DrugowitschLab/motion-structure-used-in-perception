import numpy as np
exp, log, sqrt = np.exp, np.log, np.sqrt
from os import path
import gzip
from datetime import datetime

try:
    import cPickle as pickle
except:
    import pickle

# # # # # # # # # # # # # # # # # # # # # # # #
# # #          Helper functions           # # #
# # # # # # # # # # # # # # # # # # # # # # # #

# # #  String representation of motion structure matrix  # # #
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


# Create dataset label
def create_dsl(human_readable_dsl=None):
    from datetime import datetime
    dsl = datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")
    if human_readable_dsl is not None:
        dsl += "_" + human_readable_dsl
    return dsl

# Create the output directory
def create_outdir(dsl):
    outdir = "./data/pred/myexp/" + dsl
    import os
    if outdir[-1] != "/":
        outdir += "/"
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    return outdir

# file name
def fname_of_trial(dsl, trial_number):
    fname_only = "trial_%05d.pkl.zip" % trial_number
    outdir = "./data/pred/myexp/" + dsl
    fname_full = path.join(outdir, fname_only)
    return fname_full

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

def calculate_points(globalVars, archive):
    if globalVars.trial_number > 0:   # exclude the init state
        phi_true = archive['Phi'][-1][globalVars.targets]
        phi_human = globalVars.prediction
        points = np.sum(globalVars.f_points(dist_mod2pi(phi_true, phi_human)))
        #print("> POINTS:", points)
    else:
        points = None
    return points


def build_data_dict(globalVars, archive):
    datadict = dict(
        trial_number = globalVars.trial_number,
        trial_seed = globalVars.trial_seed,
        targets = np.array(globalVars.targets),         # These are ordered as clicked
        prediction = np.array(globalVars.prediction),   # These follow that order
        choicetimes = globalVars.choicetimes,           # Fmt: [start of trial, start of decision period (rounded to frame), time of 1st choice, time of 2nd choice]
        t = np.array(archive['t']),
        Phi = np.array(archive['Phi']),
        R = np.array(archive['R']),
        visible = np.array(archive['visible']),
        points = calculate_points(globalVars, archive)
        )
    return datadict

# write data
def write_trial_to_file(fname, datadict):
    print(" > Writing to file: %s" % fname)
    with gzip.open(fname, "wb") as f:
        pickle.dump(datadict, f)
    return True
    # LOAD WITH datadict = pickle.load(gzip.open(fname, "rb"))



# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # #   The matrices for the stochastic differential equation   # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# # # Angular drift  # # #
def build_F_angular_white(tau, L):
    N, M = L.shape
    F = np.zeros((N+M,N+M))
    zeroNN = np.zeros((N,N))
    zeroMN = np.zeros((M,N))
    zeroNM = np.zeros((N,M))
    F[:] = np.vstack([ np.hstack([zeroNN,      L         ]),
                       np.hstack([zeroMN, -np.diag(1/tau)])
                     ])
    return F


# # #  Angular diffusion  # # #
def build_D_angular_white(L):
    N,M = L.shape
    D = np.zeros((N+M,N+M))
    zeroNN = np.zeros((N,N))
    zeroMN = np.zeros((M,N))
    zeroNM = np.zeros((N,M))
    oneM = np.eye(M)
    D[:] = np.vstack([ np.hstack([zeroNN, zeroNM]),
                       np.hstack([zeroMN, oneM   ]),
                     ])
    return D

# # #  Radial components can be added to make the dot orbits non-overlapping  # # #
# # #  Radial drift   # # #
def build_F_radial(tau_r, tau_vr, N):
    F = np.zeros((2*N,2*N))
    zero = np.zeros((N,N))
    oneN = np.eye(N)
    F[:] = np.vstack([ np.hstack([-oneN/tau_r, oneN]),          # We add an active location decay for stable orbits
                       np.hstack([zero       , -oneN/tau_vr])   # Velocities follow an OU process
                     ])
    return F

# # #  Radial diffusion  # # #
def build_D_radial(radial_sigma, N):
    M = N
    D = np.zeros((2*N,N+M))
    zeroNN = np.zeros((N,N))
    zeroMN = np.zeros((M,N))
    zeroNM = np.zeros((N,M))
    oneM = np.eye(M)
    D[:] = np.vstack([ np.hstack([zeroNN, zeroNM]),
                       np.hstack([zeroNN, radial_sigma * oneM   ]),  # This is simply independent diffusion
                     ])
    return D

# # #  Radii don't decay to zero but radial_mean  # # #
def build_bias_radial(radial_mean, tau_r, N):
    b = np.zeros(2*N)
    b[:N] = radial_mean / tau_r         # only locations, not velocities
    return b



# # # # # # # # # # # # # # # # # # # #
# # #   The stimulus generator    # # #
# # # # # # # # # # # # # # # # # # # #

class StructuredMotionStimulus(object):
    def __init__(self, L, tau_vphi, tau_r, tau_vr, radial_sigma, radial_mean, dt, fps, f_dW=None, phi0=None, rngseed=None, DEV=False):
        # # #  Store all parameters  # # #
        self.N, self.M = L.shape            # N dots, M motion components
        self.L = L
        if isinstance(tau_vphi, float):
            tau_vphi = np.array( [tau_vphi]*self.M )
        self.tau_vphi = tau_vphi
        self.tau_r = tau_r
        self.tau_vr = tau_vr
        self.radial_sigma = radial_sigma
        self.radial_mean = radial_mean
        self.dt = dt
        self.t_in_trial = 0.
        self.fps = fps
        self.f_dW = f_dW
        self.phi0 = phi0
        if phi0 is not None:
            assert len(phi0) == self.N, "Error: If not None, len(phi0) must equal num dots."
        self.rng = np.random.RandomState(seed=rngseed)
        self.DEV = DEV
        self.dt_per_frame = 1. / self.fps / self.dt                            # num integration time steps between frames
        assert np.isclose(self.dt_per_frame, int(round(self.dt_per_frame)))    # should be integer valued
        self.dt_per_frame = int(round(self.dt_per_frame))
        # # #  Create angular matrices  # # #
        self.Fphi = build_F_angular_white(tau_vphi, self.L)
        self.Dphi = build_D_angular_white(L)
        # # #  Create radial matrices  # # #
        self.Fr = build_F_radial(tau_r, tau_vr, self.N)
        self.br = build_bias_radial(radial_mean, tau_r, self.N)
        self.Dr = build_D_radial(radial_sigma, self.N)
        # # #  HERE are the dynamics  # # #
        self.sqrtdt = sqrt(self.dt)
        if f_dW is None:
            self.dWphi = lambda: self.rng.normal(loc=0.0, scale=self.sqrtdt, size=self.N + self.M)
        else:
            self.dWphi = f_dW(self.dt).__next__
        self.dphi = lambda x, F, D: F @ x * self.dt + D @ self.dWphi()
        self.dWr = lambda: self.rng.normal(loc=0.0, scale=self.sqrtdt, size=self.N + self.N)
        self.dr = lambda x, F, b, D: (F @ x + b) * self.dt + D @ self.dWr()
        # # # Intialize states  # # #
        self.Phi = np.zeros(self.N+self.M)
        self.R = np.zeros(2*self.N)
        self.reset_states()

    def reset_states(self):
        N = self.N
        M = self.M
        # # #  phi: draw stationary (uniform) locations  # # #
        if self.phi0 is not None:
            self.Phi[:N] = self.phi0
        else:
            self.Phi[:N] = 2 * np.pi * self.rng.rand(N)
        # # #  phi: draw stationary distribution velocities  # # #
        #Sigma = self.tau_vphi * self.L@self.L.T / 2
        if self.f_dW is None:
            Sigma = self.tau_vphi * np.eye(self.M) / 2
            if self.DEV:
                print(Sigma)
            vphi = self.rng.multivariate_normal(mean=np.zeros(M), cov=Sigma)
            self.Phi[N:] = vphi
        else:
            vphi = np.zeros(M)
            self.Phi[N:] = vphi
            nSteps = int(round(5 * min(10, np.max(self.tau_vphi)) / self.dt))
            self.advance(nSteps=nSteps)
        # # #  r: draw approx. stationary location and velocity  # # #
        self.R[N:] = self.rng.normal(loc=0., scale=sqrt(self.tau_vr/2)*self.radial_sigma, size=N)  # velocities follow OU process
        self.R[:N] = self.radial_mean + self.R[N:] * self.tau_r          # This is an approximation: integrated OU under exp decay.
        self.t_in_trial = 0.

    # # #  Euler integration # # #
    def advance(self, nSteps=None):
        if nSteps is None:
            nSteps = self.dt_per_frame
        # # #  Call above dynamics and add them to current value  # # #
        for tn in range(nSteps):
            self.Phi += self.dphi(x=self.Phi, F=self.Fphi, D=self.Dphi)
            self.Phi[:self.N] = self.Phi[:self.N] % (2 * np.pi)             # We are on a circle --> wrap locations to [0, 2*pi]
            self.R += self.dr(x=self.R, F=self.Fr, b=self.br, D=self.Dr)
            self.R[:self.N] = np.maximum(0., self.R[:self.N])               # No negative radii allowed
        self.t_in_trial += self.dt * nSteps
        return self.t_in_trial, self.Phi.copy(), self.R.copy()

    def set_seed(self, seed):
        print("Setting seed %d" % seed)
        self.rng.seed(seed)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # #   EVENT HANDLERS  # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # #


# Mouse click event
def factory_onMouseClick(globalVars):
    def onMouseClick(event):
        #print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
            #('double' if event.dblclick else 'single', event.button, event.x, event.y, event.xdata, event.ydata))
        if event.button != 1:
            print(" > MouseClick: Ignoring mouse click. Not left mouse button.")
            return
        if globalVars.phaseChanger.phase == "after":
            # Init new new trial
            globalVars.start_new_trial()
            return
        if globalVars.phaseChanger.phase != "predict":
            print(" > MouseClick: Ignoring mouse click. Not in prediction phase. ")
            return
        if not globalVars.cursor.isValid(event.ydata):
            print(" > MouseClick: Ignoring mouse click. Not in valid radius.")
            return
        print(" > Valid mouse click detected at phi = %5.1f degrees" % (360 * event.xdata/(2*np.pi),) )
        globalVars.prediction.append(event.xdata)
        globalVars.choicetimes.append(str(datetime.now()))
        if len(globalVars.prediction) < len(globalVars.targets):
            nextColor = globalVars.targetColors[len(globalVars.prediction)]
            globalVars.cursor.set_dotkwargs(color=nextColor)
        else:
            globalVars.cursor.set_visible(False)
            globalVars.phaseChanger.setPredictionMade()
    return onMouseClick

# Key press event
def factory_onkeypress(globalVars):
    def onkeypress(event):
        #print('press', event.key)
        #import sys
        #sys.stdout.flush()
        if event.key == 'escape':
            import pylab as pl
            print("Closing figure.")
            pl.close(globalVars.fig)
            #import sys
            #sys.exit(not globalVars.COMPLETED)
        #elif event.key == 'p':
            #globalVars.PAUSE ^= True
        #elif event.key == 'v' or event.key == 'h':
            #globalVars.HIDETARGETS ^= True
            #globalVars.fade_frame_state = 0
        elif event.key in (' ', 'enter') and globalVars.phaseChanger.phase == "after":
            # Init new new trial
            globalVars.start_new_trial()
    return onkeypress

def connect_event_handlers(globalVars):
    fig = globalVars.fig
    fig.canvas.mpl_connect('key_press_event', factory_onkeypress(globalVars))
    cid = fig.canvas.mpl_connect('button_press_event', factory_onMouseClick(globalVars))
    fig.canvas.mpl_connect('motion_notify_event', globalVars.cursor.mouse_move)



class PhaseChanger(object):
    def __init__(self, experiment_dict):
        self.experiment_dict = experiment_dict
        self.order = experiment_dict["order"]
        self.frames = np.array([experiment_dict[phase]["numFrames"] for phase in self.order if "numFrames" in experiment_dict[phase]])
        self.cumFrames = self.frames.cumsum()
        self.predictionMade = False
        self.phase = None

    def getPhase(self, frame):
        predict_idx = self.order.index("predict")
        if frame >= self.cumFrames[predict_idx-1] and not self.predictionMade:
            self.phase = "predict"
            return self.phase
        if frame > self.cumFrames[-1]:
            self.phase = "after"
            return self.phase
        idx = (frame < self.cumFrames).argmax()
        self.phase = self.order[idx]
        return self.phase

    def newTrial(self):
        self.phase = self.order[0]
        self.predictionMade = False
        self.isMouseCursorReset = False

    def setPredictionMade(self):
        self.predictionMade = True


# Init mouse cursor
class Cursor(object):
    def __init__(self, ax, valid_radius=[0.95,1.05]):
        self.ax = ax
        self.dot_size = 12
        self.dot_color = "r"
        self.dot = ax.plot(0, 0, 'o', c=self.dot_color, ms=self.dot_size, mew=0.)[0]
        self.isValid = lambda r: valid_radius[0] <= r <= valid_radius[1]
        self.isVisible = False
        self.dot.set_visible(self.isVisible)
        # text location in axes coords
        #self.txt = ax.text(0.7, 0.9, '', transform=ax.transAxes)

    def mouse_move(self, event):
        if not event.inaxes:
            return
        phi, r = event.xdata, event.ydata
        # update the line positions
        self.dot.set_ydata(r)
        self.dot.set_xdata(phi)
        self.dot.set_markersize(self.dot_size)
        self.dot.set_markerfacecolor(self.dot_color)
        alpha = 0.3 + 0.7 * self.isValid(r)
        self.dot.set_alpha(alpha)
        self.dot.set_visible(self.isVisible)
        #self.txt.set_text('phi=%5.1f deg, r=%1.2f' % (360 * phi/2/np.pi, r))

    def set_visible(self, b):
        self.isVisible = b
        # Set the colored dot's visibility
        self.dot.set_visible(self.isVisible)
        # Set the mouse cursor's visibility
        import pylab as pl
        import matplotlib as mpl
        if mpl.get_backend() in ("Qt4Agg", "TkAgg"):
            pass
        elif mpl.get_backend() == "Qt5Agg":
            from PyQt5.QtCore import Qt
            from PyQt5.QtWidgets import QApplication
            from PyQt5.QtGui import QCursor
            cursorShape = {True: Qt.ArrowCursor, False: Qt.BlankCursor}[self.isVisible]
            QApplication.setOverrideCursor(QCursor(cursorShape))

    def reset_mouse_position(self):
        import matplotlib as mpl
        if mpl.get_backend() in ("Qt4Agg", "TkAgg"):
            pass
        elif mpl.get_backend() == "Qt5Agg":
            from PyQt5.QtGui import QCursor
            import pylab as pl
            screenCoords = pl.gca().transAxes.transform((0.5,0.5)).astype(int)
            screenCoords += pl.get_current_fig_manager().window.geometry().getRect()[:2]
            QCursor.setPos(*tuple(screenCoords))

    def set_dotkwargs(self, **kwargs):
        if "size" in kwargs:
            self.dot_size = kwargs["size"]
            self.dot.set_markersize(self.dot_size)
        if "color" in kwargs:
            self.dot_color = kwargs["color"]
            self.dot.set_markerfacecolor(self.dot_color)
















































