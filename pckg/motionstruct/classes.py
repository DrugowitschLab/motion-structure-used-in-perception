# # # # # #
# # #   This file contains more classes than acutally being used in the experiments and analysis.
# # # # # # # # # # 

import numpy as np
from numpy import ma
from scipy.stats import multivariate_normal as multivariate_normal_pdf

# Replace slow scipy permutation by faster own one
#from sympy.combinatorics import Permutation
from motionstruct.classPermutation import Permutation
Permutation.secure = False               # Perform type checks on cost of execution speed
Permutation.print_cyclic = True          # If True, uses the slower sympy implementation for printing

pi = np.pi
numerical_infty = 1.e10                   # covariance values to mimic infinity (for velocity observations)

# # # Angular matrices # # #
# Drift normal
def build_F_angular(tau_vphi, L):
    N = L.shape[0]
    F = np.zeros((2*N,2*N))
    zero = np.zeros((N,N))
    oneN = np.eye(N)
    F[:] = np.vstack([ np.hstack([zero, oneN]),
                    np.hstack([zero, -oneN/tau_vphi])
                    ])
    return F

# Drift whitened
def build_F_angular_white(tau_vphi, L):
    N, M = L.shape
    F = np.zeros((N+M,N+M))
    zeroNN = np.zeros((N,N))
    zeroMN = np.zeros((M,N))
    zeroNM = np.zeros((N,M))
    F[:] = np.vstack([ np.hstack([zeroNN,      L         ]),
                       np.hstack([zeroMN, -np.diag(1/tau_vphi)])
                     ])
    return F

# Diffusion normal
def build_D_angular(L):
    N, M = L.shape
    D = np.zeros((2*N,N+M))
    zeroNN = np.zeros((N,N))
    zeroMN = np.zeros((M,N))
    zeroNM = np.zeros((N,M))
    oneM = np.eye(M)
    D[:] = np.vstack([ np.hstack([zeroNN, zeroNM]),
                    np.hstack([zeroNN, L   ]),
                    ])
    return D

# Diffusion whitened
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


# # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # #   W O R L D   S I M U L A T O R   # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # #

class PhiWorld(object):
    def __init__(self, tau_vphi, dt, seed, L=None, B=None, lam=None, whitespace=False, init_equidistant=False):
        if whitespace is True:
            if isinstance(tau_vphi, float):
                tau_vphi = tau_vphi * np.ones(L.shape[1])
            assert isinstance(tau_vphi, np.ndarray), " > ERROR: For whitespace, tau_vphi must be ndarray or float."
        elif whitespace is False:
            assert isinstance(tau_vphi, float), " > ERROR: Without whitespace, tau_vphi must be float."
        else:
            raise Exception(" > ERROR: whitened must be True or False")
        if np.any(tau_vphi / dt < 5.):
            print(" > WARNING: Coarse world simulation time step in units of tau_vphi! Object trajectories may be inaccurate!")
        self.whitespace = whitespace
        assert (L is not None) and (B is None) and (lam is None) or (L is None) and (B is not None) and (lam is not None),               "Initialize either via 'L' or ('B', 'lam')."
        if L is None:
            self.L = B @ np.diag(lam)
            self.B = B
            self.lam = lam
        else:
            self.L = L
        self.N, self.M = self.L.shape
        self.numVelo = self.M if whitespace else self.N
        self.tau_vphi = tau_vphi
        self.dt = dt
        self.rng = np.random.RandomState(seed)
        self.init_equidistant = init_equidistant
        self.S = None
        # for advance
        if whitespace is True:
            F = build_F_angular_white(self.tau_vphi, self.L)
            D = build_D_angular_white(self.L)
        elif whitespace is False:
            F = build_F_angular(self.tau_vphi, self.L)
            D = build_D_angular(self.L)
        sqrtdt = np.sqrt(self.dt)
        dW = lambda: self.rng.normal(0., 1., self.N + self.M)
        self.ds = lambda s: F @ s * self.dt + sqrtdt * D @ dW()

    def draw_initial_state(self):
        if self.init_equidistant:
            # Equidistant locations
            # We want to stay as compatible as possible with uniform-sims
            _ = self.rng.uniform(0., 2*pi, self.N)
            phi = np.linspace(0, 2 * np.pi, self.N+1)[:-1]
            state = self.rng.get_state()
            self.rng.shuffle(phi)
            self.rng.set_state(state)
        else:
            # Uniform locations: The default case
            phi = self.rng.uniform(0., 2*pi, self.N)
        # The following construction ensures that identical trajectories
        # are generated if tau_vphi=float (indep. of whitespace=True/False)
        if self.whitespace is True:
            L_gen = np.sqrt(1/2.) * np.diag(np.sqrt(self.tau_vphi))
        elif self.whitespace is False:
            L_gen = np.sqrt(self.tau_vphi/2.) * self.L
        vphi = L_gen @ self.rng.normal(size=self.M)
        #vphi = self.rng.multivariate_normal(np.zeros(self.numVelo), Sig)
        s = np.concatenate( (phi, vphi) )
        return s

    def dot_dist(self, s):  # only a helper for other classes
        assert s.ndim == 1, "Only one time state."
        s = s[:self.N]      # loc only
        d = s[:,None] - s   # pairs
        d += pi             # distances
        d %= (2*pi)
        d -= pi
        return d

    def draw_trajectory(self, T):
        Tn = int(round(T / self.dt))
        S = np.zeros((Tn+1, self.N+self.numVelo))
        S[0] = self.draw_initial_state()
        for tn in range(Tn):
            s = S[tn]
            S[tn+1] = s + self.ds(s)
            S[tn+1,:self.N] = S[tn+1,:self.N] % (2 * pi)
        self.S = S
        return S

    def index_of_t(self, t):
        tn = np.round(t/self.dt).astype(int)
        return tn

    def get_times(self):
        assert self.S is not None, " > ERROR: Draw trajectory first!"
        Tn = self.S.shape[0]
        return self.dt * np.arange(Tn)


class PhiWorldDataHose(object):
    def __init__(self):
        self.N = None
        self.S = None
        self.get_times = lambda: NotImplemented


# # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # #   O B S E R V A T I O N   G E N E R A T O R   # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # #

class PhiObservationGenerator(object):
    def __init__(self, gen_x, dt, seed, final_precise, gen_x_precise=None):
        self.rng =  np.random.RandomState(seed)
        self.gen_x = lambda s: gen_x(rng=self.rng, s=s)
        self.dt = dt
        self.final_precise = final_precise
        self.gen_x_precise = lambda s: gen_x_precise(rng=self.rng, s=s)
        self.T = None
        self.X = None

    def run_sim_and_generate_observations(self, T, worldsim):
        worldsim.draw_trajectory(T)
        X = self.generate_observations(T, worldsim)
        return X

    def get_times(self):
        assert self.T is not None, " > ERROR: Generate observations first!"
        eps = 1.e-10
        return np.arange(0., self.T+eps, self.dt)

    def generate_observations(self, T, worldsim):
        self.T = T
        t = self.get_times()
        tn = worldsim.index_of_t(t)
        S = worldsim.S[tn]
        X = np.array( [self.gen_x(s) for s in S] )
        if self.final_precise:
            X[-1] = self.gen_x_precise(S[-1])
        self.X = X
        return X


class PhiObservationGeneratorMOTFromFiles(object):
    def __init__(self, datadir, seed, trials, reps_per_trial, gen_x):
        self.datadir = datadir
        self.rng =  np.random.RandomState(seed)
        self.trials = trials
        self.reps_per_trial = reps_per_trial
        self.gen_x = lambda s: gen_x(rng=self.rng, s=s)
        self.load_data(datadir)
        self.current_trial = None
        self.current_rep = None

    def index_of_t(self, t, dt):
        tn = np.round(t/dt).astype(int)
        return tn

    def load_data(self, datadir):
        print(" > Searching for trial data.")
        from os import path
        fname = path.join(self.datadir, "simdata.npz")
        X = np.load(fname)
        self.trialdata = dict(
            X = X["wld_S"],             # We must load the world state (in Phi)
            t = X["obs_t"]
            )
        nTrialsFound = self.trialdata["t"].shape[0]
        tmax = self.trialdata["t"][0,-1]
        dt = ( self.trialdata["t"][0,1] - self.trialdata["t"][0,0] )
        print("   > Found %d trials of duration %.2fs at %.1fHz." % (nTrialsFound, tmax, 1/dt))
        print("   > Selecting for presentation:", self.trials)
        times = self.index_of_t(self.trialdata["t"][0], dt=dt/10)
        self.trialdata["Phi"] =  self.trialdata["X"][self.trials][:,times]
        self.trialdata["t"] =  self.trialdata["t"][self.trials]

    def get_L_groundtruth(self):
        from sys import path
        import os
        dir = os.path.abspath(self.datadir)
        path.insert(0, dir)     # make the configuration available, temporarilly
        from config import cfg as cfg_gt
        path.pop(0)             # restore the path again
        return cfg_gt["world"]["L"]

    def get_times(self):
        assert self.current_trial is not None
        return self.trialdata["t"][self.current_trial]

    def run_sim_and_generate_observations(self, T, wld, trial, rep):
        self.current_trial = trial
        self.current_rep = rep
        wld.S = self.trialdata["Phi"][trial]
        X = np.array( [self.gen_x(s) for s in wld.S] )
        self.X = X
        assert T == self.trialdata["t"][self.current_trial][-1]
        return X

class PhiObservationGeneratorLocPredict(object):
    def __init__(self, T_visible, targets, seed, dt, sigma_obs_phi):
        self.T_visible = T_visible
        self.targets = np.array(targets)
        self.rng =  np.random.RandomState(seed)
        self.dt = dt
        self.sigma_obs_phi = sigma_obs_phi
        self.T = None
        self.X = None

    def run_sim_and_generate_observations(self, T, worldsim):
        worldsim.draw_trajectory(T)
        X = self.generate_observations(T, worldsim)
        return X

    def get_times(self):
        assert self.T is not None, " > ERROR: Generate observations first!"
        eps = 1.e-10
        return np.arange(0., self.T+eps, self.dt)

    def generate_observations(self, T, worldsim):
        self.T = T
        t = self.get_times()
        tn = worldsim.index_of_t(t)
        S = worldsim.S[tn]
        N = worldsim.N
        gen_x = lambda s: np.concatenate( ( self.rng.normal(s[:N], self.sigma_obs_phi) % (2*np.pi), np.zeros(N)) )
        X = np.array( [gen_x(s) for s in S] )
        # masking velocities
        mask = np.array([[False]*N + [True]*N]*len(tn), dtype=bool)
        # mask hidden dots
        mask[t > self.T_visible, self.targets[:,None]] = True
        X = ma.masked_array(X, mask)
        self.X = X
        return X


class PhiObservationGeneratorLocPredictFromFiles(object):
    # We do this as observation generator (not a world) due to the visibility information.
    def __init__(self, datadir, seed, sigma_obs_phi):
        from os import path
        self.datadir = datadir
        self.rng = np.random.RandomState(seed)
        self.sigma_obs_phi = sigma_obs_phi
        self.reps = None
        self.next_trial = None                            # python indexing
        self.N = 0                                        # Will be inferred from visible data
        self.datafname_pattern = "trial_*.pkl.zip"        # Files (name and trial_number) use matlab indexing
        self.load_data(datadir)

    def load_data(self, datadir):
        print(" > Searching for trial data.")
        from os import path
        import glob
        filelist = glob.glob(path.join(datadir, self.datafname_pattern))
        filelist = np.sort(filelist)
        self.reps = len(filelist)
        self.trialdata = dict()
        for n, fname in enumerate(filelist):
            data = self.load_data_from_file(fname)
            assert n == data['trial'], "ERROR: Trial numbers do not match (%d vs %d)" % (n, data['trial']) # python ind.
            self.trialdata[n] = data
            maxNtrial = np.max([np.max(vt) for vt in data['visible']]) + 1
            self.N = max(self.N, maxNtrial)
        print(" > Number of dots (from data): %d" % self.N)
        self.next_trial = 0

    def load_data_from_file(self, fname):
        import pickle, gzip
        print("   > Loading data from file: %s" % fname)
        datadict = pickle.load(gzip.open(fname, "rb"))
        t = datadict['t']
        phi = datadict['Phi']                          # the "real" targets
        visible = datadict['visible']                  # Will be used for creating the mask
        targets = np.sort(datadict['targets'])         # order of target selection
        trial = datadict['trial_number'] - 1           # From matlab to python
        data = dict(t=t, phi=phi, visible=visible, targets=targets, trial=trial)
        return data

    def run_sim_and_generate_observations(self, T, worldsim):
        # T is not used since get_times() works from data.
        assert self.next_trial is not None, "Error: Load data first!"
        self.this_trial = self.next_trial
        data = self.trialdata[self.this_trial]
        # push data to wld
        assert isinstance(worldsim, PhiWorldDataHose), "Error: world must be PhiWorldDataHose-dummy."
        worldsim.get_times = lambda: self.get_times()
        worldsim.N = self.N
        worldsim.S = data['phi']
        # generate (noisy & masked) observations
        X = self.generate_observations(data)
        self.next_trial += 1
        return X

    def get_times(self):
        return self.trialdata[self.this_trial]['t']

    def generate_observations(self, data):
        t = data['t']
        S = data['phi']
        N = self.N
        if self.sigma_obs_phi > 0:
            gen_x = lambda s: np.concatenate( ( self.rng.normal(s[:N], self.sigma_obs_phi) % (2*np.pi), np.zeros(N)) )
        elif self.sigma_obs_phi == 0:
            gen_x = lambda s: np.concatenate( ( s[:N] % (2*np.pi), np.zeros(N) ) )
        else:
            raise Exception("Observation noise 'sigma_obs_phi' must be >= 0!")
        X = np.array( [gen_x(s) for s in S] )
        # masking velocities (and a priori also locations)
        mask = np.array([[True]*N + [True]*N]*len(t), dtype=bool)
        # unmask visible dots
        for i,vt in enumerate(data['visible']):
            mask[i, vt] = False
        X = ma.masked_array(X, mask)
        self.X = X
        return X


class ScreenXYObservationGenerator(object):
    def __init__(self, dt, screen_resolution, screen_aspect, relative_radius, final_min_distance=None):
        self.dt = dt
        self.final_min_distance = final_min_distance
        sx = screen_resolution[0]
        sy = screen_resolution[1]
        correction = screen_aspect / ( sx / sy)
        mean_x = sx / 2
        mean_y = sy / 2
        if screen_aspect >= 1:   # wider than high
            scale_x = relative_radius * sy / 2 / correction
            scale_y = relative_radius * sy / 2
        else: # higher than wider
            scale_x = relative_radius * sx / 2
            scale_y = relative_radius * sx / 2 * correction
        self.scale_to_sceen = lambda xnat, ynat: ( mean_x + scale_x * xnat, mean_y - scale_y * ynat )
        self.project = lambda phi: self.scale_to_sceen( -np.sin(phi), np.cos(phi) )

    def run_sim_and_generate_observations(self, T, worldsim):
        worldsim.draw_trajectory(T)
        while not self.is_valid_trial(worldsim):
            print(" > INFO: Invalid trial generated (too close final degrees). Redrawing.")
            worldsim.draw_trajectory(T)
        X = self.generate_observations(T, worldsim)
        return X

    def is_valid_trial(self, worldsim):
        if self.final_min_distance is None:
            return True
        d = worldsim.dot_dist(worldsim.S[-1])
        d = ma.masked_array(d, d<=0)
        valid = (d > self.final_min_distance).all()
        return valid

    def get_times(self):
        assert self.T is not None, " > ERROR: Generate observations first!"
        eps = 1.e-10
        return np.arange(0., self.T+eps, self.dt)

    def generate_observations(self, T, worldsim):
        self.T = T
        t = self.get_times()
        tn = worldsim.index_of_t(t)
        N = worldsim.N
        S = worldsim.S[tn]
        X = np.array( [self.project(s[:N]) for s in S] )
        X = np.swapaxes(X, 1, 2)        # from (t, x/y, dot) to (t, dot, x/y)
        self.X = X
        return X


# # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # #   K A L M A N   F I L T E R   # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # #

class PhiKalmanFilterPermutation(object):
    def __init__(self, L, tau_vphi, sigma_obs_phi, init_certain=False, gam=None, whitespace=False):
        if whitespace is True:
            if isinstance(tau_vphi, float):
                tau_vphi = tau_vphi * np.ones(L.shape[1])
            assert isinstance(tau_vphi, np.ndarray), " > ERROR: For whitespace, tau_vphi must be ndarray or float."
        elif whitespace is False:
            assert isinstance(tau_vphi, float), " > ERROR: Without whitespace, tau_vphi must be float."
        else:
            raise Exception(" > ERROR: whitened must be True or False")
        self.whitespace = whitespace
        self.N, self.M = L.shape
        self.numVelo = self.M if whitespace else self.N
        self.L = L
        self.tau_vphi = tau_vphi
        self.sigma_obs_phi = sigma_obs_phi
        self.init_certain = init_certain
        if gam is None:
            gam = Permutation(size=self.N)      # Identity
        self.set_permutation(gam)
        # current state estimates
        self.mu = None
        self.Sig = None
        self.t_last = None
        self._is_propagated = False                             # measures where we are in propagation/integration
        # Helper matrices for Kalman updates
        if whitespace is True:
            self.F_base = build_F_angular_white(self.tau_vphi, self.L)    # without inter-observation interval
            D_base = build_D_angular_white(self.L)
        elif whitespace is False:
            self.F_base = build_F_angular(self.tau_vphi, self.L)    # without inter-observation interval
            D_base = build_D_angular(self.L)
        self.Q_base = D_base @ D_base.T                         # without inter-observation interval
        self.H = np.zeros((2*self.N, self.N+self.numVelo))
        self.H[:self.N,:self.N] = np.eye(self.N)                # observation matrix
        if whitespace is True:
            self.H[self.N:,self.N:] = L
        elif whitespace is False:
            self.H[self.N:,self.N:] = np.eye(self.N)
        self.R = np.zeros((2*self.N, 2*self.N))
        self.R[:self.N,:self.N] = self.sigma_obs_phi**2 * np.eye(self.N)    # assumed observation noise in phi (Has to be extended if occlusions could occur!)
        self.R[self.N:,self.N:] = numerical_infty * np.eye(self.N)          # assumed observation noise in vphi

    def set_permutation(self, gam):
        assert isinstance(gam, Permutation), " > ERROR: Permutations must be type <Permutation> from sympy.combinatorics!"
        self.gam = gam
        self.gam_full = self.extend_permutation_to_velocities(gam)

    def extend_permutation_to_velocities(self, gam):
        N = self.N
        return Permutation(list(gam(np.arange(self.N))) + list(gam(np.arange(self.N, 2*self.N))))

    def init_filter(self, s):
        """Init with true state"""
        latentDim = self.N+self.numVelo     # 2N if not whitened; N+M if whitened
        assert len(s) == latentDim, "ERROR: Init state s does not match latentDim!"
        self.mu = s
        self.Sig = np.zeros((latentDim, latentDim))
        self.Sig[:self.N,:self.N] = self.sigma_obs_phi**2 * np.eye(self.N)
        if self.whitespace is True:
            self.Sig[self.N:,self.N:] = np.diag(self.tau_vphi) / 2.
        elif self.whitespace is False:
            self.Sig[self.N:,self.N:] = (self.tau_vphi / 2.) * self.L @ self.L.T
        if self.init_certain:
            self.Sig /= 1000.
        # apply permutation
        #self.mu = self.Gam @ self.mu
        #self.Sig = self.Gam @ self.Sig @ self.Gam.T
        # time is zero
        self.t_last = 0.
        self._is_propagated = False
        self.archive = dict(t=[self.t_last], mu=[self.mu], Sig=[self.Sig], gam=[self.gam.perm.copy()])

    def propagate_and_integrate_observation(self, x, t):
        """For the non-particle filter version, we can go straight forward."""
        self.propagate_to_time(t)
        self.integrate_observation(x)
        return self.mu, self.Sig

    def propagate_to_time(self, t):
        assert self.t_last is not None, " > ERROR: Initialize filter state first!"
        assert self._is_propagated is False, " > ERROR: Trying to propagate already propagated filter." # This could (in principle) be allowed.
        # observation time update
        dt = t - self.t_last
        self.t_last = t
        assert dt >= 0, " > ERROR: Negative interval since last observation!"
        # Let's hack in the Kalman update equations...
        N = self.N
        F = np.eye(N+self.numVelo) + dt * self.F_base
        Q = dt * self.Q_base
        H = self.H
        R = self.R
        # Prior moments
        mu_hat = F @ self.mu
        mu_hat[:N] %= (2*pi)

        Sig_hat = F @ self.Sig @ F.T + Q
        # Kalman gain
        S_res = R + H @ Sig_hat @ H.T
        K = Sig_hat @ H.T @ np.linalg.inv(S_res)
        self.propagated_variables = dict(   # All we need for updates:
            t = t,                          # time
            mu_hat = mu_hat,                # mean
            Sig_hat = Sig_hat,              # cov
            K = K                           # Kalman gain
            )
        self._is_propagated = True

    def integrate_observation(self, x):
        assert self._is_propagated is True, " > ERROR: Propagate filter before integrating new observation!"
        assert self.propagated_variables["t"] == self.t_last, " > ERROR: Inconsistent internal time information."
        # retrieve pre-computed values and required variables
        mu_hat = self.propagated_variables["mu_hat"]
        Sig_hat = self.propagated_variables["Sig_hat"]
        K = self.propagated_variables["K"]
        N = self.N
        R = self.R
        H = self.H
        # calc residual
        mu_res = self.calculate_residual_mean(x)
        # integrate observation: means
        self.mu = mu_hat + K @ mu_res
        self.mu[:N] %= 2*pi
        # integrate observation: covariance
        M = ( np.eye(N+self.numVelo) - K @ H )
        self.Sig = M @ Sig_hat @ M.T + K @ R @ K.T          # <<-- WARNING: IF data points could have varying precision, R must be permutated/adapted.
        # switch state
        self._is_propagated = False
        # store results
        self.archive["t"].append(self.propagated_variables["t"])
        self.archive["mu"].append(self.mu)
        self.archive["Sig"].append(self.Sig)
        self.archive["gam"].append(self.gam.perm.copy())

    def calculate_residual_mean(self, x, gam=None):
        """mu_hat under current propagation assuming permutation gam (defaults to self.gam)."""
        assert self.propagated_variables["t"] == self.t_last, " > ERROR: Inconsistent internal time information."
        mu_hat = self.propagated_variables["mu_hat"]
        if gam is None:
            gam_full = self.gam_full
        else:
            gam_full = self.extend_permutation_to_velocities(gam)
        N = self.N
        H = self.H
        mu_res = (gam_full(x) - H @ mu_hat)             # <<--- THIS IS WHERE THE PERMUTATION ENTERS!
        mu_res[:N] += pi
        mu_res[:N] %= (2*pi)
        mu_res[:N] -= pi
        return mu_res

    def observation_likelihood(self, x, perm=None, differential=True, location_only=True):
        """Calculate data likelihood under a hypothetical data assignment.
             differential: # whether perm is relative to current internal gam, or absolute.
               differential ==  True: used permutation is (self.gam * perm)
               differential == False: used permutation is perm
             If perm==None, either case will use the current internal gam.
             If location_only: Do not evaluate Gaussian on velocities (e.g. due to inf-variance)
        """
        if perm is None:
            gam = None
        else:
            if differential is True:
                gam = self.gam * perm
            elif differential is False:
                gam = perm
            else:
                raise Exception(" > ERROR: 'differential' must be True or False.")
        # Calc residual under permutation
        mu_res = self.calculate_residual_mean(x, gam=gam)
        # calculate pdf (This could be done once, and then be reused for efficiency)
        N = self.N
        H = self.H
        Sig_tot = H @ self.propagated_variables["Sig_hat"] @ H.T + self.R              # IF data points could have varying precision, R must be adapted.
        if location_only is True:
            pdf = multivariate_normal_pdf(mean=np.zeros(N), cov=Sig_tot[:N,:N]).pdf
            mu_res = mu_res[:N]
        else:
            pdf = multivariate_normal_pdf(mean=np.zeros(2*N), cov=Sig_tot).pdf
        # Here comes the value
        px = pdf(mu_res)
        return px

    def clone(self, perm=None, differential=True):
        """Clone the Kalman filter (incl history), but with a changed permutation.
             See 'observation_likelihood' for definition of 'perm' and 'differential'.
        """
        if perm is None:
            gam = self.gam
        else:
            if differential is True:
                gam = self.gam * perm
            elif differential is False:
                gam = perm
            else:
                raise Exception(" > ERROR: 'differential' must be True or False.")
        from copy import deepcopy
        kwargs = dict(L = self.L,
                  tau_vphi = self.tau_vphi,
                  sigma_obs_phi = self.sigma_obs_phi,
                  gam = gam,
                  whitespace=self.whitespace
                )
        c = type(self)(**kwargs)
        # copy states
        c.mu = np.copy(self.mu)
        c.Sig = np.copy(self.Sig)
        c.t_last = self.t_last
        c._is_propagated = self._is_propagated
        c.archive = deepcopy(self.archive)
        c.propagated_variables = deepcopy(self.propagated_variables)
        return c


# # #   LOCATION BASED KALMAN FILTER   # # #

class PhiKalmanFilterPermutationLocation(PhiKalmanFilterPermutation):
    def __init__(self, L, tau_vphi, sigma_obs_phi, init_certain=False, gam=None, whitespace=False):
        assert whitespace is False, " > ERROR: Location-based filter must NOT operate in white space!"
        # # #  We first initialize the velocity-based filter,...
        # super() also takes care of passing self
        super().__init__(L, tau_vphi, sigma_obs_phi, init_certain, gam, whitespace)
        # # #  ...and then clean up the mess.
        self.numVelo = 0                                                # no velocities
        self.extend_permutation_to_velocities = lambda gam: gam         # the full permutation is the permutation
        self.set_permutation(self.gam)                                  # apply it
        self.F_base = np.zeros(self.N)                                  # no linear component (the eye(N)-term is added when propagating through time
        self.Q_base = self.tau_vphi/2 * L @ L.T                         # Process noise: *Two* components are missing here (included in propagation):
                                                                        # 1) Propagation through time, multiplies a factor 1/fps
                                                                        # 2) Adaptation to correct velo-cov, multiplies another factor 1/fps
        self.H = np.eye(self.N)                                         # observation matrix
        self.R = self.sigma_obs_phi**2 * np.eye(self.N)                 # assumed observation noise in phi (Has to be extended if occlusions could occur!)

    def init_filter(self, s):
        """Init with true state"""
        self.mu = s[:self.N]
        self.Sig = np.zeros((self.N+self.numVelo, self.N+self.numVelo))
        self.Sig[:self.N,:self.N] = self.sigma_obs_phi**2 * np.eye(self.N)
        if self.init_certain:
            self.Sig /= 1000.
        # time is zero
        self.t_last = 0.
        self._is_propagated = False
        self.archive = dict(t=[self.t_last], mu=[self.mu], Sig=[self.Sig], gam=[self.gam.perm.copy()])

    def propagate_to_time(self, t):
        assert self.t_last is not None, " > ERROR: Initialize filter state first!"
        assert self._is_propagated is False, " > ERROR: Trying to propagate already propagated filter." # This could (in principle) be allowed.
        # observation time update
        dt = t - self.t_last              # Imporantly, this also affects our choice of Process noise level 2)!
        self.t_last = t
        assert dt >= 0, " > ERROR: Negative interval since last observation!"
        # Let's hack in the Kalman update equations...
        N = self.N
        F = np.eye(N+self.numVelo) + dt * self.F_base
        Q = dt**2 * self.Q_base           # Here, we accommodate 1) and 2) of above remark!
        H = self.H
        R = self.R
        # Prior moments
        mu_hat = F @ self.mu
        mu_hat[:N] %= (2*pi)

        Sig_hat = F @ self.Sig @ F.T + Q
        # Kalman gain
        S_res = R + H @ Sig_hat @ H.T
        K = Sig_hat @ H.T @ np.linalg.inv(S_res)
        self.propagated_variables = dict(   # All we need for updates:
            t = t,                          # time
            mu_hat = mu_hat,                # mean
            Sig_hat = Sig_hat,              # cov
            K = K                           # Kalman gain
            )
        self._is_propagated = True


    def calculate_residual_mean(self, x, gam=None):
        """mu_hat under current propagation assuming permutation gam (defaults to self.gam)."""
        assert self.propagated_variables["t"] == self.t_last, " > ERROR: Inconsistent internal time information."
        mu_hat = self.propagated_variables["mu_hat"]
        if gam is None:
            gam_full = self.gam_full
        else:
            gam_full = self.extend_permutation_to_velocities(gam)
        N = self.N
        H = self.H
        mu_res = (gam_full(x[:N]) - H @ mu_hat)             # <<--- THIS IS WHERE THE PERMUTATION ENTERS! and [:N] the only change to the parent class
        mu_res[:N] += pi
        mu_res[:N] %= (2*pi)
        mu_res[:N] -= pi
        return mu_res


# # #   KALMAN FILTER FOR LOCATION PREDICTION TASKS  # # #

class PhiKalmanFilterLocPredict(PhiKalmanFilterPermutation):
    def __init__(self, sigma_obs_phi, tau_vphi, L=None, B=None, lam=None, whitespace=False, valid_max_variance=None, logger=None):
        assert (L is not None) and (B is None) and (lam is None) or (L is None) and (B is not None) and (lam is not None), " > Initialize either via 'L' or ('B', 'lam')."
        if L is None:
            self.L = B @ np.diag(lam)
            self.B = B
            self.lam = lam
        else:
            self.L = L
        # # #  We first initialize the permutation-based filter,...
        # super() also takes care of passing self
        super().__init__(L=self.L, tau_vphi=tau_vphi, sigma_obs_phi=sigma_obs_phi, init_certain=False, gam=None, whitespace=whitespace)
        # # #  ...and then clean up the mess.
        self.valid_max_variance = valid_max_variance
        self.logger = logger
        self.isCalledFromPropagateAndIntegrate = False      # Will be used to guarantee correct handling of masked observation arrays

    def observation_likelihood(self, x, perm=None, differential=True, location_only=True):
        raise NotImplementedError("Not yet implemented (must work with masked input arrays).")

    def clone(self, perm=None, differential=True):
        raise NotImplementedError("Not yet implemented (must use adapted init kwargs).")

    def check_valid_max_variance(self, t):
        if self.valid_max_variance is None:
            return
        varloc = self.Sig[:self.N,:self.N].diagonal()    # Location variances
        if (varloc > self.valid_max_variance).any():
            idx = (varloc > self.valid_max_variance).nonzero()[0]
            warnstr = " > WARNING: valid_max_variance (%.3f) exceeded at time t=%.3fs for idx=%s." % (self.valid_max_variance, t, str(idx))
            if self.logger is not None:
                self.logger.warning(warnstr)
            else:
                print(warnstr)

    # Wrap propagation function (see below for explanation)
    _propagate_to_time = PhiKalmanFilterPermutation.propagate_to_time
    def propagate_to_time(self, t):
        assert self.isCalledFromPropagateAndIntegrate, "ERROR: Must be called from method 'propagate_and_integrate_observation'!"
        ret = self._propagate_to_time(t)
        return ret

    # Wrap integration function (see below for explanation)
    _integrate_observation = PhiKalmanFilterPermutation.integrate_observation
    def integrate_observation(self, x):
        assert self.isCalledFromPropagateAndIntegrate, "ERROR: Must be called from method 'propagate_and_integrate_observation'!"
        ret = self._integrate_observation(x)
        return ret

    def propagate_and_integrate_observation(self, x, t):
        """
        We reimplement the propagate & integrate method and will enforce that it is being used.
        Then two small modifications permit to use incomplete observations (= masked arrays x):
          1) Observation cov matrix R --> infty at masked values
          2) turn x into a 'normal' (unmasked) array during integration.
        """
        self.isCalledFromPropagateAndIntegrate = True
        self._R_unmasked = self.R.copy()        # save the 'normal' observation cov matrix
        if isinstance(x, ma.masked_array):
            mask = x.mask
            x_unmasked = np.array(x)
        elif isinstance(x, (np.ndarray, list, tuple)):
            mask = np.zeros(x.shape, dtype=bool)
            x_unmasked = x
        else:
            raise Exception
        self.R += numerical_infty * np.diag(mask).astype(float)  # the mask includes the masked velocities
        self.propagate_to_time(t)
        self.integrate_observation(x_unmasked)
        self.check_valid_max_variance(t)
        self.R[:] = self._R_unmasked            # restore the 'normal' observation cov matrix
        self.isCalledFromPropagateAndIntegrate = False
        return self.mu, self.Sig



# # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # #   A G E N T S  F O R   D E C I S I O N S  # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # #

class PhiArtificialAgentLocPredict(object):
    def __init__(self, f_choice, n_draws, seed):
        self.rng = np.random.RandomState(seed)
        self.f_choice = lambda mu_kal, Sig_kal: f_choice(mu_kal, Sig_kal, rng=self.rng)
        self.n_draws = n_draws

    def draw_response(self, mu_kal, Sig_kal):
        assert isinstance(mu_kal, np.ndarray)
        assert isinstance(Sig_kal, np.ndarray)
        assert mu_kal.ndim == 1, "Mean of choices must be one-dimensional."
        assert Sig_kal.shape == (len(mu_kal), len(mu_kal)), "Cov matrix does not match."
        return [self.f_choice(mu_kal, Sig_kal) for i in range(self.n_draws)]



# # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # #   D P V I   P A R T I C L E   # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # #

class DPVI_Particle (object):
    def __init__(self, kal, score=1.):
        self.kal = kal
        self.score = score

    def query(self, x, perm=None, differential=True, location_only=True):
        Fx = self.kal.observation_likelihood(x, perm=perm, differential=differential, location_only=location_only)
        new_score = self.score * Fx
        return new_score

    def select(self, score, perm=None, differential=True):
        if perm is None:        # We simply keep the particle
            keep = True
        elif (differential is True) and (perm == Permutation(size=self.kal.N)):
            keep = True
        elif (differential is False) and (perm == self.kal.gam):
            keep = True
        else:
            keep = False
        # keep or clone?
        if keep:
            self.score = score
            return self
        else:
            kal = self.kal.clone(perm=perm, differential=differential)
            c = type(self)(kal, score)
            return c


# # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # #     D P V I   F I L T E R     # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # #


# # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # #       DPVI EQUALITY FUNCS     # # # # # # #

# (3A) Define a quick proxy to test for equality
def eqf_proxy(p1, perm1, s1, p2, perm2, s2):
    gam1 = p1.kal.gam * perm1
    gam2 = p2.kal.gam * perm2
    if (gam1 == gam2) and np.isclose(s1, s2):
        return True
    else:
        return False


def eqf_by_perm_only(p1, perm1, s1, p2, perm2, s2):
    gam1 = p1.kal.gam * perm1
    gam2 = p2.kal.gam * perm2
    if gam1 == gam2:
        return True
    else:
        return False


def eqf_unique_snowflake(p1, perm1, s1, p2, perm2, s2):
    return False


# # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # #       DPVI QUERY ROUTINES     # # # # # # #

def qr_naive(DPVI, x, equality_func):
    maxP = DPVI.maxP
    P = DPVI.P
    perm_proposal = DPVI.perm_proposal
    # (2) Calculate the new score for all particles under all proposals
    ps = []
    perms = []
    scores = []
    for p in P:
        for perm in perm_proposal:
            s = p.query(x, perm=perm, differential=True, location_only=True)
            ps.append(p)
            perms.append(perm)
            scores.append(s)
    # (3) Eliminate duplicates
    # (3B) Throw away duplicates
    unique_ps = []
    unique_perms = []
    unique_scores = []
    for i,(p1, perm1, s1) in enumerate(zip(ps, perms, scores)):
        for (p2, perm2, s2) in zip(ps[i+1:], perms[i+1:], scores[i+1:]):
            if equality_func(p1, perm1, s1, p2, perm2, s2):
                continue
        unique_ps.append(p1)
        unique_perms.append(perm1)
        unique_scores.append(s1)
    # (4) Select the (maximally) maxP best particles
    sorted_idx = np.argsort(unique_scores)[::-1][:maxP]   # highest score first
    particles = [ unique_ps[i] for i in sorted_idx ]
    permutations = [ unique_perms[i] for i in sorted_idx ]
    scores = [ unique_scores[i] for i in sorted_idx ]
    return (particles, permutations, scores)


def qr_heapq(DPVI, x, equality_func):
    maxP = DPVI.maxP
    P = DPVI.P
    perm_proposal = DPVI.perm_proposal
    # (2) Calculate the new score for all particles under all proposals
    import heapq
    heap = []
    idx = 0     # The running index only serves for the heappush to always find an ordering.
    for p in P:
        for perm in perm_proposal:
            s = p.query(x, perm=perm, differential=True, location_only=True)
            heapq.heappush(heap, (-s, idx, p, perm, s))
            idx += 1
    particles, permutations, scores = [], [], []
    while (len(heap) > 0) and (len(particles) < maxP):
        _,_,p1,perm1,s1 = heapq.heappop(heap)
        discard = False
        for p2,perm2,s2 in zip(particles, permutations, scores):
            # If we are "identical" to an already selected candidate,
            # the already selected one is better and we can discard the new candidate
            if equality_func(p1, perm1, s1, p2, perm2, s2):
                discard = True
                break
        if not discard:
            # no hit? Take the particle!
            particles.append(p1)
            permutations.append(perm1)
            scores.append(s1)
    return (particles, permutations, scores)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # #     THE ACTUAL DPVI FILTER    # # # # # # #

class DPVI_Particle_Filter (object):
    def __init__(self, maxParticles, perm_proposal, equality_func):
        self.maxP = maxParticles
        assert isinstance(maxParticles, int) and (maxParticles > 0), " > ERROR: maxParticles must be positive integer!"
        self.set_proposal_permutations(perm_proposal)
        self.equality_func = equality_func
        # We start with one particle
        self.P = None

    def set_proposal_permutations(self, perm_proposal):
        self.perm_proposal = perm_proposal

    def set_initial_kal(self, kal):
        p = DPVI_Particle(kal)
        self.P = [p]

    def propagate_evaluate_select_and_integrate_observation(self, x, t, query_routine=qr_heapq):
        # (1) Propagate all particles' Kalman filters from t_last to t.
        for p in self.P:
            p.kal.propagate_to_time(t)
        particles, permutations, scores = query_routine(self, x, self.equality_func)
        smax = scores[0]
        P = []
        for (pi, permi, si) in zip(particles, permutations, scores):
            p = pi.select(si/smax, perm=permi)
            P.append(p)
        self.P = P
        # (5) Actually integrate the observation
        for p in self.P:
            p.kal.integrate_observation(x)
        return smax




























