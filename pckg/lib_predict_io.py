import os
import numpy as np

VERBOSE = False

from motionstruct.functions import dist_mod2pi

datafname_exp_pattern = "trial_*.pkl.zip"
cfgfname = lambda path, dsl: os.path.join(path, dsl, "config.py")

def datafname_sim(path, dsl):
    fname_list = ("simdata.pkl", "simdata.pickle.zip")
    for fname in fname_list:
        fname = os.path.join(path, dsl, fname)
        if os.path.isfile(fname):
            return fname


def integrated_hidden_motion(fname):
    """Load the integrated hidden dot motion of a single trial."""
    import pickle, gzip
    if VERBOSE:
        print("   > Loading data from file: %s" % fname)
    with gzip.open(fname, "rb") as f:
        X = pickle.load(f)
    tar = np.sort(X["targets"])
    hidx = np.array([ not(tar[0] in v) for v in X["visible"] ])
    ht = X["t"][hidx][:-1]
    hphi = X["Phi"][hidx][:,tar]
    dhphi = dist_mod2pi( hphi[1:] - hphi[:-1] )
    from scipy.integrate import cumtrapz
    hphiint = cumtrapz(dhphi, dx=1., axis=0)[-1]
    h0 = hidx.argmax()
    hphi0 = X["Phi"][h0,tar]
    return hphi0, hphiint

def load_integrated_hidden_motion(path_exp, dsl_exp):
    """Load the integrated hidden dot motion of DSL."""
    print(" > Searching for trial data.")
    import glob
    filelist = glob.glob(os.path.join(path_exp, dsl_exp, datafname_exp_pattern))
    filelist = np.sort(filelist)
    nTrials = len(filelist)
    print(" > %d trials found." % nTrials)
    Hphi0 = []
    Hphiint = []
    for n, fname in enumerate(filelist):
        hphi0, hphiint = integrated_hidden_motion(fname)
        Hphi0.append(hphi0)
        Hphiint.append(hphiint)
    return Hphi0, Hphiint


def load_phi_true_and_pred(fname, targets):
    """Load for one experiment trial: true value and human prediction."""
    import pickle, gzip
    if VERBOSE:
        print("   > Loading data from file: %s" % fname)
    datadict = pickle.load(gzip.open(fname, "rb"))
    phi_true = datadict['Phi'][-1,targets]      # the "real" targets
    idx = datadict['targets'].argsort()         # order of target selection
    phi_pred = datadict['prediction'][idx]
    return phi_true, phi_pred


def load_experiment_data(path_exp, dsl_exp):
    """Load for all experiment trials: true value and human prediction."""
    print(" > Load experiment cfg from file: %s" % cfgfname(path_exp, dsl_exp))
    from shutil import copyfile
    copyfile(cfgfname(path_exp, dsl_exp), "./tmp_config.py")
    from tmp_config import targets, B, lam
    L_exp = B @ np.diag(lam)
    nTarget = len(targets)
    print(" > Searching for trial data.")
    import glob
    filelist = glob.glob(os.path.join(path_exp, dsl_exp, datafname_exp_pattern))
    filelist = np.sort(filelist)
    nTrials = len(filelist)
    print(" > %d trials found." % nTrials)
    Phi_true = np.zeros((nTrials,nTarget))
    Phi_pred = np.zeros((nTrials,nTarget))
    for n, fname in enumerate(filelist):
        phi_true, phi_pred = load_phi_true_and_pred(fname, targets)
        Phi_true[n] = phi_true
        Phi_pred[n] = phi_pred
    return targets, L_exp, Phi_true, Phi_pred


def load_response_times(path_exp, dsl_exp):
    # Find trial list
    print(" > Searching for trial data.")
    import os, glob
    filelist = glob.glob(os.path.join(path_exp, dsl_exp, datafname_exp_pattern))
    filelist = np.sort(filelist)
    nTrials = len(filelist)
    print(" > %d trials found." % nTrials)
    RTraw = []
    Targets = []
    import pickle, gzip
    from datetime import datetime
    # load times and targets
    for fname in filelist:
        with  gzip.open( fname, 'rb' ) as f:
            X = pickle.load(f)
        t_animstopped = datetime.strptime(X['choicetimes'][1], "%Y-%m-%d %H:%M:%S.%f")
        RTraw.append([ (datetime.strptime(tstr, "%Y-%m-%d %H:%M:%S.%f") - t_animstopped).total_seconds() \
                        for tstr in X['choicetimes'][2:] ])
        Targets.append(X['targets'])
    return np.array(RTraw), np.array(Targets)


def find_matching_trials(path_exp, dsl_exp):
    # Find trial list
    print(" > Searching for trial data.")
    import os, glob
    filelist = glob.glob(os.path.join(path_exp, dsl_exp, datafname_exp_pattern))
    filelist = np.sort(filelist)
    nTrials = len(filelist)
    print(" > %d trials found." % nTrials)
    samedict = dict()
    import pickle, gzip
    # load seeds
    for fname in filelist:
        with  gzip.open( fname, 'rb' ) as f:
            X = pickle.load(f)
        seed = X["trial_seed"]
        idx = X["trial_number"]
        if seed in samedict:
            samedict[seed].append(idx)
        else:
            samedict[seed] = [idx]
    # sanity checks
    R = None
    for seed in samedict:
        if R is None:
            R = len(samedict[seed])
        assert len(samedict[seed]) == R
    assert nTrials == R * len(samedict)
    return samedict


def load_simulation_data(path_sim, dsl_sim, targets):
    """Load for all trials: Kalman filter predictions under multiple models."""
    from shutil import copyfile
    copyfile(cfgfname(path_sim, dsl_sim), "./tmp_config_sim.py")
    import tmp_config_sim
    import importlib
    importlib.reload(tmp_config_sim)
    # from tmp_config_sim import cfg
    cfg = tmp_config_sim.cfg
    import pickle, gzip
    print(" > Unpickle data from archive: %s" % datafname_sim(path_sim, dsl_sim))
    print("   > This may take a moment...", end="", flush=True)
    archive = pickle.load(gzip.open( datafname_sim(path_sim, dsl_sim), 'rb' ) )
    print("Done.")
    # Discart non-kalmal data
    general_keys = ('obs_t', 'wld_t', 'obs_X', 'wld_S')
    for key in general_keys:
        # np.array(archive.pop(key))
        archive.pop(key)
    # get the kalnames
    kalnames = list(archive.keys())
    fixed_names = ["groundtruth", "TRU", "independent", "IND", "global", "GLO", "CNT", "clusters", "CLU", "CLI", "hierarchy (simple)", "SDH", "hierarchy (complex)", "CDH"]
    kalnames_sorted = []
    for name in fixed_names:
        if name in kalnames:
            kalnames_sorted += [kalnames.pop(kalnames.index(name))]
    kalnames = kalnames_sorted + kalnames
    # load that thing
    kaldata = dict()
    optional_data = ("choice", "points")
    # print(kalnames)
    for kname in kalnames:
        X = archive[kname]
        Phi = np.array(X['kal_mu'])[:,-1,targets]
        Sig = np.array(X['kal_Sig'])[:,-1,targets][:,:,targets]         # They are all identical anyway
        L = cfg["tracker"][kname]['B'] @ np.diag(cfg["tracker"][kname]['lam'])
        kaldata[kname] = dict(L=L, Phi=Phi, Sig=Sig)
        for key in optional_data:
            if key in X:
                kaldata[kname][key] = np.array(X[key])
    return kalnames, kaldata
