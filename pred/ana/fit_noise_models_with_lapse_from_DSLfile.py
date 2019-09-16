import numpy as np
import pylab as pl
import os
from lib_predict_io import dist_mod2pi, load_simulation_data, load_experiment_data, cfgfname
tr, inv, log, det, pi, sqrt = np.trace, np.linalg.inv, np.log, np.linalg.det, np.pi, np.sqrt
from scipy.stats import multivariate_normal as mvn
import pandas as pd

# # # # PARAMETERS # # # #

exppath = "./data/paper/"
simpath = "./data/sim/"
outFilename = "fitResults_predict_MarApr2019.pkl.zip"

from DSLs_predict_MarApr2019 import experiment_label, conditions, subjects, DSLs
nKals = 11

VERBOSE = False

IGNORE_COV = False      # restrict the fit to Sig =  a * [[1,0],[0,0]] + b * [[0,0],[0,1]]
FITLAPSE = True              # If False, assume l0 (fixed)
USEGRAD = True          # Try to use gradient information for numerical optimization

a0, b0 = 0.1, 0.1
l0 = 0.02                    # lapse probability

# # # # \PARAMETERS # # # #

assert not (IGNORE_COV and USEGRAD), "ERROR: Gradients are for the full covariance matrix."


# Helper matrices
M1 = np.zeros((2,2))
M1[0,0] = 1.
M2 = np.zeros((2,2))
M2[1,1] = 1.

if IGNORE_COV:
    # THE ONE WITHOUT OFF-DIAGONALS
    gen_Sig = lambda Sig_kal: lambda a, b: a*M1 + b*M2
else:
    # THE REAL ONE
    gen_Sig = lambda Sig_kal: lambda a, b: a*np.eye(len(Sig_kal)) + b*Sig_kal/np.mean(Sig_kal.diagonal())


def calc_logl_of_trial(x, mu, Sig, l):
    pdf = mvn(mean=[0]*len(mu), cov=Sig).pdf
    d = dist_mod2pi
    assert x.shape == (2,)
    perm = lambda x,i : x if i==0 else x[::-1]
    ll = log( (1-l) * pdf( d(perm(x,0),mu) )  +  l * pdf( d(perm(x,1),mu) ) )
    return ll

def log_likelihood(theta, X, Mu, Sig_kal):
    if FITLAPSE:
        a,b,l = theta
    else:
        a,b = theta
        l = l0
    Sig = gen_Sig(Sig_kal)(a, b)
    return np.sum( [ calc_logl_of_trial(x, mu, Sig, l) for (x,mu) in zip(X,Mu) ] )

nSubjects = len(subjects)
nConditions = len(conditions)

# # # # AUX FUNCTIONS # # # #

# derivatives
def dLL_da(theta, X, Mu, Sig_kal):
    Sk = Sig_kal
    if FITLAPSE:
        a,b,l = theta
    else:
        a,b = theta
        l = l0
    Sig = gen_Sig(Sk)(a, b)
    pdf = mvn(mean=[0]*len(Mu[0]), cov=Sig).pdf
    I = np.eye(len(Mu[0]))
    d = dist_mod2pi
    perm = lambda x,i : x if i==0 else x[::-1]
    iS = inv(Sig)
    grad = 0.
    for (x,mu) in zip(X,Mu):
        x0 = d(perm(x,0),mu)
        x1 = d(perm(x,1),mu)
        c1 = (1-l) * (-1/2) * tr(iS) * pdf(x0)
        c2 = (1-l) * 1/2 * np.sum(iS @ np.outer(x0,x0) @ iS * I) * pdf(x0)
        c3 = l * (-1/2) * tr(iS) * pdf(x1)
        c4 = l * 1/2 * np.sum(iS @ np.outer(x1,x1) @ iS * I) * pdf(x1)
        grad += (c1+c2+c3+c4) / np.exp(calc_logl_of_trial(x, mu, Sig, l))
    return grad

def dLL_db(theta, X, Mu, Sig_kal):
    Sk = Sig_kal
    Snorm = np.mean(Sk.diagonal())
    if FITLAPSE:
        a,b,l = theta
    else:
        a,b = theta
        l = l0
    Sig = gen_Sig(Sk)(a, b)
    pdf = mvn(mean=[0]*len(Mu[0]), cov=Sig).pdf
    I = np.eye(len(Mu[0]))
    d = dist_mod2pi
    perm = lambda x,i : x if i==0 else x[::-1]
    iS = inv(Sig)
    grad = 0.
    for (x,mu) in zip(X,Mu):
        x0 = d(perm(x,0),mu)
        x1 = d(perm(x,1),mu)
        c1 = (1-l) * (-1/2) * tr(iS@Sk/Snorm) * pdf(x0)
        c2 = (1-l) * 1/2 * np.sum(iS @ np.outer(x0,x0) @ iS * Sk/Snorm) * pdf(x0)
        c3 = l * (-1/2) * tr(iS@Sk/Snorm) * pdf(x1)
        c4 = l * 1/2 * np.sum(iS @ np.outer(x1,x1) @ iS * Sk/Snorm) * pdf(x1)
        grad += (c1+c2+c3+c4) / np.exp(calc_logl_of_trial(x, mu, Sig, l))
    return grad

def dLL_dl(theta, X, Mu, Sig_kal):
    Sk = Sig_kal
    if FITLAPSE:
        a,b,l = theta
    else:
        a,b = theta
        l = l0
    Sig = gen_Sig(Sk)(a, b)
    pdf = mvn(mean=[0]*len(Mu[0]), cov=Sig).pdf
    I = np.eye(len(Mu[0]))
    d = dist_mod2pi
    perm = lambda x,i : x if i==0 else x[::-1]
    grad = 0.
    for (x,mu) in zip(X,Mu):
        x0 = d(perm(x,0),mu)
        x1 = d(perm(x,1),mu)
        grad += (-pdf(x0) + pdf(x1)) /  np.exp(calc_logl_of_trial(x, mu, Sig, l))
    return grad

f_grad = lambda *args: -np.array([dLL_da(*args), dLL_db(*args), dLL_dl(*args)])


def df_empty(columns, dtypes, index=None):
    import pandas as pd
    assert len(columns)==len(dtypes)
    df = pd.DataFrame(index=index)
    for c,d in zip(columns, dtypes):
        df[c] = pd.Series(dtype=d)
    return df

def append_to_df(df, dtypes, *X):
    assert len(X) == len(dtypes)
    N = len(X[0])
    for i,x in enumerate(X):
        assert len(x) == N
        assert isinstance(x[0], dtypes[i])
    for n in range(N):
        idx = len(df)
        data = tuple([x[n] for x in X])
        df.loc[idx] = data


def fit_model_to_subject(Phi_human, Phi_kal, Sig_kal):
    from scipy.optimize import minimize
    f_opt = lambda *args: -1 * log_likelihood(*args)
    if FITLAPSE:
        bounds = ([0.0001, np.inf], [0.0001, np.inf], [0.0001, 1.])
        x0 = [a0, b0, l0]
    else:
        bounds = ([0.0001, np.inf], [0.0001, np.inf])
        x0 = [a0, b0]
    if USEGRAD:
        assert FITLAPSE, "ERROR: Requires lapse"
        opt_kwargs = dict(method="SLSQP", bounds=bounds,  options={'disp' : VERBOSE, 'ftol' : 1.e-6})
        res = minimize( fun=f_opt, jac=f_grad, x0=x0, args=(Phi_human, Phi_kal, Sig_kal), **opt_kwargs )
        if not res.success:
            print("   > Gradient-based did not converge. Attempt purely numerical with more iterations.")
            opt_kwargs = dict(method="SLSQP", bounds=bounds,  options={'disp' : VERBOSE, 'eps' : 1.4901161193847656e-5, 'ftol' : 1.0e-3, 'maxiter' : 400})
            res = minimize( fun=f_opt, x0=x0, args=(Phi_human, Phi_kal, Sig_kal), **opt_kwargs )
    else:
        opt_kwargs = dict(method="SLSQP", bounds=bounds,  options={'disp' : VERBOSE, 'eps' : 1.4901161193847656e-5, 'ftol' : 1.0e-3, 'maxiter' : 200})
        res = minimize( fun=f_opt, x0=x0, args=(Phi_human, Phi_kal, Sig_kal), **opt_kwargs )
    if not res.success:
        print("* "*29, "\n* * *  WARNING: Fit did NOT converge successfully! * * *\n" + "* "*29)
    ll = log_likelihood(res.x, Phi_human, Phi_kal, Sig_kal)
    if FITLAPSE:
        a,b,l = res.x
    else:
        a,b = res.x
        l = l0
    Sig_opt = gen_Sig(Sig_kal)(a,b)
    resdict = dict(a=a, b=b, l=l, ll=ll, Sig_opt=Sig_opt)
    return resdict

# # # #  PREPARATION and DATA  # # # #


def fit_models(DSL):
    # # # IMPORT FROM EXP  # # #
    from shutil import copyfile
    copyfile(cfgfname(exppath, DSL["experiment"]), "./tmp_config.py")
    import tmp_config
    import importlib
    importlib.reload(tmp_config)
    # from tmp_config import
    targets, B, lam = tmp_config.targets, tmp_config.B, tmp_config.lam
    L_exp = B @ np.diag(lam)        # We will use this as a check.

    # # # LOAD EXP DATA  # # #
    # The targets are returned in canonical order (e.g. [5,6])
    _, _, Phi_true, Phi_pred = load_experiment_data(exppath, DSL["experiment"])

    # # # LOAD SIM DATA  # # #
    kalnames, kaldata = load_simulation_data(path_sim=simpath, dsl_sim=DSL['kals_noiseless'], targets=targets)
    assert (nKals == len(kalnames)), "nKals = %d, but found %d in the data!" % (nKals, len(kalnames))
    TRUname = "TRU" if "TRU" in kalnames else "groundtruth"
    if kaldata[TRUname]["Phi"].ndim == 3:
        # discard the time data
        for kname in kalnames:
            kaldata[kname]["Phi"] = kaldata[kname]["Phi"][:,-1,:]
            kaldata[kname]["Sig"] = kaldata[kname]["Sig"][:,-1,:]

    if kaldata[TRUname]["L"].shape  == L_exp.shape:
        assert (kaldata[TRUname]["L"] == L_exp).all(), "Argh! Motion struct of kalman['TRU'] does not match world!"
    else:
        Covsim = kaldata[TRUname]["L"] @ kaldata[TRUname]["L"].T
        Covexp = L_exp @ L_exp.T
        assert (Covsim == Covexp).all(), "Argh! Motion struct of kalman['TRU'] does not match world!"
    assert (kaldata[TRUname]["Phi"].shape == Phi_true.shape), "Trials or targets do not match!"

    LL = np.zeros(nKals)
    A = np.zeros(nKals)
    B = np.zeros(nKals)
    L = np.zeros(nKals)
    NF = np.zeros(nKals)
    for kn, kname in enumerate(kalnames):
        if VERBOSE: print("   > Fitting model (%d/%d): '%s'." % (kn+1, nKals, kname))
        Sig_kal=kaldata[kname]['Sig'][0]
        resdict = fit_model_to_subject(Phi_human=Phi_pred, Phi_kal=kaldata[kname]["Phi"], Sig_kal=Sig_kal)
        ll, a, b, l = resdict["ll"], resdict["a"], resdict["b"], resdict["l"]
        noise_fraction = a / (a + b)
        LL[kn] = ll
        A[kn] = a
        B[kn] = b
        L[kn] = l
        NF[kn] = noise_fraction
    return kalnames, LL, A, B, L, NF


LL = np.zeros((nConditions, nSubjects, nKals))
A = np.zeros((nConditions, nSubjects, nKals))
B = np.zeros((nConditions, nSubjects, nKals))
L = np.zeros((nConditions, nSubjects, nKals))
NF = np.zeros((nConditions, nSubjects, nKals))

cols = ("subj", "cond", "kal", "a", "b", "pL", "nf", "ll", "*")
dtypes = (str, str, str, float, float, float, float, float, str)
df = df_empty(columns=cols, dtypes=dtypes)


for cn, cond in enumerate(conditions):
    for sn, subj in enumerate(subjects):
        if subj in DSLs[cond]:
            DSL = DSLs[cond][subj]
            kalnames, LL[cn,sn], A[cn,sn], B[cn,sn], L[cn,sn], NF[cn,sn] = fit_models(DSL)
            best_fitting = LL[cn,sn].argmax()
            best_str = ["*" if (n == best_fitting) else "" for n in range(nKals)]
            append_to_df(df, dtypes, [subj]*nKals, [cond]*nKals, kalnames, A[cn,sn], B[cn,sn], L[cn,sn], NF[cn,sn], LL[cn,sn], best_str)
            print("\n# # #  RESULTS FOR: subject %s condition %s  # # #" % (subj, cond))
            for kn, kname in enumerate(kalnames):
                print("Model: %20s" % kname + ", a=%7.4f b=%7.4f, NoiseFrac=%5.1f%%, p_lapse=%.3f, LogLikelihood (higher is better): %.4f" % ( A[cn,sn,kn], B[cn,sn,kn], NF[cn,sn,kn]*100, L[cn,sn,kn], LL[cn,sn,kn]) + ("*" if kn==best_fitting else "") )
        else:
            print(" > Skip condition '%s' -> subject '%s' because no DSL was defined." % (cond, subj))


if outFilename is not None:
    print(" > Store fit results to file: %s" % outFilename )
    df.to_pickle(outFilename, compression="gzip")
else:
    print(" > Warning: Fit results NOT saved!")
