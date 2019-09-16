import numpy as np
import pylab as pl
from itertools import product
from lib_predict_io import find_matching_trials, load_experiment_data, load_simulation_data
from motionstruct.functions import dist_mod2pi

def score_sep(vb, vn):
    """We combine var_bias and Sig_noise in a score,
       ranging from 0 (only bias) to 1 (only noise)
    """
    vn = vn.diagonal()
    vb = np.array(vb) if np.ndim(vb) == 1 else np.array(vb).diagonal()
    return vn / (vb + vn)


# # #   PARAMS
from DSLs_predict_MarApr2019 import experiment_label, conditions, subjects, DSLs

score = score_sep

path_exp = "./data/paper/"
path_sim = "./data/sim/"

conditions = list(conditions)

outfname_data = "bias_variance_analysis_%s.pkl.zip" %experiment_label

# # #  END OF PARAMS

def df_empty(columns, dtypes, index=None):
    import pandas as pd
    assert len(columns)==len(dtypes)
    df = pd.DataFrame(index=index)
    for c,d in zip(columns, dtypes):
        df[c] = pd.Series(dtype=d)
    return df


columns = ("subj", "cond", "Sig_all", "Sig_var", "Var_Sig_var", "Sig_bias", "score_green", "score_red")
dtypes =  ( str,    str,    object,    object,    object,        object,     float,        float   )
df = df_empty(columns=columns, dtypes=dtypes)

for s,c in product( subjects, conditions ):
    # LOAD DATA
    if s not in DSLs[c]:
        print("\n\n * * *  Skipping %s %s  * * *\n\n" % (s,c))
        continue
    dsl_exp = DSLs[c][s]["experiment"]
    dsl_sim = DSLs[c][s]["kals_noiseless"]
    samedict = find_matching_trials(path_exp, dsl_exp)
    targets, L_exp, Phi_true, Phi_pred = load_experiment_data(path_exp, dsl_exp)
    kalnames, kaldata = load_simulation_data(path_sim, dsl_sim, targets)
    Phi_opt = kaldata["TRU"]["Phi"]
    # Extract some params
    N = len(samedict)
    P = len(list(samedict.values())[0])
    D = Phi_opt.shape[1]
    assert Phi_opt.shape == (N*P, D)
    assert Phi_pred.shape == (N*P, D)
    # Prepare sorted arrays
    Mu_opt = np.zeros((N, D))
    X = np.zeros((N,P,D))
    for n,seed in enumerate(samedict):
        for p,tr in enumerate(samedict[seed]):
            if p == 0:
                Mu_opt[n] = Phi_opt[tr-1]
            else:
                # WARNING: Check atol for fully noiseless sims
                assert np.allclose(dist_mod2pi(Mu_opt[n], Phi_opt[tr-1]), 0. , rtol=0., atol=0.01)
            X[n,p] = Phi_pred[tr-1]
    # ESTIMATE Bias and Variance
    # CIRCULAR Estimators
    from scipy.stats import circmean
    Mu_opt %= 2*np.pi
    X %= 2*np.pi
    Xmean = circmean(X, axis=1)
    estimator_Sig_noise = np.array([ np.cov(x.T, ddof=1) for x in dist_mod2pi(X - Xmean[:,None,:]) ]).mean(0)
    varESN =  np.array([ np.cov(x.T, ddof=1) for x in dist_mod2pi(X - Xmean[:,None,:]) ]).var(0)
    variance_vs_opt = np.array([ np.diag((x**2).mean(0)) for x in dist_mod2pi(X - Mu_opt[:,None,:]) ]).mean(0)
    estimator_var_bias = variance_vs_opt - estimator_Sig_noise
    # Append to df
    sg, sr = score(estimator_var_bias, estimator_Sig_noise)
    data = (s, c, variance_vs_opt, estimator_Sig_noise, varESN, estimator_var_bias, sg, sr)
    df.loc[len(df)] = data

    print( "\n\nEstimates for %s on condition %s" % (s,c))
    print( "Estimate, assuming a circular world:" )
    print( "Est. score: %s\nvar_bias =\n%s\nSig_noise =\n%s" % ( str(score(estimator_var_bias, estimator_Sig_noise)), str(estimator_var_bias), str(estimator_Sig_noise) ) )
    print( "\n\n")


if outfname_data is not None:
    print(" > Store fit results to file: %s" % outfname_data )
    df.to_pickle(outfname_data, compression="gzip")
else:
    print(" > Warning: Fit results NOT saved!")

