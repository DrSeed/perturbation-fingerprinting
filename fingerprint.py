# RNA-fingerprinting-style perturbation mapping: denoised fingerprints + sum-of-single-effects regression.
import numpy as np
from scipy.special import logsumexp


def baseline_factors(controls, n_factors=10):
    """Mean and top PCs of control cells (cells x genes): the variation every cell has anyway."""
    mu = controls.mean(axis=0)
    _, _, Vt = np.linalg.svd(controls - mu, full_matrices=False)
    return mu, Vt[:n_factors].T


def remove_baseline(Y, mu, Phi):
    R = Y - mu
    return R - (R @ Phi) @ Phi.T


def learn_fingerprints(controls, perturbed, labels, n_factors=10):
    """Per perturbation: residuals after removing baseline factors, then a rank-one SVD.
    The leading gene vector is the fingerprint, signed so it points along the mean residual."""
    mu, Phi = baseline_factors(controls, n_factors)
    names = sorted(set(labels))
    fps = []
    for k in names:
        R = remove_baseline(perturbed[labels == k], mu, Phi)
        v = np.linalg.svd(R, full_matrices=False)[2][0]
        fps.append(v if v @ R.mean(axis=0) >= 0 else -v)
    return np.array(fps).T, names  # genes x K, unit-norm columns


def _single_effect(X, r, s2, prior_grid):
    """One-effect Bayesian regression for every prior variance in the grid at once; keep the best."""
    xtx = (X ** 2).sum(axis=0)
    bhat, shat2 = X.T @ r / xtx, s2 / xtx
    s02 = prior_grid[:, None]
    lbf = 0.5 * np.log(shat2 / (s02 + shat2)) + 0.5 * bhat ** 2 / shat2 * s02 / (s02 + shat2)
    mean_lbf = logsumexp(lbf, axis=1) - np.log(X.shape[1])
    g = int(np.argmax(mean_lbf))
    alpha = np.exp(lbf[g] - logsumexp(lbf[g]))
    return alpha, s02[g, 0] / (s02[g, 0] + shat2) * bhat, mean_lbf[g]


def susie(X, y, s2, L=3, n_iter=50, prior_grid=np.logspace(-2, 2, 25)):
    """Sum of single effects (Wang et al. 2020) fitted by IBSS. The prior variance of each effect is
    chosen by maximum marginal likelihood on a grid; an effect with no support is switched off.
    Returns per-effect inclusion probabilities (L x K) and log Bayes factors."""
    K = X.shape[1]
    alpha, mean, lbf = np.full((L, K), 1 / K), np.zeros((L, K)), np.zeros(L)
    for _ in range(n_iter):
        old = alpha.copy()
        for l in range(L):
            r = y - X @ ((alpha * mean).sum(axis=0) - alpha[l] * mean[l])
            a, m, b = _single_effect(X, r, s2, prior_grid)
            alpha[l], mean[l], lbf[l] = (a, m, b) if b > 0 else (np.full(K, 1 / K), np.zeros(K), 0.0)
        if np.abs(alpha - old).max() < 1e-4:
            break
    return alpha, lbf


def credible_sets(alpha, lbf, X, coverage=0.95, min_lbf=3.0, min_purity=0.4):
    """Smallest set of perturbations holding `coverage` of each effect's posterior. Dropped if the
    effect is weak (log BF) or the set is impure (members not alike), as in SuSiE."""
    C = np.abs(np.corrcoef(X.T))
    sets = []
    for a, b in zip(alpha, lbf):
        if b < min_lbf:
            continue
        order = np.argsort(a)[::-1]
        members = order[: np.searchsorted(np.cumsum(a[order]), coverage) + 1]
        purity = np.median(C[np.ix_(members, members)]) if len(members) > 1 else 1.0
        if purity >= min_purity and not any(set(members) == set(s) for s in sets):
            sets.append(members)
    return sets


def map_cells(query_controls, query_cells, fingerprints, n_factors=10, L=3):
    """For each query cell: remove the query's own baseline, regress on fingerprints, return credible sets."""
    mu, Phi = baseline_factors(query_controls, n_factors)
    s2 = remove_baseline(query_controls, mu, Phi).var()
    R = remove_baseline(query_cells, mu, Phi)
    out = []
    for r in R:
        alpha, lbf = susie(fingerprints, r, s2, L=L)
        out.append((credible_sets(alpha, lbf, fingerprints), alpha, lbf))
    return out
