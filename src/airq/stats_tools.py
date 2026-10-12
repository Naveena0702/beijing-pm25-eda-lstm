"""Mine: statistical helpers (distribution fitting, autocorrelation, smoothing and decomposition)."""
import numpy as np
import pandas as pd
from scipy import stats, optimize


# --------------------------------------------------------------------------- continuous fits
CONTINUOUS = {"Lognormal": stats.lognorm, "Gamma": stats.gamma, "Weibull": stats.weibull_min}


def fit_continuous(x, dists=CONTINUOUS):
    """Maximum-likelihood fits with location fixed at 0; returns parameters, AIC, BIC and KS distance."""
    x = np.asarray(x, dtype=float)
    rows = {}
    for name, dist in dists.items():
        params = dist.fit(x, floc=0)
        ll = float(dist.logpdf(x, *params).sum())
        k = len(params) - 1
        rows[name] = {"params": [float(p) for p in params], "loglik": ll,
                      "AIC": 2 * k - 2 * ll, "BIC": k * np.log(len(x)) - 2 * ll,
                      "KS_D": float(stats.kstest(x, dist.cdf, args=params).statistic)}
    return rows


# --------------------------------------------------------------------------- discrete fits
def fit_betabinom(k, n):
    """MLE of a beta-binomial(n, a, b) for counts k out of n trials."""
    k = np.asarray(k)

    def nll(theta):
        a, b = np.exp(theta)
        return -stats.betabinom.logpmf(k, n, a, b).sum()

    m = k.mean() / n
    res = optimize.minimize(nll, [np.log(m + 1e-3), np.log(1 - m + 1e-3)], method="Nelder-Mead",
                            options={"xatol": 1e-8, "fatol": 1e-8, "maxiter": 5000})
    a, b = np.exp(res.x)
    return float(a), float(b), float(-res.fun)


# --------------------------------------------------------------------------- dependence over time
def acf(x, nlags):
    x = np.asarray(x, dtype=float) - np.mean(x)
    denom = np.dot(x, x)
    return np.array([1.0] + [np.dot(x[:-k], x[k:]) / denom for k in range(1, nlags + 1)])


def pacf_from_acf(r, nlags):
    """Durbin-Levinson recursion: partial autocorrelations from autocorrelations."""
    pac = np.zeros(nlags + 1)
    pac[0] = 1.0
    phi = np.zeros((nlags + 1, nlags + 1))
    phi[1, 1] = r[1]
    pac[1] = r[1]
    for k in range(2, nlags + 1):
        num = r[k] - np.dot(phi[k - 1, 1:k], r[1:k][::-1])
        den = 1 - np.dot(phi[k - 1, 1:k], r[1:k])
        phi[k, k] = num / den
        phi[k, 1:k] = phi[k - 1, 1:k] - phi[k, k] * phi[k - 1, 1:k][::-1]
        pac[k] = phi[k, k]
    return pac


# --------------------------------------------------------------------------- smoothing
def moving_average(s, window):
    return s.rolling(window, center=True, min_periods=window // 2).mean()


def running_median(s, window):
    """Resistant smoother: centred running median (insensitive to isolated extreme days)."""
    return s.rolling(window, center=True, min_periods=window // 2).median()


def classical_decomposition(monthly):
    """Additive decomposition of a monthly series: 2x12 centred moving-average trend,
    month-specific seasonal indices (centred to sum to zero) and the remaining irregular part."""
    w = np.r_[1 / 24, np.full(11, 1 / 12), 1 / 24]          # 2x12 MA weights over 13 months
    trend = monthly.rolling(13, center=True).apply(lambda v: float(np.dot(v, w)), raw=True)
    detr = monthly - trend
    idx = detr.groupby(detr.index.month).mean()
    idx = idx - idx.mean()
    seasonal = pd.Series(monthly.index.month.map(idx).to_numpy(), index=monthly.index)
    resid = monthly - trend - seasonal
    return trend, seasonal, resid, idx


# --------------------------------------------------------------------------- effect sizes
def epsilon_squared(h, n):
    """Effect size for the Kruskal-Wallis H statistic."""
    return h / ((n ** 2 - 1) / (n + 1))
