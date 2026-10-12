"""Represent and refine: every exploratory chart used in the report (colour-blind-safe, print-ready)."""
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_rgba
from matplotlib.patches import FancyBboxPatch, Patch
from scipy import stats

from .config import (FIG_DIR, NUMERIC, WIND_DIRS, NORTHERLY, SOUTHERLY, URBAN, LIMIT_PM25, WHO_24H,
                     VIOLET, AQUA, ORANGE, INK, INK2, MUTED, GRID, AXIS, NEUTRAL)
from . import stats_tools as stt

DIVERGING = LinearSegmentedColormap.from_list(
    "violet_orange", ["#2e2470", "#4a3aa7", "#a59ee0", "#f0efec", "#f5b294", "#eb6834", "#9c3a12"])


def wash(color, alpha=0.18):
    return to_rgba(color, alpha)


def save(fig, name):
    """Write a 220-dpi PNG and, inside Jupyter, also display the figure inline."""
    fig.savefig(FIG_DIR / f"{name}.png", bbox_inches="tight", facecolor="white")
    if "ipykernel" in sys.modules:
        plt.show()
    plt.close(fig)


def hgrid(ax, axis="y"):
    ax.grid(axis=axis, color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)


# --------------------------------------------------------------------------- data quality
def missing_heatmap(mm):
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    im = ax.imshow(mm.values, cmap="Purples", vmin=0, vmax=max(8, float(mm.values.max())), aspect="auto")
    for i in range(mm.shape[0]):
        for j in range(mm.shape[1]):
            v = mm.values[i, j]
            ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=7.2, color="white" if v > 5 else INK)
    ax.set_xticks(range(mm.shape[1]), mm.columns, rotation=0, fontsize=8)
    ax.set_yticks(range(mm.shape[0]), mm.index, fontsize=8)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    cb = fig.colorbar(im, ax=ax, pad=0.015)
    cb.set_label("Missing hours (%)")
    cb.outline.set_visible(False)
    fig.tight_layout()
    save(fig, "g01_missing_heatmap")


def missing_profile(miss_pct, gaps):
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5), gridspec_kw={"width_ratios": [1, 1.15]})
    s = pd.Series(miss_pct).sort_values(ascending=False)
    ax[0].bar(range(len(s)), s.values, color=VIOLET, width=0.6)
    for i, v in enumerate(s.values):
        ax[0].text(i, v + 0.08, f"{v:.2f}", ha="center", fontsize=7.2, color=INK2)
    ax[0].set_xticks(range(len(s)), s.index, rotation=45, ha="right")
    ax[0].set_ylabel("Missing (% of 420,768 hours)")
    ax[0].set_title("a  Missing values by variable", loc="left")
    hgrid(ax[0])
    edges = np.array([1, 2, 3, 4, 7, 13, 25, 49, 97, 400])
    counts, _ = np.histogram(gaps, bins=edges)
    labels = ["1", "2", "3", "4-6", "7-12", "13-24", "25-48", "49-96", ">96"]
    cum = 100 * np.cumsum(counts) / counts.sum()
    ax[1].bar(range(len(counts)), counts, color=VIOLET, width=0.6)
    for i, (c, cu) in enumerate(zip(counts, cum)):
        ax[1].text(i, c + 25, f"{c:,}\n(cum. {cu:.0f}%)", ha="center", va="bottom", fontsize=7, color=INK2)
    ax[1].set_ylim(0, counts.max() * 1.25)
    ax[1].set_xticks(range(len(counts)), labels)
    ax[1].set_xlabel("Length of a PM2.5 gap (consecutive missing hours)")
    ax[1].set_ylabel("Number of gaps")
    ax[1].set_title("b  How long are the PM2.5 gaps?", loc="left")
    hgrid(ax[1])
    fig.tight_layout()
    save(fig, "g02_missing_profile")


def outlier_panels(st_raw, flags, both):
    """a: one flagged single-hour spike in context; b: PM2.5 versus PM10 physical-consistency check."""
    fig, ax = plt.subplots(1, 2, figsize=(10.4, 3.7), gridspec_kw={"width_ratios": [1.35, 1]})
    s = st_raw["PM2.5"]
    med = s.rolling(7, center=True, min_periods=3).median()
    hits = s[flags & (s > med)]
    centre = (hits - med[hits.index]).idxmax() if len(hits) else s.idxmax()
    win = s.loc[centre - pd.Timedelta(hours=36): centre + pd.Timedelta(hours=36)]
    mwin = np.expm1(np.log1p(win).rolling(7, center=True, min_periods=3).median())
    ax[0].plot(win.index, win.values, color=VIOLET, lw=1.6, label="Observed hourly PM2.5")
    ax[0].plot(mwin.index, mwin.values, color=INK2, lw=1.1, ls="--", label="7-hour rolling median")
    f = flags.reindex(win.index).fillna(False).astype(bool)
    ax[0].plot(win.index[f], win[f], "o", color=ORANGE, ms=8, mec="white", mew=1.5, label="Flagged single-hour spike")
    ax[0].set_ylabel("PM2.5 (ug/m3)")
    ax[0].set_title("a  Spike filter at work (Dongsi)", loc="left")
    ax[0].legend(loc="upper left")
    ax[0].tick_params(axis="x", rotation=25)
    hgrid(ax[0])
    hb = ax[1].hexbin(np.log10(both["PM10"]), np.log10(both["PM2.5"]), gridsize=55, cmap="Purples", bins="log", mincnt=1)
    lim = [0.2, 3.05]
    ax[1].plot(lim, lim, color=ORANGE, lw=1.6, label="PM2.5 = PM10")
    ax[1].set_xlim(lim); ax[1].set_ylim(lim)
    ax[1].set_xlabel("log10 PM10"); ax[1].set_ylabel("log10 PM2.5")
    share = 100 * (both["PM2.5"] > both["PM10"]).mean()
    ax[1].set_title(f"b  {share:.1f}% of hours have PM2.5 > PM10", loc="left")
    ax[1].legend(loc="lower right")
    cb = fig.colorbar(hb, ax=ax[1], pad=0.02)
    cb.set_label("Hours (log scale)")
    cb.outline.set_visible(False)
    fig.tight_layout()
    save(fig, "g03_outliers")
    return {"example_time": str(centre), "example_value": float(s[centre]), "example_median": round(float(med[centre]), 1)}


def imputation_example(before, after, col="CO"):
    """Show one isolated 8-24 hour gap (no other gaps within 36 h) before and after interpolation."""
    filled = before[col].isna() & after[col].notna()
    d = np.diff(np.r_[0, filled.to_numpy().astype(int), 0])
    starts, ends = np.where(d == 1)[0], np.where(d == -1)[0]
    pick = None
    for a, b in zip(starts, ends):
        if 8 <= b - a <= 24:
            lo, hi = max(0, a - 36), min(len(filled), b + 36)
            ctx = before[col].iloc[lo:hi].isna().to_numpy().copy()
            ctx[a - lo:b - lo] = False
            if not ctx.any():
                pick = (a, b)
                break
    if pick is None:
        return None
    a, b = pick
    sl = slice(max(0, a - 36), min(len(filled), b + 36))
    t = before.index[sl]
    fig, ax = plt.subplots(figsize=(9, 3.1))
    ax.axvspan(before.index[a] - pd.Timedelta(minutes=30), before.index[b - 1] + pd.Timedelta(minutes=30),
               color=wash(ORANGE, 0.12), lw=0)
    ax.plot(t, after[col].iloc[sl], color=ORANGE, lw=1.6, ls="--", label="Filled by time-based linear interpolation")
    ax.plot(t, before[col].iloc[sl], color=VIOLET, lw=2, marker="o", ms=3, label="Observed hourly values")
    ax.text(before.index[a] + (before.index[b - 1] - before.index[a]) / 2, ax.get_ylim()[1] * 0.96,
            f"{b - a}-hour gap", ha="center", va="top", fontsize=8.5, color=INK2)
    ax.set_ylabel(f"{col} (ug/m3)")
    ax.legend(loc="upper left")
    hgrid(ax)
    fig.tight_layout()
    save(fig, "g04_imputation_example")
    return {"start": str(before.index[a]), "hours": int(b - a)}


# --------------------------------------------------------------------------- temporal structure
def long_term(daily):
    ma = stt.moving_average(daily, 30)
    rm = stt.running_median(daily, 31)
    fig, ax = plt.subplots(figsize=(10.5, 4.2))
    ax.plot(daily.index, daily.values, color=wash(VIOLET, 0.45), lw=0.7, label="Daily mean (12-site average)")
    ax.plot(ma.index, ma.values, color=ORANGE, lw=1.8, label="30-day moving average")
    ax.plot(rm.index, rm.values, color=VIOLET, lw=1.8, label="31-day running median (resistant)")
    ax.axhline(LIMIT_PM25, color=INK2, lw=1, ls=":", label="China Grade II 24-h limit (75 ug/m3)")
    ax.axhline(WHO_24H, color=INK2, lw=1, ls="-.", label="WHO 2021 24-h guideline (15 ug/m3)")
    syear = np.where(daily.index.month >= 3, daily.index.year, daily.index.year - 1)
    means = daily.groupby(syear).mean()
    for y0, m in means.items():
        mid = pd.Timestamp(f"{y0}-09-01")
        ax.text(mid, 492, f"Mar {y0} - Feb {y0 + 1}\nmean {m:.1f}", ha="center", va="top", fontsize=7.8, color=INK2,
                bbox=dict(facecolor="white", edgecolor=GRID, lw=0.6, pad=2))
    ax.set_ylim(0, 500)
    ax.set_ylabel("PM2.5 (ug/m3)")
    hgrid(ax)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3, fontsize=8)
    fig.tight_layout()
    save(fig, "g05_long_term_trend")
    return {int(k): round(float(v), 2) for k, v in means.items()}


def decomposition(monthly):
    trend, seasonal, resid, idx = stt.classical_decomposition(monthly)
    fig, ax = plt.subplots(4, 1, figsize=(9.5, 6.4), sharex=True)
    ax[0].plot(monthly.index, monthly.values, color=VIOLET, lw=1.8, marker="o", ms=4, mec="white", mew=1)
    ax[0].set_ylabel("Observed")
    ax[1].plot(trend.index, trend.values, color=VIOLET, lw=2)
    ax[1].set_ylabel("Trend (2x12 MA)")
    cols = [ORANGE if v > 0 else VIOLET for v in seasonal.values]
    ax[2].bar(seasonal.index, seasonal.values, width=22, color=cols)
    ax[2].axhline(0, color=AXIS, lw=0.8)
    ax[2].set_ylabel("Seasonal")
    ax[2].set_ylim(-32, 42)
    ax[2].legend(handles=[Patch(color=ORANGE, label="Above trend (polluted season)"),
                          Patch(color=VIOLET, label="Below trend (cleaner season)")], loc="upper center", ncol=2, fontsize=7.5)
    ax[3].axhline(0, color=AXIS, lw=0.8)
    ax[3].vlines(resid.index, 0, resid.values, color=INK2, lw=1.2)
    ax[3].plot(resid.index, resid.values, "o", color=INK2, ms=3)
    ax[3].set_ylabel("Irregular")
    ax[-1].set_xlabel("Month")
    for a in ax:
        hgrid(a)
    fig.align_ylabels(ax)
    fig.tight_layout()
    save(fig, "g06_decomposition")
    return trend, seasonal, resid, idx


def hour_month_heatmap(df):
    tab = df.pivot_table(index="month", columns="hour", values="PM2.5", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(10, 4.1))
    im = ax.imshow(tab.values, cmap="Oranges", aspect="auto")
    ax.set_yticks(range(12), ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    ax.set_xticks(range(0, 24, 2), [f"{h:02d}" for h in range(0, 24, 2)])
    ax.set_xlabel("Hour of day (local time)")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    cb = fig.colorbar(im, ax=ax, pad=0.015)
    cb.set_label("Mean PM2.5 (ug/m3)")
    cb.outline.set_visible(False)
    fig.tight_layout()
    save(fig, "g07_hour_month_heatmap")
    return tab


def acf_pacf(x, nlags=72):
    r = stt.acf(x, nlags)
    p = stt.pacf_from_acf(r, nlags)
    band = 1.96 / np.sqrt(len(x))
    fig, ax = plt.subplots(1, 2, figsize=(10.4, 3.3))
    for a, vals, name in [(ax[0], r, "a  Autocorrelation (ACF)"), (ax[1], p, "b  Partial autocorrelation (PACF)")]:
        a.axhspan(-band, band, color=wash(VIOLET, 0.15), lw=0)
        a.axhline(0, color=AXIS, lw=0.8)
        a.vlines(range(nlags + 1), 0, vals, color=VIOLET, lw=1.3)
        a.plot(range(nlags + 1), vals, "o", color=VIOLET, ms=3)
        for d in (24, 48, 72):
            a.axvline(d, color=MUTED, lw=0.8, ls=":")
        a.set_xlabel("Lag (hours)")
        a.set_title(name, loc="left")
    ax[0].set_ylabel("Correlation of log(1 + PM2.5)")
    fig.tight_layout()
    save(fig, "g16_acf_pacf")
    return r, p, band


# --------------------------------------------------------------------------- distributions
def continuous_fit(x, fits):
    fig, ax = plt.subplots(1, 2, figsize=(10.4, 3.9))
    lx = np.log10(x)
    ax[0].hist(lx, bins=70, density=True, color=NEUTRAL, edgecolor="white", lw=0.3, label="Observed hours")
    g = np.linspace(lx.min(), lx.max(), 400)
    xs = 10 ** g
    colors = {"Lognormal": VIOLET, "Gamma": AQUA, "Weibull": ORANGE}
    dist = {"Lognormal": stats.lognorm, "Gamma": stats.gamma, "Weibull": stats.weibull_min}
    for name, f in fits.items():
        ax[0].plot(g, dist[name].pdf(xs, *f["params"]) * xs * np.log(10), color=colors[name], lw=2,
                   label=f"{name} (KS D = {f['KS_D']:.3f})")
    ax[0].set_xlabel("PM2.5, log10(ug/m3)")
    ax[0].set_ylabel("Density")
    ax[0].set_title("a  Candidate densities", loc="left")
    ax[0].legend(loc="upper left", fontsize=8)
    q = np.linspace(0.005, 0.995, 120)
    emp = np.quantile(x, q)
    marks = {"Lognormal": "o", "Gamma": "s", "Weibull": "^"}
    for name, f in fits.items():
        th = dist[name].ppf(q, *f["params"])
        ax[1].plot(th, emp, marks[name], ms=4.5, color=colors[name], mec="white", mew=0.6, label=name)
    lims = [emp.min() * 0.8, emp.max() * 1.25]
    ax[1].plot(lims, lims, color=INK2, lw=1, label="Perfect agreement")
    ax[1].set_xscale("log"); ax[1].set_yscale("log")
    ax[1].set_xlabel("Theoretical quantile (ug/m3)")
    ax[1].set_ylabel("Empirical quantile (ug/m3)")
    ax[1].set_title("b  Q-Q comparison, 0.5th to 99.5th percentile", loc="left")
    ax[1].legend(loc="upper left", fontsize=8)
    hgrid(ax[1], "both")
    fig.tight_layout()
    save(fig, "g08_continuous_fit")


def discrete_fit(k, n, a, b, episodes):
    obs = np.bincount(k, minlength=n + 1)
    N = len(k)
    p = k.mean() / n
    e_bin = N * stats.binom.pmf(np.arange(n + 1), n, p)
    e_bb = N * stats.betabinom.pmf(np.arange(n + 1), n, a, b)
    fig, ax = plt.subplots(1, 2, figsize=(10.6, 3.8), gridspec_kw={"width_ratios": [1.45, 1]})
    ax[0].bar(range(n + 1), obs, color=NEUTRAL, width=0.75, label="Observed station-days")
    ax[0].plot(range(n + 1), e_bin, color=ORANGE, lw=1.6, marker="o", ms=4, mec="white", mew=0.8,
               label=f"Binomial (n = 24, p = {p:.3f})")
    ax[0].plot(range(n + 1), e_bb, color=VIOLET, lw=2, marker="s", ms=4, mec="white", mew=0.8,
               label=f"Beta-binomial (alpha = {a:.2f}, beta = {b:.2f})")
    ax[0].set_xlabel("Hours in the day with PM2.5 > 75 ug/m3")
    ax[0].set_ylabel("Number of station-days")
    ax[0].set_title("a  Daily exceedance hours", loc="left")
    ax[0].legend(loc="upper center", fontsize=8)
    ax[0].set_xticks(range(0, 25, 2))
    hgrid(ax[0])
    L = np.sort(episodes)
    surv = 1 - np.arange(len(L)) / len(L)
    ax[1].step(L, surv, where="post", color=VIOLET, lw=2, label="Observed episodes")
    pg = 1 / episodes.mean()
    lg = np.arange(1, L.max() + 1)
    ax[1].plot(lg, (1 - pg) ** (lg - 1), color=ORANGE, lw=1.6, ls="--", label=f"Geometric (mean = {episodes.mean():.1f} h)")
    ax[1].set_yscale("log")
    ax[1].set_xlim(0, np.quantile(L, 0.999))
    ax[1].set_ylim(1 / len(L), 1.05)
    ax[1].set_xlabel("Episode length (consecutive hours > 75 ug/m3)")
    ax[1].set_ylabel("P(length >= x)")
    ax[1].set_title("b  Survival of pollution episodes", loc="left")
    ax[1].legend(loc="upper right", fontsize=8)
    hgrid(ax[1], "both")
    fig.tight_layout()
    save(fig, "g09_discrete_fit")
    return obs, e_bin, e_bb


# --------------------------------------------------------------------------- spatial / meteorological
def station_boxplots(df):
    order = df.groupby("station")["PM2.5"].median().sort_values().index.tolist()
    data = [df.loc[df.station == s, "PM2.5"].dropna().values for s in order]
    fig, ax = plt.subplots(figsize=(10, 4.2))
    bp = ax.boxplot(data, widths=0.55, patch_artist=True, showfliers=False, medianprops=dict(color=INK, lw=1.6))
    for kk, (box, s) in enumerate(zip(bp["boxes"], order)):
        col = VIOLET if s in URBAN else AQUA
        box.set(facecolor=wash(col, 0.22), edgecolor=col, linewidth=1.3)
        for part in ("whiskers", "caps"):
            bp[part][2 * kk].set(color=col); bp[part][2 * kk + 1].set(color=col)
    ax.plot(range(1, len(order) + 1), [d.mean() for d in data], "D", color=INK, ms=4.5, mec="white", mew=0.8)
    ax.set_yscale("log")
    ax.set_xticks(range(1, len(order) + 1), order, rotation=30, ha="right")
    ax.set_ylabel("Hourly PM2.5 (ug/m3, log scale)")
    ax.axhline(LIMIT_PM25, color=INK2, lw=1, ls=":")
    hgrid(ax)
    ax.legend(handles=[Patch(facecolor=wash(VIOLET, 0.22), edgecolor=VIOLET, label="Urban site"),
                       Patch(facecolor=wash(AQUA, 0.22), edgecolor=AQUA, label="Suburban site"),
                       plt.Line2D([], [], color=INK, marker="D", ls="", ms=4.5, label="Mean"),
                       plt.Line2D([], [], color=INK2, ls=":", label="75 ug/m3 limit")],
              loc="upper center", bbox_to_anchor=(0.5, 1.12), ncol=4, fontsize=8)
    fig.tight_layout()
    save(fig, "g10_station_boxplots")
    return order


def wind_polar(df):
    g = df.groupby("wd")["PM2.5"].agg(["mean", "size"]).reindex(WIND_DIRS)
    theta = np.deg2rad(np.arange(16) * 22.5)
    fig = plt.figure(figsize=(9.6, 4.3))
    for k, (col, title, colour) in enumerate([("mean", "a  Mean PM2.5 by wind direction (ug/m3)", VIOLET),
                                              ("size", "b  Frequency of each direction (% of hours)", AQUA)]):
        ax = fig.add_subplot(1, 2, k + 1, projection="polar")
        vals = g[col].values if col == "mean" else 100 * g[col].values / g[col].sum()
        ax.bar(theta, vals, width=np.deg2rad(19), color=colour, edgecolor="white", lw=0.8, alpha=0.9)
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_xticks(theta, WIND_DIRS, fontsize=7.5)
        ax.tick_params(axis="y", labelsize=7, labelcolor=MUTED)
        ax.set_rlabel_position(int(np.argmin(vals)) * 22.5)      # radial labels sit on the shortest wedge
        ax.grid(color=GRID, lw=0.7)
        ax.spines["polar"].set_color(AXIS)
        ax.set_title(title, fontsize=9.5, pad=14, loc="left", fontweight="bold")
    fig.tight_layout()
    save(fig, "g11_wind_polar")
    return g


def wind_speed_conditional(df):
    bins = [0, 1, 2, 3, 4, 5, 20]
    labels = ["0-1", "1-2", "2-3", "3-4", "4-5", ">5"]
    d = df.dropna(subset=["PM2.5", "WSPM", "wd"]).copy()
    d["sector"] = np.where(d["wd"].isin(NORTHERLY), "Northerly (WNW to NNE)",
                           np.where(d["wd"].isin(SOUTHERLY), "Southerly (ESE to SSW)", "Other"))
    d["band"] = pd.cut(d["WSPM"], bins, labels=labels, right=False)
    fig, ax = plt.subplots(figsize=(8.4, 3.9))
    out = {}
    x = np.arange(len(labels))
    for sector, col, mk in [("Northerly (WNW to NNE)", VIOLET, "o"), ("Southerly (ESE to SSW)", ORANGE, "s")]:
        q = d[d.sector == sector].groupby("band", observed=False)["PM2.5"].quantile([0.25, 0.5, 0.75]).unstack()
        ax.fill_between(x, q[0.25], q[0.75], color=wash(col, 0.13), lw=0)
        ax.plot(x, q[0.5], color=col, marker=mk, ms=6, mec="white", mew=1.2, lw=2, label=f"{sector}: median, IQR band")
        out[sector] = {lab: round(float(v), 1) for lab, v in zip(labels, q[0.5])}
    ax.set_xticks(x, labels)
    ax.set_xlabel("Wind speed (m/s)")
    ax.set_ylabel("PM2.5 (ug/m3)")
    ax.legend(loc="upper right", fontsize=8.3)
    hgrid(ax)
    fig.tight_layout()
    save(fig, "g12_wind_speed_by_sector")
    return out


def spearman_triangle(df):
    c = df[NUMERIC].corr(method="spearman")
    mask = np.triu(np.ones_like(c, dtype=bool), k=1)
    fig, ax = plt.subplots(figsize=(6.8, 5.6))
    shown = np.where(mask, np.nan, c.values)
    im = ax.imshow(shown, cmap=DIVERGING, vmin=-1, vmax=1)
    for i in range(len(c)):
        for j in range(len(c)):
            if not mask[i, j]:
                v = c.values[i, j]
                ax.text(j, i, f"{v:+.2f}" if i != j else "1", ha="center", va="center", fontsize=7.4,
                        color="white" if abs(v) > 0.6 else INK)
    ax.set_xticks(range(len(c)), c.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(c)), c.index)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cb.set_label("Spearman rho (violet negative, orange positive)")
    cb.outline.set_visible(False)
    fig.tight_layout()
    save(fig, "g13_spearman_triangle")
    return c


def co_and_ratio(df):
    d = df.dropna(subset=["PM2.5", "CO"])
    x, y = np.log(d["CO"].values), np.log(d["PM2.5"].values)
    b1, b0 = np.polyfit(x, y, 1)
    rho = stats.spearmanr(x, y).statistic
    fig, ax = plt.subplots(1, 2, figsize=(10.4, 3.9), gridspec_kw={"width_ratios": [1.15, 1]})
    hb = ax[0].hexbin(x, y, gridsize=60, cmap="Purples", bins="log", mincnt=1)
    gx = np.linspace(x.min(), x.max(), 50)
    ax[0].plot(gx, b0 + b1 * gx, color=ORANGE, lw=2, label=f"ln PM2.5 = {b0:.2f} + {b1:.2f} ln CO")
    ax[0].set_xlabel("ln CO (ug/m3)"); ax[0].set_ylabel("ln PM2.5 (ug/m3)")
    ax[0].set_title(f"a  Co-emission signature (Spearman rho = {rho:.2f})", loc="left")
    ax[0].legend(loc="lower right", fontsize=8)
    cb = fig.colorbar(hb, ax=ax[0], pad=0.02)
    cb.set_label("Hours (log scale)")
    cb.outline.set_visible(False)
    r = df.dropna(subset=["PM2.5", "PM10"])
    r = r[(r["PM10"] > 0)]
    ratio = (r["PM2.5"] / r["PM10"]).clip(upper=1.5)
    seasons = ["Winter", "Spring", "Summer", "Autumn"]
    data = [ratio[r["season"] == s].values for s in seasons]
    bp = ax[1].boxplot(data, widths=0.5, patch_artist=True, showfliers=False, medianprops=dict(color=INK, lw=1.6),
                       whiskerprops=dict(color=VIOLET), capprops=dict(color=VIOLET))
    for box in bp["boxes"]:
        box.set(facecolor=wash(VIOLET, 0.2), edgecolor=VIOLET, linewidth=1.3)
    ax[1].axhline(1, color=INK2, lw=1, ls=":")
    ax[1].set_xticks(range(1, 5), seasons)
    ax[1].set_ylabel("PM2.5 / PM10 ratio")
    ax[1].set_title("b  Fine-particle share by season", loc="left")
    hgrid(ax[1])
    fig.tight_layout()
    save(fig, "g14_co_and_ratio")
    return {"slope": round(float(b1), 3), "intercept": round(float(b0), 3), "spearman": round(float(rho), 3),
            "ratio_median": {s: round(float(np.median(v)), 3) for s, v in zip(seasons, data)}}


def season_heating(df):
    seasons = ["Winter", "Spring", "Summer", "Autumn"]
    d = df.dropna(subset=["PM2.5"])
    fig, ax = plt.subplots(1, 2, figsize=(10.2, 3.7), gridspec_kw={"width_ratios": [1.4, 1]})
    data = [np.log10(d.loc[d.season == s, "PM2.5"].values) for s in seasons]
    vp = ax[0].violinplot(data, showmedians=True, showextrema=False, widths=0.75)
    for b in vp["bodies"]:
        b.set_facecolor(wash(VIOLET, 0.3)); b.set_edgecolor(VIOLET); b.set_alpha(None); b.set_linewidth(1)
    vp["cmedians"].set_color(INK); vp["cmedians"].set_linewidth(1.6)
    ax[0].set_xticks(range(1, 5), seasons)
    ax[0].set_yticks([0, 1, 2, 3], ["1", "10", "100", "1000"])
    ax[0].set_ylabel("PM2.5 (ug/m3, log scale)")
    ax[0].set_title("a  Seasonal distributions", loc="left")
    hgrid(ax[0])
    h = [d.loc[d.heating, "PM2.5"].values, d.loc[~d.heating, "PM2.5"].values]
    bp = ax[1].boxplot([np.log10(v) for v in h], widths=0.5, patch_artist=True, showfliers=False,
                       medianprops=dict(color=INK, lw=1.6))
    for kk, col in enumerate([ORANGE, VIOLET]):
        bp["boxes"][kk].set(facecolor=wash(col, 0.22), edgecolor=col, linewidth=1.3)
        bp["whiskers"][2 * kk].set(color=col); bp["whiskers"][2 * kk + 1].set(color=col)
        bp["caps"][2 * kk].set(color=col); bp["caps"][2 * kk + 1].set(color=col)
    ax[1].set_xticks([1, 2], ["Heating season\n(15 Nov - 15 Mar)", "Non-heating"])
    ax[1].set_yticks([0, 1, 2, 3], ["1", "10", "100", "1000"])
    ax[1].set_title("b  Central heating effect", loc="left")
    hgrid(ax[1])
    fig.tight_layout()
    save(fig, "g15_season_heating")
    H, pH = stats.kruskal(*[np.asarray(10 ** v) for v in data])
    U, pU = stats.mannwhitneyu(h[0], h[1], alternative="two-sided")
    n = sum(len(v) for v in data)
    return {"kruskal_H": round(float(H), 1), "kruskal_p": float(pH), "epsilon2": round(float(stt.epsilon_squared(H, n)), 4),
            "season_median": {s: round(float(np.median(10 ** v)), 1) for s, v in zip(seasons, data)},
            "season_mean": {s: round(float(np.mean(10 ** v)), 1) for s, v in zip(seasons, data)},
            "season_p90": {s: round(float(np.quantile(10 ** v, 0.9)), 1) for s, v in zip(seasons, data)},
            "heating_median": round(float(np.median(h[0])), 1), "nonheating_median": round(float(np.median(h[1])), 1),
            "heating_mean": round(float(np.mean(h[0])), 1), "nonheating_mean": round(float(np.mean(h[1])), 1),
            "heating_p90": round(float(np.quantile(h[0], 0.9)), 1), "nonheating_p90": round(float(np.quantile(h[1], 0.9)), 1),
            "mw_p": float(pU), "mw_auc": round(float(U / (len(h[0]) * len(h[1]))), 3)}


# --------------------------------------------------------------------------- model diagram
def lstm_diagram(n_features, lookback, hidden, layers, horizon):
    fig, ax = plt.subplots(figsize=(10.4, 2.9))
    ax.set_xlim(0, 10.4); ax.set_ylim(0, 2.9); ax.axis("off")
    boxes = [(0.1, f"Input window\n{lookback} hours x {n_features} features\n(standardised)", False),
             (2.25, f"LSTM layer 1\n{hidden} units\n(tanh / sigmoid gates)", True),
             (4.4, f"LSTM layer 2\n{hidden} units\ndropout 0.2 between layers", True),
             (6.55, f"Dense head\nLinear({hidden}->{hidden // 2}), ReLU\nDropout 0.2", True),
             (8.7, f"Output\nlast log PM2.5 +\nLinear({hidden // 2}->{horizon}), t+1 ... t+{horizon}", False)]
    for i, (x0, txt, inner) in enumerate(boxes):
        ax.add_patch(FancyBboxPatch((x0, 0.75), 1.6, 1.45, boxstyle="round,pad=0.03,rounding_size=0.1",
                                    facecolor=wash(VIOLET, 0.12) if inner else "#f3f2ee",
                                    edgecolor=VIOLET if inner else INK2, lw=1.3))
        ax.text(x0 + 0.8, 1.48, txt, ha="center", va="center", fontsize=8.1, color=INK)
        if i < len(boxes) - 1:
            ax.annotate("", xy=(x0 + 2.12, 1.48), xytext=(x0 + 1.65, 1.48),
                        arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1.1))
    ax.text(5.2, 0.3, "Residual learning: a skip connection adds the latest observed log PM2.5 to the head's output, so the "
                      "network learns the change from persistence; outputs are inverted with expm1() to ug/m3.",
            ha="center", fontsize=7.8, style="italic", color=INK2)
    save(fig, "g18_lstm_architecture")
