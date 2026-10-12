"""Represent and refine (model results): tuning, feature importance, learning curves and test diagnostics."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix

from .config import AQI_BREAKS, AQI_LABELS, LIMIT_PM25, VIOLET, AQUA, ORANGE, INK, INK2, MUTED, AXIS, NEUTRAL
from .viz import save, wash, hgrid

# ordinal ramp for forecast horizons (darker = nearer horizon)
H_COLORS = {1: "#2e2470", 3: "#6f62c8", 6: "#a59ee0"}


def tuning_chart(tuning, best_id):
    fig, ax = plt.subplots(figsize=(8.6, 3.4))
    x = np.arange(len(tuning))
    cols = [VIOLET if t["config"] == best_id else NEUTRAL for t in tuning]
    ax.bar(x, [t["val_mae"] for t in tuning], color=cols, width=0.55)
    for i, t in enumerate(tuning):
        ax.text(i, t["val_mae"] + 0.05, f"{t['val_mae']:.2f}", ha="center", fontsize=8, color=INK2)
    ax.set_xticks(x, [f"L = {t['lookback']} h\nH = {t['hidden']}" for t in tuning])
    lo = min(t["val_mae"] for t in tuning)
    ax.set_ylim(lo - 1.5, max(t["val_mae"] for t in tuning) + 0.6)
    ax.set_ylabel("Validation MAE (ug/m3)")
    ax.set_xlabel("Look-back window L and LSTM hidden units H (selected configuration in violet)")
    hgrid(ax)
    fig.tight_layout()
    save(fig, "g19_tuning")


def importance_chart(imp, labels):
    d = imp[:12][::-1]
    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    ax.barh([labels.get(r["feature"], r["feature"]) for r in d], [r["rmse_increase"] for r in d],
            xerr=[r["sd"] for r in d], color=VIOLET, height=0.6, error_kw=dict(ecolor=INK2, lw=0.8, capsize=2))
    ax.set_xlabel("Increase in validation RMSE when the feature is shuffled (ug/m3)")
    hgrid(ax, "x")
    fig.tight_layout()
    save(fig, "g20_permutation_importance")


def learning_curves(hist):
    ep = np.arange(1, len(hist["train_loss"]) + 1)
    be = hist["best_epoch"]
    for key_tr, key_va, ylabel, name in [("train_loss", "val_loss", "Huber loss on standardised log(1 + PM2.5)", "g21_loss_curve"),
                                         ("train_mae", "val_mae", "MAE over 6 horizons (ug/m3)", "g22_mae_curve")]:
        fig, ax = plt.subplots(figsize=(7.4, 3.5))
        ax.plot(ep, hist[key_tr], color=VIOLET, lw=2, marker="o", ms=4, mec="white", mew=0.8,
                label="Training" if key_tr == "train_loss" else "Training (monitor sample)")
        ax.plot(ep, hist[key_va], color=ORANGE, lw=2, marker="s", ms=4, mec="white", mew=0.8, label="Validation")
        ax.axvline(be, color=MUTED, lw=1, ls=":")
        ax.text(be + 0.2, ax.get_ylim()[1] * 0.98, f"best epoch {be}", fontsize=8, va="top", color=INK2)
        ax.set_xlabel("Epoch"); ax.set_ylabel(ylabel)
        ax.legend()
        hgrid(ax)
        fig.tight_layout()
        save(fig, name)


def forecast_window(times, obs, p1, p6, start, end):
    m = (times >= pd.Timestamp(start)) & (times <= pd.Timestamp(end))
    fig, ax = plt.subplots(figsize=(10.4, 3.7))
    ax.plot(times[m], obs[m], color=INK, lw=2, label="Observed PM2.5")
    ax.plot(times[m], p1[m], color=VIOLET, lw=1.6, label="LSTM forecast, 1 hour ahead")
    ax.plot(times[m], p6[m], color=ORANGE, lw=1.6, label="LSTM forecast, 6 hours ahead")
    ax.axhline(LIMIT_PM25, color=INK2, lw=1, ls=":")
    ax.text(times[m][0], LIMIT_PM25 + 8, "75 ug/m3 limit", fontsize=7.8, color=INK2)
    ax.set_ylabel("PM2.5 (ug/m3)")
    ax.legend(loc="upper left")
    ax.tick_params(axis="x", rotation=20)
    hgrid(ax)
    fig.tight_layout()
    save(fig, "g23_forecast_window")


def scatter_pred_obs(y, p, table):
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
    for a, h in zip(ax, [0, 5]):
        hb = a.hexbin(np.log10(y[:, h]), np.log10(np.clip(p[:, h], 1, None)), gridsize=45, cmap="Purples", bins="log", mincnt=1)
        lim = [0.2, 3.0]
        a.plot(lim, lim, color=ORANGE, lw=1.6)
        a.set_xlim(lim); a.set_ylim(lim)
        a.set_xlabel("Observed, log10(ug/m3)")
        a.set_ylabel("Forecast, log10(ug/m3)")
        r = table[h]
        a.set_title(f"{'a' if h == 0 else 'b'}  t + {h + 1} h:  R2 = {r['R2']:.3f},  MAE = {r['MAE']:.1f}", loc="left")
        cb = fig.colorbar(hb, ax=a, pad=0.02)
        cb.set_label("Forecasts (log scale)")
        cb.outline.set_visible(False)
    fig.tight_layout()
    save(fig, "g24_pred_vs_obs")


def residuals(y, p):
    e1, e6 = p[:, 0] - y[:, 0], p[:, 5] - y[:, 5]
    fig, ax = plt.subplots(1, 2, figsize=(10.4, 3.7), gridspec_kw={"width_ratios": [1.1, 1]})
    bins = np.linspace(-120, 120, 97)
    ax[0].hist(np.clip(e1, -120, 120), bins=bins, histtype="stepfilled", color=wash(VIOLET, 0.3), edgecolor=VIOLET, lw=1.1,
               label="t + 1 h")
    ax[0].hist(np.clip(e6, -120, 120), bins=bins, histtype="step", color=ORANGE, lw=1.6, label="t + 6 h")
    lap = stats.laplace.fit(e1)
    xs = np.linspace(-120, 120, 400)
    ax[0].plot(xs, stats.laplace.pdf(xs, *lap) * len(e1) * (bins[1] - bins[0]), color=INK, ls="--", lw=1.1,
               label=f"Laplace fit to t + 1 (scale {lap[1]:.1f})")
    ax[0].set_yscale("log"); ax[0].set_ylim(1, None)
    ax[0].set_xlabel("Forecast error = forecast - observed (ug/m3)")
    ax[0].set_ylabel("Count (log scale)")
    ax[0].set_title("a  Error distribution", loc="left")
    ax[0].legend(fontsize=8)
    bands = [0, 35, 75, 115, 150, 250, 2000]
    lab = ["<35", "35-75", "75-115", "115-150", "150-250", ">250"]
    cat = pd.cut(y[:, 0], bands, labels=lab, right=False)
    out = {}
    for h, e, col, mk in [(1, e1, VIOLET, "o"), (6, e6, ORANGE, "s")]:
        g = pd.Series(np.abs(e)).groupby(cat, observed=False).mean()
        b = pd.Series(e).groupby(cat, observed=False).mean()
        ax[1].plot(range(len(lab)), g.values, color=col, lw=2, marker=mk, ms=5, mec="white", mew=0.8, label=f"MAE, t + {h} h")
        ax[1].plot(range(len(lab)), b.values, color=col, lw=1.2, ls=":", marker=mk, ms=3.5, label=f"Mean bias, t + {h} h")
        out[h] = {"mae_by_band": {k: round(float(v), 2) for k, v in g.items()},
                  "bias_by_band": {k: round(float(v), 2) for k, v in b.items()}}
    ax[1].axhline(0, color=AXIS, lw=0.8)
    ax[1].set_xticks(range(len(lab)), lab)
    ax[1].set_xlabel("Observed PM2.5 band at t + 1 (ug/m3)")
    ax[1].set_ylabel("ug/m3")
    ax[1].set_title("b  Error grows with concentration", loc="left")
    ax[1].legend(fontsize=7.5, ncol=2, loc="lower left")
    hgrid(ax[1])
    fig.tight_layout()
    save(fig, "g25_residuals")
    return {"laplace_scale_t1": round(float(lap[1]), 2), "by_band": out,
            "median_abs_err_t1": round(float(np.median(np.abs(e1))), 2), "median_abs_err_t6": round(float(np.median(np.abs(e6))), 2)}


def horizon_skill(tab_lstm, tab_pers, tab_snaive):
    h = [r["horizon_h"] for r in tab_lstm]
    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    ax.plot(h, [r["MAE"] for r in tab_lstm], color=VIOLET, lw=2, marker="o", ms=6, mec="white", mew=1, label="LSTM")
    ax.plot(h, [r["MAE"] for r in tab_pers], color=ORANGE, lw=2, marker="s", ms=6, mec="white", mew=1, label="Persistence (last value)")
    ax.plot(h, [r["MAE"] for r in tab_snaive], color=AQUA, lw=2, marker="^", ms=6, mec="white", mew=1, label="Same hour yesterday")
    for a, b in zip(tab_lstm, tab_pers):
        ax.text(a["horizon_h"], a["MAE"] - 2.4, f"{100 * (1 - a['MAE'] / b['MAE']):.0f}%", ha="center", fontsize=8, color=INK2)
    ax.set_xlabel("Forecast horizon (hours ahead)")
    ax.set_ylabel("Test MAE (ug/m3)")
    ax.set_xticks(h)
    ax.legend(loc="center left", fontsize=8.3)
    ax.text(0.99, 0.03, "numbers under the LSTM line: MAE reduction versus persistence", transform=ax.transAxes,
            ha="right", fontsize=7.5, style="italic", color=INK2)
    hgrid(ax)
    fig.tight_layout()
    save(fig, "g26_horizon_skill")


def aqi_confusion(y, p):
    fig, ax = plt.subplots(1, 2, figsize=(10.6, 4.6))
    out = {}
    for a, h in zip(ax, [0, 5]):
        t = np.digitize(y[:, h], AQI_BREAKS[1:-1])
        f = np.digitize(p[:, h], AQI_BREAKS[1:-1])
        cm = confusion_matrix(t, f, labels=range(6))
        rown = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        a.imshow(rown, cmap="Purples", vmin=0, vmax=1)
        for i in range(6):
            for j in range(6):
                if cm[i, j]:
                    a.text(j, i, f"{100 * rown[i, j]:.0f}", ha="center", va="center", fontsize=8,
                           color="white" if rown[i, j] > 0.5 else INK)
        a.set_xticks(range(6), AQI_LABELS, rotation=35, ha="right", fontsize=8)
        a.set_yticks(range(6), AQI_LABELS, fontsize=8)
        a.set_xlabel("Forecast category"); a.set_ylabel("Observed category")
        for sp in a.spines.values():
            sp.set_visible(False)
        exact = np.trace(cm) / cm.sum()
        within1 = sum(cm[i, j] for i in range(6) for j in range(6) if abs(i - j) <= 1) / cm.sum()
        a.set_title(f"{'a' if h == 0 else 'b'}  t + {h + 1} h: exact {100 * exact:.1f}%, within one {100 * within1:.1f}%",
                    loc="left", fontsize=9.5)
        out[h + 1] = {"exact": round(float(exact), 4), "within_one": round(float(within1), 4), "cm": cm.tolist()}
    fig.tight_layout()
    save(fig, "g27_aqi_confusion")
    return out


def exceedance_roc(y, p, persistence):
    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    out = {}
    for h in (1, 3, 6):
        lab = (y[:, h - 1] > LIMIT_PM25).astype(int)
        fpr, tpr, _ = roc_curve(lab, p[:, h - 1])
        auc = roc_auc_score(lab, p[:, h - 1])
        ax.plot(fpr, tpr, color=H_COLORS[h], lw=2, label=f"LSTM, t + {h} h (AUC = {auc:.3f})")
        yhat = (p[:, h - 1] > LIMIT_PM25).astype(int)
        tn, fp, fn, tp = confusion_matrix(lab, yhat).ravel()
        out[h] = {"auc": round(float(auc), 4), "tpr": round(tp / (tp + fn), 4), "fpr": round(fp / (fp + tn), 4),
                  "precision": round(tp / max(tp + fp, 1), 4), "prevalence": round(float(lab.mean()), 4)}
        ax.plot(out[h]["fpr"], out[h]["tpr"], "o", color=H_COLORS[h], ms=7, mec="white", mew=1.5)
    lab6 = (y[:, 5] > LIMIT_PM25).astype(int)
    auc_p = roc_auc_score(lab6, persistence)
    fpr, tpr, _ = roc_curve(lab6, persistence)
    ax.plot(fpr, tpr, color=ORANGE, lw=2, ls="--", label=f"Persistence, t + 6 h (AUC = {auc_p:.3f})")
    ax.plot([0, 1], [0, 1], color=MUTED, lw=0.9, ls=":")
    ax.set_xlabel("False alarm rate (FPR)"); ax.set_ylabel("Hit rate (TPR)")
    ax.legend(loc="lower right", fontsize=8)
    ax.text(0.02, 0.97, "dots: operating point when the alert\nthreshold is the 75 ug/m3 limit itself",
            transform=ax.transAxes, va="top", fontsize=7.5, style="italic", color=INK2)
    hgrid(ax, "both")
    fig.tight_layout()
    save(fig, "g28_exceedance_roc")
    out["persistence_auc_t6"] = round(float(auc_p), 4)
    return out
