"""Acquire, parse and filter: loading the 12 station files, auditing quality, cleaning and feature engineering."""
import zipfile

import numpy as np
import pandas as pd

from .config import DATA_ZIP, NUMERIC, POLLUTANTS, WIND_DIRS, MAX_INTERP_GAP


# --------------------------------------------------------------------------- acquire / parse
def load_raw(zip_path=DATA_ZIP):
    """Read every station CSV straight from the UCI zip archive and add a datetime column."""
    frames = []
    with zipfile.ZipFile(zip_path) as z:
        for name in sorted(z.namelist()):
            if name.endswith(".csv"):
                with z.open(name) as fh:
                    frames.append(pd.read_csv(fh))
    df = pd.concat(frames, ignore_index=True)
    df["datetime"] = pd.to_datetime(df[["year", "month", "day", "hour"]])
    return df.sort_values(["station", "datetime"]).reset_index(drop=True)


# --------------------------------------------------------------------------- filter: audit
def run_lengths(mask):
    """Lengths of consecutive True runs in a boolean array (used for gap analysis)."""
    m = np.asarray(mask, dtype=int)
    d = np.diff(np.r_[0, m, 0])
    return np.where(d == -1)[0] - np.where(d == 1)[0]


def audit(df):
    """Collect every data-quality fact quoted in the report."""
    out = {"rows": int(len(df)), "stations": int(df["station"].nunique()),
           "start": str(df["datetime"].min()), "end": str(df["datetime"].max()),
           "exact_duplicates": int(df.duplicated().sum()),
           "key_duplicates": int(df.duplicated(["station", "datetime"]).sum())}
    expected = pd.date_range(df["datetime"].min(), df["datetime"].max(), freq="h")
    out["hours_per_station"] = int(len(expected))
    out["complete_index"] = bool(all(len(g) == len(expected) for _, g in df.groupby("station")))
    miss = df[NUMERIC + ["wd"]].isna().sum()
    out["missing_counts"] = {k: int(v) for k, v in miss.items()}
    out["missing_pct"] = {k: round(100 * v / len(df), 2) for k, v in miss.items()}
    out["rows_with_any_missing_pct"] = round(100 * df[NUMERIC + ["wd"]].isna().any(axis=1).mean(), 2)
    gaps = np.concatenate([run_lengths(g["PM2.5"].isna()) for _, g in df.groupby("station")])
    out["pm25_gaps"] = {"n": int(len(gaps)), "median": float(np.median(gaps)), "max": int(gaps.max()),
                        "share_le_3h": round(100 * (gaps <= 3).mean(), 1),
                        "share_le_24h": round(100 * (gaps <= 24).mean(), 1),
                        "hours_in_gaps_gt_24h_pct": round(100 * gaps[gaps > 24].sum() / gaps.sum(), 1)}
    out["capped_values"] = {"PM2.5>=999": int((df["PM2.5"] >= 999).sum()), "PM10>=999": int((df["PM10"] >= 999).sum()),
                            "CO>=10000": int((df["CO"] >= 10000).sum()), "O3>500": int((df["O3"] > 500).sum())}
    both = df.dropna(subset=["PM2.5", "PM10"])
    out["pm25_gt_pm10"] = {"rows": int((both["PM2.5"] > both["PM10"]).sum()), "of": int(len(both)),
                           "pct": round(100 * (both["PM2.5"] > both["PM10"]).mean(), 2),
                           "gross_rows": int((both["PM2.5"] > 1.1 * both["PM10"] + 10).sum())}
    return out, gaps


def missing_matrix(df):
    """Percentage of missing hours for every station x variable pair."""
    return df.groupby("station")[NUMERIC + ["wd"]].apply(lambda g: g.isna().mean() * 100)


# Lunar New Year's Eve and the Lantern Festival: fireworks cause genuine one-hour spikes,
# so the spike filter is switched off from 18:00 on these evenings until 06:00 next morning.
FIREWORK_EVENINGS = ["2014-01-30", "2014-02-14", "2015-02-18", "2015-03-05",
                     "2016-02-07", "2016-02-22", "2017-01-27", "2017-02-11"]


def firework_mask(index):
    m = np.zeros(len(index), dtype=bool)
    for day in FIREWORK_EVENINGS:
        t0 = pd.Timestamp(day) + pd.Timedelta(hours=18)
        m |= (index >= t0) & (index <= t0 + pd.Timedelta(hours=12))
    return m


def hampel_flags(series, window=7, n_sigmas=4.0, min_ratio=2.0, min_abs=25.0):
    """Flag single-hour glitches in PM2.5.

    A reading is flagged only when ALL of the following hold (computed on log(1 + x)):
      1. Hampel rule: it lies more than `n_sigmas` robust SDs (1.4826 x MAD) from the
         centred `window`-hour rolling median;
      2. it differs from that median by more than a factor of `min_ratio` and by more
         than `min_abs` ug/m3 (so low-level sensor noise is ignored);
      3. it jumps away from BOTH neighbouring hours in the same direction (a one-hour
         spike or dip) - genuine haze episodes build up over several hours and are kept;
      4. it is not on a fireworks evening.
    """
    s = series.astype(float)
    x = np.log1p(s)
    med = x.rolling(window, center=True, min_periods=3).median()
    mad = (x - med).abs().rolling(window, center=True, min_periods=3).median()
    dev = x - med
    lr = np.log(min_ratio)
    robust = dev.abs() > n_sigmas * 1.4826 * mad
    big = (dev.abs() > lr) & ((s - np.expm1(med)).abs() > min_abs)
    single = ((x - x.shift(1) > lr) & (x - x.shift(-1) > lr)) | ((x.shift(1) - x > lr) & (x.shift(-1) - x > lr))
    return robust & big & single & ~firework_mask(s.index)


# --------------------------------------------------------------------------- filter: cleaning
def fill_short_gaps(s, max_gap=MAX_INTERP_GAP):
    """Time-based linear interpolation for gaps of at most `max_gap` hours only."""
    isna = s.isna().to_numpy()
    if not isna.any():
        return s, 0
    lengths = np.zeros(len(s), dtype=int)
    d = np.diff(np.r_[0, isna.astype(int), 0])
    for a, b in zip(np.where(d == 1)[0], np.where(d == -1)[0]):
        lengths[a:b] = b - a
    interp = s.interpolate(method="time", limit_area="inside")
    fill = isna & (lengths <= max_gap) & interp.notna().to_numpy()
    out = s.copy()
    out[fill] = interp[fill]
    return out, int(fill.sum())


def clean_station(df_station, max_gap=MAX_INTERP_GAP):
    """Clean one station: remove instrument caps, de-spike PM2.5, interpolate short gaps."""
    st = df_station.sort_values("datetime").set_index("datetime").copy()
    log = {}
    caps = {"PM2.5": st["PM2.5"] >= 999, "PM10": st["PM10"] >= 999, "CO": st["CO"] >= 10000, "O3": st["O3"] > 500}
    for col, m in caps.items():
        log[f"cap_removed_{col}"] = int(m.sum())
        st.loc[m, col] = np.nan
    spikes = hampel_flags(st["PM2.5"])
    log["hampel_spikes_PM2.5"] = int(spikes.sum())
    st.loc[spikes, "PM2.5"] = np.nan
    for col in NUMERIC:
        st[col], n = fill_short_gaps(st[col], max_gap)
        log[f"interpolated_{col}"] = n
    # wind direction is categorical: carry the last observed direction across short gaps
    wd_missing = st["wd"].isna()
    st["wd"] = st["wd"].ffill(limit=max_gap)
    log["ffilled_wd"] = int((wd_missing & st["wd"].notna()).sum())
    log["remaining_missing_any"] = int(st[NUMERIC + ["wd"]].isna().any(axis=1).sum())
    return st, log


# --------------------------------------------------------------------------- transform
def add_features(st):
    """Physically meaningful encodings: wind vector components and cyclical calendar terms."""
    st = st.copy()
    angle = np.deg2rad(st["wd"].map({d: 22.5 * i for i, d in enumerate(WIND_DIRS)}).astype(float))
    # meteorological convention: direction the wind blows FROM, so the flow vector is reversed
    st["wind_u"] = -st["WSPM"] * np.sin(angle)
    st["wind_v"] = -st["WSPM"] * np.cos(angle)
    idx = st.index
    for name, value, period in [("hour", idx.hour, 24), ("month", idx.month - 1, 12), ("dow", idx.dayofweek, 7)]:
        st[f"{name}_sin"] = np.sin(2 * np.pi * value / period)
        st[f"{name}_cos"] = np.cos(2 * np.pi * value / period)
    for col in POLLUTANTS + ["RAIN"]:
        st[f"log_{col}"] = np.log1p(st[col])
    return st


MODEL_FEATURES = ["log_PM2.5", "log_PM10", "log_SO2", "log_NO2", "log_CO", "log_O3",
                  "TEMP", "PRES", "DEWP", "log_RAIN", "WSPM", "wind_u", "wind_v",
                  "hour_sin", "hour_cos", "month_sin", "month_cos", "dow_sin", "dow_cos"]


def season_of(month):
    return np.select([np.isin(month, [12, 1, 2]), np.isin(month, [3, 4, 5]), np.isin(month, [6, 7, 8])],
                     ["Winter", "Spring", "Summer"], "Autumn")


def heating_flag(dt):
    """Beijing's central heating season runs from 15 November to 15 March."""
    md = dt.dt.month * 100 + dt.dt.day
    return (md >= 1115) | (md <= 315)
