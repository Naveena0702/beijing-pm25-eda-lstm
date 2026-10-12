"""Project-wide constants: paths, variable groups, thresholds and the plotting style.

Every module imports from here so that a single change (for example the target
station or the forecast horizon) propagates through the whole pipeline.
"""
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
DATA_ZIP = ROOT / "data" / "PRSA2017_Data_20130301-20170228.zip"
FIG_DIR = ROOT / "figures"
RES_DIR = ROOT / "results"
DASH_DIR = ROOT / "dashboard"
for _d in (FIG_DIR, RES_DIR, DASH_DIR):
    _d.mkdir(parents=True, exist_ok=True)

SEED = 2024
TARGET_STATION = "Dongsi"            # central urban site used for the deep learning model

POLLUTANTS = ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3"]
METEO = ["TEMP", "PRES", "DEWP", "RAIN", "WSPM"]
NUMERIC = POLLUTANTS + METEO
UNITS = {"PM2.5": "ug/m3", "PM10": "ug/m3", "SO2": "ug/m3", "NO2": "ug/m3", "CO": "ug/m3",
         "O3": "ug/m3", "TEMP": "deg C", "PRES": "hPa", "DEWP": "deg C", "RAIN": "mm", "WSPM": "m/s"}

# 16-point compass, clockwise from north (used for wind vectors and the polar chart)
WIND_DIRS = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
             "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
NORTHERLY = {"NW", "NNW", "N", "NNE", "WNW"}
SOUTHERLY = {"SE", "SSE", "S", "SSW", "ESE"}

URBAN = ["Dongsi", "Tiantan", "Guanyuan", "Wanshouxigong", "Aotizhongxin", "Nongzhanguan", "Wanliu", "Gucheng"]
SUBURBAN = ["Changping", "Dingling", "Huairou", "Shunyi"]

# China's Ambient Air Quality Standard GB 3095-2012 (Grade II, 24-hour PM2.5) and the
# individual AQI breakpoints of HJ 633-2012, applied here to hourly concentrations.
LIMIT_PM25 = 75.0
WHO_24H = 15.0
AQI_BREAKS = [0, 35, 75, 115, 150, 250, float("inf")]
AQI_LABELS = ["Excellent", "Good", "Light", "Moderate", "Heavy", "Severe"]

# Chronological split by "pollution years" (March to February): 2 years train, 1 year validation, 1 year test,
# so that validation and test both contain every season, including a full heating season.
TRAIN_END = "2015-02-28 23:00"
VAL_END = "2016-02-29 23:00"
HORIZON = 6                          # forecast the next 6 hours
MAX_INTERP_GAP = 24                  # gaps up to 24 h are interpolated, longer gaps stay missing

# Colour-blind-safe categorical theme (validated all-pairs: CVD delta-E >= 9.2), fixed order:
# violet, aqua, orange.  Sequential magnitude uses one-hue ramps (Purples for densities,
# Oranges for pollutant concentration) and polarity uses violet <-> orange around a grey midpoint.
VIOLET, AQUA, ORANGE = "#4a3aa7", "#1baf7a", "#eb6834"
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
NEUTRAL = "#d6d5ce"
STYLE = {
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9.5, "axes.titlesize": 10.5, "axes.titleweight": "bold", "axes.labelsize": 9.5,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8.5, "legend.frameon": False,
    "text.color": INK, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "xtick.color": AXIS, "ytick.color": AXIS, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": AXIS, "axes.linewidth": 0.9,
    "xtick.direction": "out", "ytick.direction": "out", "axes.grid": False,
    "lines.linewidth": 1.8, "lines.solid_capstyle": "round",
    "figure.dpi": 110, "savefig.dpi": 220, "image.cmap": "Purples",
}
# legacy names used by a few helpers
MID, LIGHT = INK2, NEUTRAL


def apply_style():
    from cycler import cycler
    plt.rcParams.update(STYLE)
    plt.rcParams["axes.prop_cycle"] = cycler(color=[VIOLET, AQUA, ORANGE])
