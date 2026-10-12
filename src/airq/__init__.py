"""airq: exploratory analysis and LSTM forecasting of PM2.5 for the Beijing multi-site air-quality data.

Modules
-------
config       paths, constants, thresholds and the monochrome plotting style
data         loading, quality audit, cleaning (caps, Hampel filter, gap interpolation), features
stats_tools  distribution fitting, beta-binomial MLE, ACF / PACF, smoothers, decomposition
viz          exploratory figures
model        windowing, the PyTorch LSTM, training loop, metrics, permutation importance
viz_results  learning curves and test-set diagnostics
"""
