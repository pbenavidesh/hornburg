# %%
import os
import sys
import jsonlines
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error as mape
from sklearn.datasets import fetch_openml
from sklearn.model_selection import TimeSeriesSplit
from sklearn.gaussian_process.kernels import RBF, ExpSineSquared, RationalQuadratic, WhiteKernel

from statsmodels.tsa.seasonal import STL
from arch.bootstrap import MovingBlockBootstrap

sys.path.append(os.path.join("..", "..", "library"))
from svr_mape import svr_original_mape
from sklearn.svm import SVR

from bootstrap_function import block_bootstrapping

# %%

svr_algorithm = "sklearn"


# %% Obtain data
co2 = fetch_openml(data_id=41187, as_frame=True)

co2_data = (
    co2
    .frame
    [["year", "month", "day", "co2"]]
    .assign(
        date = lambda k: pd.to_datetime(k[['year', 'month', 'day']])
    )
    .sort_values(by="date")
        .groupby(pd.Grouper(key="date", freq="ME"))
        .agg({"co2": "mean"})
        .dropna()
        .reset_index()
)

X_ = (co2_data["date"].dt.year + co2_data["date"].dt.month / 12).to_numpy().reshape(-1, 1)
y_ = co2_data["co2"].to_numpy()

X, y = X_[67:], y_[67:]

co2 = pd.Series(
    y.flatten(), index=pd.date_range("1-1-1964", periods=len(y), freq="MS"), name="CO2"
)

# %% Predict new values
long_term_trend_kernel = 50.0**2 * RBF(length_scale=50.0)
seasonal_kernel = (
    2.0**2
    * RBF(length_scale=100.0)
    * ExpSineSquared(length_scale=1.0, periodicity=1.0, periodicity_bounds="fixed")
)
irregularities_kernel = 0.5**2 * RationalQuadratic(length_scale=1.0, alpha=1.0)
noise_kernel = 0.1**2 * RBF(length_scale=0.1) + WhiteKernel(
    noise_level=0.1**2, noise_level_bounds=(1e-5, 1e5)
)
co2_kernel = (
    long_term_trend_kernel + seasonal_kernel + irregularities_kernel + noise_kernel
)
co2_kernel

# %% set parameters
parameter_path = os.path.join(f"average_{svr_algorithm}.jsonl")
parameters = []
with jsonlines.open(parameter_path) as reader:
    for obj in reader:
        parameters.append(obj)

test_size = 60
params = {i for i in parameters if int(i["test_size"]) == test_size}
params = {k: v for k,v in params.items()}
params['kernel'] = co2_kernel

n_splits = int(145 / test_size)
tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size)

# %%
if svr_algorithm == "sklearn":
    svr_model = SVR
elif svr_algorithm == "mape":
    svr_model = svr_original_mape
else:
    raise NameError(f"svr_model is either 'sklearn' or 'mape', got '{svr_algorithm}'")

# %% split number
metric_all, prediction_all, indices_all = [], [], []
for split_number, (train_index, test_index) in enumerate(tscv.split(X)):
    print(f"Split number {split_number}")
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = co2.iloc[train_index], co2.iloc[test_index]
    
    y_train_b = block_bootstrapping(y_train, seasonal_jump=12, block_size=3, n_bootstraps=30)

    metric_split, prediction_split = [], []
    for y_train in y_train_b: 
        model = svr_model(**params)
        
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        
        metric = mape(y_test.values, y_pred.flatten())
        
        metric_split.append(metric)
        # print(f"mape {split_number}: {metric:.5f}")
        
        prediction_split.append(y_pred.flatten())
    
    metric_all.append(np.mean(metric_split, axis=0))
    prediction_all.extend(np.mean(prediction_split, axis=0))
    indices_all.extend(X_test)
        
plt.title(f"test_size: {test_size}")
plt.plot(X_, y_, alpha=0.5, color="k", linestyle=':', label='true')
plt.plot(indices_all, prediction_all, label='predict', color='tab:red', alpha=0.5)
plt.show()

print(f"average mape: {np.mean(metric_all)}")
# %%
