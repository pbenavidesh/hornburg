# %% libraries
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

sys.path.append(os.path.join("..", "..", "library"))
from svr_mape import svr_original_mape
from sklearn.svm import SVR

svr_algorithm = "mape"

# %%
parameter_path = os.path.join(f"average_{svr_algorithm}.jsonl")
parameters = []
with jsonlines.open(parameter_path) as reader:
    for obj in reader:
        parameters.append(obj)

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

X = (co2_data["date"].dt.year + co2_data["date"].dt.month / 12).to_numpy().reshape(-1, 1)
y = co2_data["co2"].to_numpy()

X_, y_ = X[67:], y[67:]

# %% kernel
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

# %%

if svr_algorithm == "sklearn":
    svr_model = SVR
elif svr_algorithm == "mape":
    svr_model = svr_original_mape
else:
    raise NameError(f"svr_model is either 'sklearn' or 'mape', got '{svr_algorithm}'")

# %% load parameters
for parameter in parameters:
    
    test_size = int(parameter['test_size'])
    
    # print(f"test_size: {test_size}")
    
    n_splits = int(145 / test_size)
    
    metric_all, prediction_all, indices_all = [], [], []
    tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size)
    for split_number, (train_index, test_index) in enumerate(tscv.split(X_)):
        X_train, X_test = X_[train_index], X_[test_index]
        y_train, y_test = y_[train_index], y_[test_index]
        
        model = svr_model(
            kernel=co2_kernel,
            C=parameter['C'],
            epsilon=parameter['epsilon']
        )
        
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        
        metric = mape(y_test.flatten(), y_pred.flatten())
        
        metric_all.append(metric)
        # print(f"mape {split_number}: {metric:.5f}")
        
        prediction_all.extend(y_pred.flatten())
        indices_all.extend(X_test)
        
    plt.title(f"MAPE test_size {test_size} ({svr_algorithm}) : ({np.mean(metric_all):.6f})")
    plt.plot(X_, y_, alpha=0.5, color="k", linestyle=':', label='true')
    plt.plot(indices_all, prediction_all, label='predict', color='tab:red', alpha=0.5)
    plt.legend()
    plt.show()
    
    print(f"average mape ({test_size}): {np.mean(metric_all)}")
        

# %%
