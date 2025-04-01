# %%
import os
import sys
import jsonlines
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error as mape
from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.join("..", "..", "library"))
from svr_mape import svr_original_mape
from sklearn.svm import SVR
from sklearn.utils import resample
import numpy as np

# %%
svr_algorithm = "mape"

# %%
parameter_path = os.path.join(f"average_{svr_algorithm}.jsonl")
parameters = []
with jsonlines.open(parameter_path) as reader:
    for obj in reader:
        parameters.append(obj)

# %%
diabetes = pd.read_csv("diabetes.csv", delimiter="\t")
X_train, X_test, y_train, y_test = train_test_split(diabetes.drop(columns="Y"), diabetes.Y, test_size=0.225, random_state=42)
X_train, y_train = X_train.values, y_train.values
X_test, y_test = X_test.values, y_test.values

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

if svr_algorithm == "sklearn":
    svr_model = SVR
elif svr_algorithm == "mape":
    svr_model = svr_original_mape
else:
    raise NameError(f"svr_model is either 'sklearn' or 'mape', got '{svr_algorithm}'")

# %% predict
errors = []
for params in parameters:
    model = svr_model(
        C=params["C"],
        epsilon=params["epsilon"],
        kernel="rbf",
        gamma=params["gamma"]
    )
    model.fit(X_train, y_train)

    n_iterations = 1000
    n_size = int(len(X_test) * 0.8)
    scores = []
    for i in range(n_iterations):

        idx = resample(np.arange(len(X_test)), n_samples=n_size, replace=False, random_state=i+1)
        X, y = X_test[idx], y_test[idx]

        yhat = model.predict(X).flatten()
        score = mape(y, yhat)
        scores.append(score)

    error = np.array(scores).mean()
    print('MAPE: %.5f' % error)
    errors.append(error)

print(f"MAPE mean ({svr_algorithm}): {np.mean(errors):.5f}")

# %%
