# %%

import numpy as np
import cvxpy as cp
from sklearn.utils.validation import check_X_y
from sklearn.metrics.pairwise import pairwise_kernels


class svr_original_mape:
    
    def __init__(self, C=1, epsilon=1, kernel="linear", verbose=False, **kernel_params):
        self.C = C
        self.epsilon = epsilon
        self.verbose = verbose
        self.kernel = kernel
        self.kwds = kernel_params
        
    def check_x_y(self, X, y):
        X, y = check_X_y(X, y)
        
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
        
        return X, y
    
    def param(self, X):
        m = X.shape[0]
        
        # parameters
        if isinstance(self.kernel, str):
            K = pairwise_kernels(X, metric=self.kernel, filter_params=True, **self.kwds)
        else:
            K = self.kernel(X)
        self.K = K
        # R = np.zeros((m, m))
        R = np.eye(m) * 1e-4
        self.G = self.K + R
        onev = np.ones((m, 1))
        
        return onev, m
        
    def solve(self, X, y):
            
        onev, m = self.param(X)
            
        # variables
        self.beta = cp.Variable((m, 1), name="beta")

        # Problem
        objective = cp.Minimize(
            cp.quad_form(self.beta, cp.psd_wrap(self.G)) 
            + 2 * (self.epsilon * y.T / 100) @ cp.abs(self.beta)
            - 2 * y.T @ self.beta
        )
        
        self.constraints = [
            self.beta <= self.C * (100 / y),
            self.beta >= - self.C * (100 / y),
            self.beta.T @ onev == 0,
        ]
        
        problem = cp.Problem(objective, self.constraints)
        
        # solve problem
        problem.solve(verbose=self.verbose, solver=cp.MOSEK)
        self.status = problem.status
        if self.status != "optimal":
            return self
        self.objective = problem.value
        
        return self     

    def non_optimal(self, y):
        self.sup_num = 0
        self.support_vectors = np.array([0])
        self.support_labels = np.array([0])
        self.support_beta = np.array([0])
        self.b = np.mean(y)
        
    def fit(self, X, y):
        
        # check X and y
        X, y = self.check_x_y(X, y)
        try:
            self.solve(X, y)
        except cp.SolverError:
            self.status = "infeasible"
        
        if self.status != "optimal":
            self.non_optimal(y)
            return self
        
        self.beta = self.beta.value
        
        beta_1 = self.beta.flatten()
        # Use different conditions to compute beta_1 due to internal numerical unstability
        if self.C < 1e3:
            indx = np.abs(beta_1) > 1e-5
        else:
            indx = np.abs(beta_1) > (self.C / 1e5)
        if np.sum(indx) != 0:
            self.sup_num = np.sum(indx)
            self.support_vectors = X[indx, :]
            self.support_labels = y[indx, :]
            self.support_beta = self.beta[indx, :]
            
            # calculate epsilon
            epsilon_beta = np.where(self.support_beta >= 0, -self.epsilon/100, self.epsilon/100)
            if isinstance(self.kernel, str):
                support_kernel = pairwise_kernels(self.support_vectors, metric=self.kernel, filter_params=True, **self.kwds)
            else:
                support_kernel = self.kernel(self.support_vectors)
            
            # Identify bounded and unbounded support vectors
            if self.C < 1e3:
                num_eps = 1e-4
                bounded_sv = (np.abs(self.support_beta - (100 * self.C / self.support_labels)) < num_eps) | (np.abs(self.support_beta + (100 * self.C / self.support_labels)) < num_eps)
            else:
                num_eps = (self.C / 1e5)
                bounded_sv = (np.abs(self.support_beta - (100 * self.C / self.support_labels)) < num_eps) | (np.abs(self.support_beta + (100 * self.C / self.support_labels)) < num_eps)
            unbounded_sv = ~bounded_sv
            unbounded_sv = unbounded_sv.flatten()
            self.unbounded_sv = unbounded_sv
            if np.any(unbounded_sv):
                self.b = np.mean(
                    (epsilon_beta[unbounded_sv] * self.support_labels[unbounded_sv]) - self.support_beta.T @ support_kernel[:, unbounded_sv] + self.support_labels[unbounded_sv]
                )
            else:
                self.b = np.mean(
                    (epsilon_beta * self.support_labels) - self.support_beta.T @ support_kernel + self.support_labels
                )

        else:
            self.non_optimal(y)
            
        return self
    
    def predict(self, X):
        return self.decision_function(X)
    
    def decision_function(self, X):
        if self.sup_num == 0:
            return np.ones(X.shape[0]) * self.b
        if isinstance(self.kernel, str):
            K = pairwise_kernels(self.support_vectors, X, metric=self.kernel, filter_params=True, **self.kwds)
        else:
            K = self.kernel(self.support_vectors, X)
        return self.support_beta.T @ K + self.b
    
    def score(self, X, y):
        return np.mean(self.predict(X) == y)
    
    def get_params(self):
        return {"C": self.C, "epsilon": self.epsilon}
    
    def set_params(self, **params):
        self.C = params["C"]
        self.epsilon = params["epsilon"]
        return self

# %%
if __name__ == "__main__":
    from sklearn.metrics import mean_absolute_percentage_error
    from sklearn.datasets import make_regression
    from sklearn.svm import SVR
    import matplotlib.pyplot as plt
    from sklearn.gaussian_process.kernels import RBF
    from sklearn.gaussian_process.kernels import ExpSineSquared
    
    # Create a regression dataset
    X, y = make_regression(
        n_samples=100,
        n_features=1, 
        noise=20,
        random_state=1
    )
    
    y = y + 200  
    
    C=1000
    epsilon=1e-1
    gamma=1e1
    
    
    seasonal_kernel = (
    2.0**2
    # * RBF(length_scale=100.0)
    * ExpSineSquared(length_scale=1.0, periodicity=1.0, periodicity_bounds="fixed")
    )
    
    model_mape = svr_original_mape(
            C = C,
            epsilon = epsilon,
            kernel = RBF(length_scale=.50),
            gamma=gamma,
            verbose=False
        )
    
    model_sklrn = SVR(
            C = C,
            epsilon = epsilon,
            kernel = "rbf",
            gamma=gamma
    )
    
    model_mape.fit(X, y)
    model_sklrn.fit(X, y)
    
    y_mape = model_mape.predict(X)
    y_pred_sklearn = model_sklrn.predict(X)
    
# %%
    
    
    plt.title("Support vectors")
    plt.scatter(X, y)
    plt.scatter(model_mape.support_vectors, model_mape.support_labels, color="red")
    plt.scatter(X[model_sklrn.support_], y[model_sklrn.support_], color="green", marker=".")
    plt.show()
    
    print("b from model_sklrn:", model_sklrn.intercept_[0])
    print("b from moodel:", model_mape.b)
    

    plt.scatter(X, y)
    plt.scatter(X, y_mape, marker="o", label="mape")
    plt.scatter(X, y_pred_sklearn, marker=".", label="sklearn")
    plt.title("Predictions")
    plt.legend()
    plt.show()
    
    support_indices_sklearn = model_sklrn.support_
    dual_coef_full_sklearn = np.zeros_like(y)
    dual_coef_full_sklearn[support_indices_sklearn] = model_sklrn.dual_coef_

    d = np.linspace(-2, 2, y.shape[0])
    plt.scatter(d, dual_coef_full_sklearn, label="sklearn", marker="*")
    plt.scatter(d, model_mape.beta.flatten() * y, label="mape_l1", marker=".", color="red")
    # plt.ylim(-0.0001, 0.0001)
    # plt.ylim(-10, 10)
    plt.legend()
    plt.show()
    
    
# %%

    def print_accuracy(y, y_pred, model):
        print(f"{model} accuracy: {mean_absolute_percentage_error(y, y_pred.flatten())}")

    print_accuracy(y, y_pred_sklearn, "sklearn")
    print_accuracy(y, y_mape, "mape_l1")
    
    
# %%
