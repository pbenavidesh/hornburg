# %%

import numpy as np
import cvxpy as cp
from sklearn.utils.validation import check_X_y
from sklearn.metrics.pairwise import pairwise_kernels

class svr_original_l1:
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
        K = pairwise_kernels(X, metric=self.kernel, filter_params=True, **self.kwds)
        self.K = K
        R = np.zeros((m, m))
        self.G = self.K + R
        onev = np.ones((m, 1))
        
        return onev, m
    
    def solve(self, X, y):
            
            onev, m = self.param(X)
            
            # variables
            self.alpha = cp.Variable((m, 1), name="alpha")
            self.alpha_ = cp.Variable((m, 1), name="alpha_")
    
            # Problem
            objective = cp.Minimize(
                cp.quad_form((self.alpha - self.alpha_), cp.psd_wrap(self.G))
                + 2* self.epsilon * (self.alpha + self.alpha_).T @ onev
                - 2 * (self.alpha - self.alpha_).T @ y
            )
            
            
            self.constraints = [
                self.alpha >= 0,
                self.alpha_ >= 0,
                self.alpha <= self.C,
                self.alpha_ <= self.C,
                (self.alpha - self.alpha_).T @ onev == 0
            ]
            
            problem = cp.Problem(objective, self.constraints)
            
            # solve problem
            problem.solve(verbose=self.verbose)
            self.status = problem.status
            if self.status != "optimal":
                return self
            self.objective = problem.value
            self.beta = self.alpha.value - self.alpha_.value
            
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
        except:
            self.status = "infeasible"
        
        if self.status != "optimal":
            self.non_optimal(y)
            return self
        
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
            epsilon_beta = np.where(self.support_beta >= 0, -self.epsilon, self.epsilon)
            support_kernel = pairwise_kernels(self.support_vectors, metric=self.kernel, filter_params=True, **self.kwds)
            
            # Identify bounded and unbounded support vectors
            if self.C < 1e3:
                num_eps = 1e-5
                bounded_sv = (np.abs(self.support_beta - self.C) < num_eps) | (np.abs(self.support_beta + self.C) < num_eps)
            else:
                num_eps = (self.C / 1e5)
                bounded_sv = (np.abs(self.support_beta - self.C) < num_eps) | (np.abs(self.support_beta + self.C) < num_eps)
            unbounded_sv = ~bounded_sv
            unbounded_sv = unbounded_sv.flatten()
            self.unbounded_sv = unbounded_sv
            if np.any(unbounded_sv):
                self.b = np.mean(
                    epsilon_beta[unbounded_sv] - self.support_beta.T @ support_kernel[:, unbounded_sv] + self.support_labels[unbounded_sv]
                )
            else:
                self.b = np.mean(
                    epsilon_beta - self.support_beta.T @ support_kernel + self.support_labels
                )

        else:
            self.non_optimal(y)
            
        return self
    
    def predict(self, X):
        return self.decision_function(X)
    
    def decision_function(self, X):
        if self.sup_num == 0:
            return np.ones(X.shape[0]) * self.b
        K = pairwise_kernels(self.support_vectors, X, metric=self.kernel, filter_params=True, **self.kwds)
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
    from sklearn.datasets import make_regression
    from sklearn.svm import SVR
    import matplotlib.pyplot as plt
    
    # Create a regression dataset
    X, y = make_regression(
        n_samples=100,
        n_features=1, 
        noise=20,
        random_state=1
    )
    
    C=1e5
    epsilon=1e-2
    gamma=100
    
    model = svr_original_l1(
            C = C,
            epsilon = epsilon,
            kernel = "rbf",
            gamma=gamma,
        )
    
    model_svr = SVR(
            C = C,
            epsilon = epsilon,
            kernel = "rbf",
            gamma=gamma
    )
    
    model.fit(X, y)
    model_svr.fit(X, y)
    
    y_pred = model.predict(X)
    y_pred_svr = model_svr.predict(X)
    
    plt.title("Support vectors")
    plt.scatter(X, y)
    plt.scatter(model.support_vectors, model.support_labels, color="red")
    plt.scatter(X[model_svr.support_], y[model_svr.support_], color="green", marker=".")
    plt.show()
    
    print("b from model_svr:", model_svr.intercept_[0])
    print("b from moodel:", model.b)
    

    plt.scatter(X, y)
    plt.scatter(X, y_pred, marker=".")
    plt.scatter(X, y_pred_svr, marker=".")
    plt.title("Predictions")
    plt.show()
# %%
