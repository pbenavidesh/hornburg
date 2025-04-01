import numpy as np
from statsmodels.tsa.seasonal import STL
from arch.bootstrap import MovingBlockBootstrap

def block_bootstrapping(y_train, seasonal_jump=12, block_size=3, n_bootstraps=10):
    """
    Performs bootstrapping on the training data.

    Args:
        y_train (np.ndarray): The training data.
        seasonal_jump (int): The seasonal jump parameter for STL decomposition.
        block_size (int): The block size for Moving Block Bootstrap.
        n_bootstraps (int): The number of bootstrap samples to generate.

    Returns:
        np.ndarray: An array of bootstrapped y_train samples.
    """

    stl = STL(y_train, seasonal_jump=seasonal_jump)
    res = stl.fit()

    mbb = MovingBlockBootstrap(block_size, res.resid)

    bootstrap_samples = [data for data in mbb.bootstrap(n_bootstraps)]

    # Extract the bootstrap samples into an array
    y_train_b = np.array([sample[0][0].to_numpy() + res.trend + res.seasonal for sample in bootstrap_samples])
    
    return y_train_b
