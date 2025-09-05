# Simulating data

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seed
np.random.seed(42)

def generate_var_coefficients(N, p, persistence=0.7, cross_effects=0.1):
    """
    Generate VAR coefficients A1, A2, ..., Ap
    
    Parameters:
    N : jumlah variabel
    p : jumlah lag
    persistence : diagonal elements A1 (autoregressive effect)
    cross_effects : off-diagonal elements
    """
    A = np.zeros((N, N, p))
    
    # A1: persistence di diagonal, cross-effects di off-diagonal
    A[:, :, 0] = persistence * np.eye(N)
    np.fill_diagonal(A[:, :, 0], persistence)
    
    # Add cross effects
    for i in range(N):
        for j in range(N):
            if i != j:
                A[i, j, 0] = np.random.uniform(-cross_effects, cross_effects)
    
    # A2, A3, ..., Ap dengan efek yang menurun
    for lag in range(1, p):
        decay = 0.3 / lag  # Efek menurun
        A[:, :, lag] = np.random.uniform(-decay, decay, (N, N))
    
    return A

def generate_covariance_matrix(N, correlation=0.3):
    """Generate positive definite covariance matrix"""
    if N == 1:
        return np.array([[1.0]])
    
    # Generate random correlation
    R = np.random.randn(N, N)
    R = R @ R.T
    
    # Normalize to correlation matrix
    d = np.sqrt(np.diag(R))
    R = R / np.outer(d, d)
    
    # Apply correlation strength
    Sigma = (1 - correlation) * np.eye(N) + correlation * R
    
    return Sigma

def simulate_var_data(N, p, T, mu=None, A=None, Sigma=None, burn_in=100):
    """
    Simulate VAR(p) data
    
    Parameters:
    N : jumlah variabel
    p : jumlah lag  
    T : sample size
    mu : konstanta (N x 1)
    A : AR coefficients (N x N x p)
    Sigma : error covariance (N x N)
    burn_in : observasi awal yang dibuang
    """
    # Set default parameters
    if mu is None:
        mu = np.random.normal(0, 0.1, N)
    
    if A is None:
        A = generate_var_coefficients(N, p)
    
    if Sigma is None:
        Sigma = generate_covariance_matrix(N)
    
    # Total sample dengan burn-in
    total_T = T + burn_in + p
    Y = np.zeros((total_T, N))
    
    # Initial conditions
    for t in range(p):
        Y[t, :] = np.random.multivariate_normal(mu, Sigma)
    
    # Generate VAR process
    for t in range(p, total_T):
        # AR component
        ar_part = np.zeros(N)
        for lag in range(p):
            ar_part += A[:, :, lag] @ Y[t-1-lag, :]
        
        # Mean + AR + Error
        mean_t = mu + ar_part
        error_t = np.random.multivariate_normal(np.zeros(N), Sigma)
        Y[t, :] = mean_t + error_t
    
    # Remove burn-in
    Y_final = Y[burn_in:, :]
    
    return Y_final, mu, A, Sigma

def create_design_matrices(Y, p):
    """
    Create design matrices for VAR estimation
    
    Returns:
    Y_matrix : (T-p) x N dependent variables
    X_matrix : (T-p) x K regressors [1, y_{t-1}, ..., y_{t-p}]
    """
    T, N = Y.shape
    T_effective = T - p
    K = 1 + p * N  # konstanta + p lags * N variables
    
    # Y matrix (dependent variables)
    Y_matrix = Y[p:, :]
    
    # X matrix (regressors)
    X_matrix = np.zeros((T_effective, K))
    
    for t in range(T_effective):
        col_idx = 0
        
        # Konstanta
        X_matrix[t, col_idx] = 1
        col_idx += 1
        
        # Lagged variables
        for lag in range(p):
            Y_lag = Y[p + t - 1 - lag, :]  # y_{t-1-lag}
            X_matrix[t, col_idx:col_idx+N] = Y_lag
            col_idx += N
    
    return Y_matrix, X_matrix

def plot_var_data(Y, title="Simulated VAR Data"):
    """Plot time series data"""
    T, N = Y.shape
    
    fig, axes = plt.subplots(N, 1, figsize=(12, 3*N))
    if N == 1:
        axes = [axes]
    
    for i in range(N):
        axes[i].plot(Y[:, i], linewidth=1)
        axes[i].set_title(f'Variable {i+1}')
        axes[i].grid(True, alpha=0.3)
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def print_var_info(Y, mu, A, Sigma):
    """Print VAR model information"""
    T, N = Y.shape
    p = A.shape[2]
    
    print("="*50)
    print("VAR MODEL INFORMATION")
    print("="*50)
    print(f"Dimensions: N={N} variables, p={p} lags, T={T} observations")
    print(f"\nConstants (μ):")
    print(mu)
    print(f"\nAR Coefficients A1:")
    print(A[:, :, 0])
    if p > 1:
        print(f"\nAR Coefficients A2:")
        print(A[:, :, 1])
    print(f"\nError Covariance (Σ):")
    print(Sigma)
    print(f"\nData Summary:")
    print(f"Mean: {np.mean(Y, axis=0)}")
    print(f"Std:  {np.std(Y, axis=0)}")

# =============================================================================
# IMPLEMENTASI
# =============================================================================

print("Simulating VAR Data for Bayesian Estimation")
print("=" * 50)

# Parameter simulasi
N = 3      # 3 variabel
p = 2      # VAR(2)  
T = 1000    # 200 observasi

# Simulate data
Y, mu_true, A_true, Sigma_true = simulate_var_data(N, p, T)

# Print info
print_var_info(Y, mu_true, A_true, Sigma_true)

# Create design matrices untuk estimasi
Y_matrix, X_matrix = create_design_matrices(Y, p)

print(f"\nDesign Matrices:")
print(f"Y_matrix shape: {Y_matrix.shape}")  
print(f"X_matrix shape: {X_matrix.shape}")
print(f"K (total regressors): {X_matrix.shape[1]}")

# Plot data
plot_var_data(Y, "Simulated VAR(2) Data with N=3 Variables")

# Create DataFrame untuk analisis lebih lanjut
df = pd.DataFrame(Y, columns=[f'y{i+1}' for i in range(N)])
df.index.name = 'time'

print(f"\nDataFrame created:")
print(df.head())
print(f"\nDataFrame info:")
print(df.describe())

print("\n" + "="*50)
print("Data simulation completed!")
print("Variables available:")
print("- Y: simulated data (T x N)")  
print("- Y_matrix: dependent vars for estimation")
print("- X_matrix: regressors for estimation")
print("- mu_true, A_true, Sigma_true: true parameters")
print("- df: pandas DataFrame of the data")
print("="*50)