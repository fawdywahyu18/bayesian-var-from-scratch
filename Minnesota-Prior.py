import numpy as np
from scipy import linalg
from scipy.stats import invwishart
import matplotlib.pyplot as plt

def estimate_univariate_ar_variances(Y, p=1):
    """
    Estimate univariate AR error variances untuk Minnesota prior
    
    Parameters:
    Y : (T, N) data matrix
    p : lag order untuk univariate AR
    
    Returns:
    sigma_hat_squared : (N,) array of error variances
    """
    T, N = Y.shape
    sigma_hat_squared = np.zeros(N)
    
    for i in range(N):
        y = Y[:, i]
        
        # Create lagged data untuk univariate AR(p)
        y_dep = y[p:]
        X_ar = np.column_stack([np.ones(len(y_dep))] + 
                              [y[p-j:-j] for j in range(1, p+1)])
        
        # OLS regression
        beta = linalg.inv(X_ar.T @ X_ar) @ X_ar.T @ y_dep
        residuals = y_dep - X_ar @ beta
        
        # Error variance
        sigma_hat_squared[i] = np.var(residuals, ddof=p+1)
    
    return sigma_hat_squared

def create_minnesota_prior_V(N, p, lambda_0, lambda_1, sigma_hat_squared):
    """
    Create Minnesota prior covariance matrix V
    
    Parameters:
    N : jumlah variabel
    p : lag order
    lambda_0 : shrinkage parameter untuk konstanta
    lambda_1 : shrinkage parameter untuk AR coefficients  
    sigma_hat_squared : (N,) univariate AR error variances
    
    Returns:
    V : (K, K) prior covariance matrix (diagonal)
    """
    K = 1 + p * N
    V = np.zeros((K, K))
    
    # Konstanta (first element)
    V[0, 0] = lambda_0
    
    # AR coefficients
    idx = 1
    for lag in range(1, p + 1):  # lag = 1, 2, ..., p
        for var in range(N):     # variable = 1, 2, ..., N
            V[idx, idx] = lambda_1 / (lag**2 * sigma_hat_squared[var])
            idx += 1
    
    return V

def create_minnesota_prior_mean(N, p):
    """
    Create Minnesota prior mean A_bar (random walk assumption)
    
    Returns:
    A_bar : (K, N) prior mean matrix
    """
    K = 1 + p * N
    A_bar = np.zeros((K, N))
    
    # Random walk: y_t = y_{t-1} + u_t
    # Konstanta = 0, A1 = I_N, A2 = ... = Ap = 0
    
    # A1 diagonal = 1 (unit root assumption)
    for i in range(N):
        A_bar[1 + i, i] = 1.0  # A1[i,i] = 1
    
    # A2, A3, ..., Ap = 0 (already zero-initialized)
    
    return A_bar

def minnesota_prior_parameters(Y, p, lambda_0=1.0, lambda_1=0.2, nu=None, S_scale=1.0):
    """
    Set up complete Minnesota prior parameters
    
    Parameters:
    Y : (T, N) data
    p : lag order
    lambda_0 : shrinkage untuk konstanta
    lambda_1 : shrinkage untuk AR coefficients
    nu : degrees of freedom untuk Inverse Wishart (default: N+2)
    S_scale : scaling factor untuk S matrix
    
    Returns:
    A_bar : prior mean
    V : prior covariance  
    S : scale matrix untuk Sigma
    nu : degrees of freedom
    """
    T, N = Y.shape
    
    # 1. Univariate AR variances
    sigma_hat_squared = estimate_univariate_ar_variances(Y, p=1)
    
    # 2. Prior mean (random walk)
    A_bar = create_minnesota_prior_mean(N, p)
    
    # 3. Prior covariance matrix V
    V = create_minnesota_prior_V(N, p, lambda_0, lambda_1, sigma_hat_squared)
    
    # 4. Prior for Sigma (Inverse Wishart)
    if nu is None:
        nu = N + 2  # Slightly informative
    
    # S matrix: diagonal dengan univariate variances
    S = S_scale * np.diag(sigma_hat_squared)
    
    print("Minnesota Prior Setup:")
    print("=" * 40)
    print(f"λ₀ (konstanta shrinkage): {lambda_0}")
    print(f"λ₁ (AR shrinkage): {lambda_1}")
    print(f"ν (degrees of freedom): {nu}")
    print(f"Univariate AR variances: {sigma_hat_squared}")
    print(f"A_bar shape: {A_bar.shape}")
    print(f"V shape: {V.shape}")
    print(f"S shape: {S.shape}")
    
    return A_bar, V, S, nu, sigma_hat_squared

def evaluate_minnesota_prior_pdf(A, Sigma, A_bar, V, S, nu):
    """
    Evaluate Minnesota prior PDF sesuai persamaan (19)
    
    p(A,Σ) ∝ |Σ|^(-(ν+N+K+1)/2) * exp(-1/2 * tr[Σ^(-1)(A-A_bar)'V^(-1)(A-A_bar)])
                                  * exp(-1/2 * tr[Σ^(-1)S])
    """
    K, N = A.shape
    
    # Log determinant Sigma
    sign, logdet_Sigma = np.linalg.slogdet(Sigma)
    log_det_term = -0.5 * (nu + N + K + 1) * logdet_Sigma
    
    # Quadratic form untuk A
    A_diff = A - A_bar
    V_inv = linalg.inv(V)
    Sigma_inv = linalg.inv(Sigma)
    
    quad_A = np.trace(Sigma_inv @ A_diff.T @ V_inv @ A_diff)
    quad_A_term = -0.5 * quad_A
    
    # Trace term untuk S
    trace_S = np.trace(Sigma_inv @ S)
    trace_S_term = -0.5 * trace_S
    
    log_prior = log_det_term + quad_A_term + trace_S_term
    
    return log_prior

def plot_minnesota_prior_structure(V, A_bar, sigma_hat_squared, N, p):
    """Plot struktur Minnesota prior"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # 1. Prior covariance matrix V (diagonal)
    axes[0].semilogy(np.diag(V), 'bo-', markersize=4)
    axes[0].set_title('Prior Variances (log scale)')
    axes[0].set_xlabel('Parameter Index')
    axes[0].set_ylabel('Prior Variance')
    axes[0].grid(True, alpha=0.3)
    
    # Add labels
    labels = ['Const'] + [f'A{l}[{i}]' for l in range(1,p+1) for i in range(1,N+1)]
    axes[0].set_xticks(range(len(labels)))
    axes[0].set_xticklabels(labels, rotation=45)
    
    # 2. Prior mean A_bar
    vmax = np.max(np.abs(A_bar))
    im = axes[1].imshow(A_bar, cmap='RdBu', vmin=-vmax, vmax=vmax)
    axes[1].set_title('Prior Mean A_bar')
    axes[1].set_xlabel('Variable')
    axes[1].set_ylabel('Parameter')
    plt.colorbar(im, ax=axes[1])
    
    # 3. Univariate AR variances
    axes[2].bar(range(N), sigma_hat_squared)
    axes[2].set_title('Univariate AR Variances')
    axes[2].set_xlabel('Variable')
    axes[2].set_ylabel('Variance')
    axes[2].set_xticks(range(N))
    axes[2].set_xticklabels([f'y{i+1}' for i in range(N)])
    
    plt.tight_layout()
    plt.show()

# =============================================================================
# IMPLEMENTASI MINNESOTA PRIOR
# =============================================================================

print("Setting up Minnesota Prior Distribution")
print("=" * 50)

# Hyperparameters
lambda_0 = 1.0    # Shrinkage untuk konstanta
lambda_1 = 0.2    # Shrinkage untuk AR coefficients
nu_prior = N + 2  # Degrees of freedom

# Setup Minnesota prior
A_bar, V, S_prior, nu, sigma_hat_sq = minnesota_prior_parameters(
    Y, p, lambda_0, lambda_1, nu_prior
)

print(f"\nPrior Mean A_bar:")
print(A_bar)

print(f"\nPrior Covariance V (diagonal elements):")
print(np.diag(V))

print(f"\nPrior Scale Matrix S:")
print(S_prior)

# Evaluate prior PDF pada true parameters
A_true_full = reshape_true_parameters(mu_true, A_true, N, p)
log_prior_true = evaluate_minnesota_prior_pdf(
    A_true_full, Sigma_true, A_bar, V, S_prior, nu
)

# Evaluate prior PDF pada OLS estimates  
log_prior_ols = evaluate_minnesota_prior_pdf(
    A_hat, Sigma_expected, A_bar, V, S_prior, nu
)

print(f"\nPrior PDF Evaluation:")
print(f"Log-prior at TRUE params: {log_prior_true:.2f}")
print(f"Log-prior at OLS params:  {log_prior_ols:.2f}")
print(f"Difference:              {log_prior_ols - log_prior_true:.2f}")

# Compare dengan random walk assumption
print(f"\nRandom Walk Prior vs True Parameters:")
print(f"Prior mean A_bar[1:4,:] (A1):")
print(A_bar[1:4, :])
print(f"True A1:")
print(A_true[:, :, 0])
print(f"Difference:")
print(A_bar[1:4, :] - A_true[:, :, 0])

# Plot prior structure
plot_minnesota_prior_structure(V, A_bar, sigma_hat_sq, N, p)

print("\n" + "=" * 50)
print("Minnesota Prior Setup Completed!")
print("Key Components:")
print("- A_bar: Random walk prior mean")
print("- V: Shrinkage covariance matrix")  
print("- S_prior: Scale matrix for Σ")
print("- nu: Degrees of freedom")
print("Ready for Bayesian inference!")
print("=" * 50)