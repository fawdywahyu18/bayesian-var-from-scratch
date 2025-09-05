import numpy as np
from scipy import linalg
from scipy.stats import invwishart, chi2
import matplotlib.pyplot as plt
import pandas as pd

def compute_posterior_parameters(Y_matrix, X_matrix, A_bar, V, S_prior, nu_prior):
    """
    Compute posterior parameters menggunakan persamaan (25)
    
    Returns:
    V_star : posterior covariance
    A_star : posterior mean  
    S_star : posterior scale matrix
    nu_star : posterior degrees of freedom
    """
    T, N = Y_matrix.shape
    
    # V* = (V^-1 + X'X)^-1
    V_inv = linalg.inv(V)
    XtX = X_matrix.T @ X_matrix
    V_star = linalg.inv(V_inv + XtX)
    
    # A* = V* (V^-1 A + X'Y)
    A_star = V_star @ (V_inv @ A_bar + X_matrix.T @ Y_matrix)
    
    # ν* = ν + T  
    nu_star = nu_prior + T
    
    # S* = S + Y'Y + A'V^-1 A - A*'V*^-1 A*
    YtY = Y_matrix.T @ Y_matrix
    term1 = A_bar.T @ V_inv @ A_bar
    term2 = A_star.T @ linalg.inv(V_star) @ A_star
    S_star = S_prior + YtY + term1 - term2
    
    return V_star, A_star, S_star, nu_star

def compute_posterior_moments(A_star, V_star, S_star, nu_star):
    """
    Compute posterior means dan variances
    """
    N = S_star.shape[0]
    
    # Posterior mean
    A_mean = A_star
    Sigma_mean = S_star / (nu_star - N - 1)
    
    # Posterior variance untuk A (diagonal elements saja)
    A_var_diag = np.diag(Sigma_mean)[:, None] @ np.diag(V_star)[None, :]
    
    return A_mean, Sigma_mean, A_var_diag

def compute_credible_intervals(A_star, V_star, S_star, nu_star, alpha=0.05):
    """
    Compute credible intervals untuk posterior parameters
    """
    K, N = A_star.shape
    
    # Expected Sigma
    Sigma_mean = S_star / (nu_star - N - 1)
    
    # Standard errors untuk A
    A_se = np.zeros((K, N))
    for i in range(K):
        for j in range(N):
            A_se[i, j] = np.sqrt(Sigma_mean[j, j] * V_star[i, i])
    
    # t-distribution critical value (approximate)
    t_crit = chi2.ppf(1 - alpha/2, df=nu_star) / nu_star
    t_crit = np.sqrt(t_crit)
    
    # Credible intervals
    A_lower = A_star - t_crit * A_se
    A_upper = A_star + t_crit * A_se
    
    return A_lower, A_upper, A_se

def compare_estimates(A_true_full, A_hat, A_bar, A_star, param_names):
    """
    Compare different estimates
    """
    K, N = A_star.shape
    
    # Create comparison DataFrame
    results = []
    
    for i in range(K):
        for j in range(N):
            results.append({
                'Parameter': f'{param_names[i]}_eq{j+1}',
                'True': A_true_full[i, j],
                'OLS': A_hat[i, j], 
                'Prior': A_bar[i, j],
                'Posterior': A_star[i, j],
                'OLS_Error': A_hat[i, j] - A_true_full[i, j],
                'Post_Error': A_star[i, j] - A_true_full[i, j]
            })
    
    df_compare = pd.DataFrame(results)
    
    return df_compare

def plot_parameter_evolution(A_true_full, A_hat, A_bar, A_star, A_se, param_names):
    """
    Plot parameter evolution: Prior → OLS → Posterior
    """
    K, N = A_star.shape
    
    fig, axes = plt.subplots(N, 1, figsize=(12, 4*N))
    if N == 1:
        axes = [axes]
    
    for eq in range(N):
        x_pos = np.arange(K)
        width = 0.2
        
        # Plot different estimates
        axes[eq].bar(x_pos - 1.5*width, A_true_full[:, eq], width, 
                    label='True', alpha=0.8, color='green')
        axes[eq].bar(x_pos - 0.5*width, A_bar[:, eq], width, 
                    label='Prior', alpha=0.8, color='red')
        axes[eq].bar(x_pos + 0.5*width, A_hat[:, eq], width, 
                    label='OLS', alpha=0.8, color='blue')
        axes[eq].bar(x_pos + 1.5*width, A_star[:, eq], width, 
                    label='Posterior', alpha=0.8, color='purple')
        
        # Add error bars untuk posterior
        axes[eq].errorbar(x_pos + 1.5*width, A_star[:, eq], 
                         yerr=1.96*A_se[:, eq], fmt='none', color='black', alpha=0.5)
        
        axes[eq].set_title(f'Equation {eq+1}: y{eq+1}')
        axes[eq].set_xlabel('Parameters')
        axes[eq].set_ylabel('Coefficient Value')
        axes[eq].set_xticks(x_pos)
        axes[eq].set_xticklabels(param_names, rotation=45)
        axes[eq].legend()
        axes[eq].grid(True, alpha=0.3)
        axes[eq].axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plot_sigma_comparison(Sigma_true, Sigma_expected, S_prior, Sigma_mean):
    """
    Plot Sigma evolution
    """
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    # True Sigma
    im1 = axes[0].imshow(Sigma_true, cmap='RdBu')
    axes[0].set_title('True Σ')
    plt.colorbar(im1, ax=axes[0])
    
    # Prior S
    im2 = axes[1].imshow(S_prior, cmap='RdBu')
    axes[1].set_title('Prior S')
    plt.colorbar(im2, ax=axes[1])
    
    # Likelihood expectation
    im3 = axes[2].imshow(Sigma_expected, cmap='RdBu')
    axes[2].set_title('Likelihood E[Σ]')
    plt.colorbar(im3, ax=axes[2])
    
    # Posterior mean
    im4 = axes[3].imshow(Sigma_mean, cmap='RdBu')
    axes[3].set_title('Posterior E[Σ]')
    plt.colorbar(im4, ax=axes[3])
    
    plt.tight_layout()
    plt.show()

# =============================================================================
# IMPLEMENTASI POSTERIOR COMPUTATION
# =============================================================================

print("Computing Bayesian VAR Posterior Distribution")
print("=" * 55)

# Compute posterior parameters (persamaan 25)
V_star, A_star, S_star, nu_star = compute_posterior_parameters(
    Y_matrix, X_matrix, A_bar, V, S_prior, nu
)

print("Posterior Parameters Computed:")
print(f"V* shape: {V_star.shape}")
print(f"A* shape: {A_star.shape}")  
print(f"S* shape: {S_star.shape}")
print(f"ν*: {nu_star}")

# Compute posterior moments
A_mean, Sigma_mean, A_var_diag = compute_posterior_moments(A_star, V_star, S_star, nu_star)

print(f"\nPosterior Means:")
print(f"E[A|Y]:")
print(A_mean)
print(f"E[Σ|Y]:")
print(Sigma_mean)

# Compute credible intervals
A_lower, A_upper, A_se = compute_credible_intervals(A_star, V_star, S_star, nu_star)

print(f"\nPosterior Standard Errors:")
print(A_se)

# Parameter names untuk plotting
param_names = ['Const'] + [f'A{l}[{i}]' for l in range(1, p+1) for i in range(1, N+1)]

# Comparison table
df_comparison = compare_estimates(A_true_full, A_hat, A_bar, A_star, param_names)

print(f"\nParameter Comparison Summary:")
print("=" * 40)
print(f"Mean Absolute Error (OLS):       {df_comparison['OLS_Error'].abs().mean():.4f}")
print(f"Mean Absolute Error (Posterior): {df_comparison['Post_Error'].abs().mean():.4f}")
print(f"Max Absolute Error (OLS):        {df_comparison['OLS_Error'].abs().max():.4f}")
print(f"Max Absolute Error (Posterior):  {df_comparison['Post_Error'].abs().max():.4f}")

# Detailed comparison
print(f"\nDetailed Parameter Comparison:")
print(df_comparison.round(4))

# Visualizations
print(f"\nGenerating Visualizations...")

# Parameter evolution plot
plot_parameter_evolution(A_true_full, A_hat, A_bar, A_star, A_se, param_names)

# Sigma comparison plot  
plot_sigma_comparison(Sigma_true, Sigma_expected, S_prior, Sigma_mean)

# Summary statistics
print(f"\n" + "=" * 55)
print("BAYESIAN VAR POSTERIOR ANALYSIS SUMMARY")
print("=" * 55)

print(f"Data: N={N} variables, T={T} observations, p={p} lags")
print(f"Prior: Minnesota with λ₀={lambda_0}, λ₁={lambda_1}")
print(f"Posterior: Normal-Wishart with ν*={nu_star}")

# Check if posterior improved over OLS
ols_mae = df_comparison['OLS_Error'].abs().mean()
post_mae = df_comparison['Post_Error'].abs().mean()
improvement = (ols_mae - post_mae) / ols_mae * 100

print(f"\nAccuracy Comparison:")
print(f"OLS MAE:       {ols_mae:.4f}")
print(f"Posterior MAE: {post_mae:.4f}")
print(f"Improvement:   {improvement:.1f}%")

# Prior influence
prior_influence = np.mean(np.abs(A_star - A_hat) / (np.abs(A_hat) + 1e-8))
print(f"Prior Influence: {prior_influence:.1%} deviation from OLS")

print("=" * 55)
print("Bayesian VAR estimation completed successfully!")
print("Results show posterior combines prior beliefs with data evidence.")
print("=" * 55)