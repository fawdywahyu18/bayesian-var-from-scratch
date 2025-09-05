import numpy as np
from scipy import linalg
from scipy.stats import multivariate_normal, invwishart
import pandas as pd

def ols_estimator(Y_matrix, X_matrix):
    """
    OLS estimator untuk VAR
    A_hat = (X'X)^-1 X'Y
    """
    XtX_inv = linalg.inv(X_matrix.T @ X_matrix)
    A_hat = XtX_inv @ X_matrix.T @ Y_matrix
    
    return A_hat, XtX_inv

def compute_residuals(Y_matrix, X_matrix, A_hat):
    """
    Compute residuals: U = Y - XA
    """
    U = Y_matrix - X_matrix @ A_hat
    return U

def likelihood_normal_wishart_form(Y_matrix, X_matrix):
    """
    Likelihood function dalam bentuk Normal-Wishart (Persamaan 16)
    
    Returns:
    A_hat : OLS estimator koefisien
    XtX_inv : (X'X)^-1 untuk covariance A
    S_hat : residual sum of squares matrix
    df : degrees of freedom
    """
    T, N = Y_matrix.shape
    T_eff, K = X_matrix.shape
    
    # OLS estimator
    A_hat, XtX_inv = ols_estimator(Y_matrix, X_matrix)
    
    # Residuals
    U = compute_residuals(Y_matrix, X_matrix, A_hat)
    
    # Residual sum of squares matrix
    S_hat = U.T @ U
    
    # Degrees of freedom
    df = T - K - N - 1
    
    print("Likelihood Function - Normal-Wishart Form")
    print("="*45)
    print(f"A|Y,Σ ~ MN_{K}×{N}(A_hat, Σ, (X'X)^-1)")
    print(f"Σ|Y ~ IW_{N}(S_hat, {df})")
    print(f"\nA_hat shape: {A_hat.shape}")
    print(f"(X'X)^-1 shape: {XtX_inv.shape}")
    print(f"S_hat shape: {S_hat.shape}")
    print(f"Degrees of freedom: {df}")
    
    return A_hat, XtX_inv, S_hat, df

def log_likelihood_value(Y_matrix, X_matrix, A, Sigma):
    """
    Compute log-likelihood value untuk parameter A dan Sigma tertentu
    Berdasarkan persamaan (14)
    
    L(A,Σ;Y) ∝ |Σ|^(-T/2) * exp(-1/2 * tr[Σ^-1(Y-XA)'(Y-XA)])
    """
    T, N = Y_matrix.shape
    
    # Residuals
    U = Y_matrix - X_matrix @ A
    
    # Log-determinant term
    sign, logdet_Sigma = np.linalg.slogdet(Sigma)
    log_det_term = -0.5 * T * logdet_Sigma
    
    # Quadratic form
    Sigma_inv = linalg.inv(Sigma)
    quad_form = np.trace(Sigma_inv @ U.T @ U)
    quad_term = -0.5 * quad_form
    
    # Constant term (dapat diabaikan untuk optimization)
    const_term = -0.5 * T * N * np.log(2 * np.pi)
    
    log_likelihood = const_term + log_det_term + quad_term
    
    return log_likelihood

def likelihood_at_ols(Y_matrix, X_matrix):
    """
    Evaluate likelihood pada OLS estimates
    """
    T, N = Y_matrix.shape
    
    # OLS estimates
    A_hat, _ = ols_estimator(Y_matrix, X_matrix)
    U = compute_residuals(Y_matrix, X_matrix, A_hat)
    Sigma_hat = (U.T @ U) / T  # ML estimator
    
    # Log-likelihood
    ll = log_likelihood_value(Y_matrix, X_matrix, A_hat, Sigma_hat)
    
    print(f"Likelihood evaluated at OLS estimates:")
    print(f"Log-likelihood: {ll:.2f}")
    
    return ll, A_hat, Sigma_hat

def matric_variate_normal_pdf(A, M, Sigma, V_inv):
    """
    Matric-variate normal PDF untuk A|Y,Σ
    Persamaan: MN(M, Σ, V)
    """
    K, N = A.shape
    
    # Log-determinant terms
    sign_Sigma, logdet_Sigma = np.linalg.slogdet(Sigma)
    sign_V, logdet_V = np.linalg.slogdet(linalg.inv(V_inv))
    
    log_det_term = -0.5 * K * logdet_Sigma - 0.5 * N * logdet_V
    
    # Quadratic form
    diff = A - M
    Sigma_inv = linalg.inv(Sigma)
    quad_form = np.trace(Sigma_inv @ diff.T @ V_inv @ diff)
    quad_term = -0.5 * quad_form
    
    # Constant
    const_term = -0.5 * K * N * np.log(2 * np.pi)
    
    log_pdf = const_term + log_det_term + quad_term
    
    return log_pdf

def inverse_wishart_pdf(Sigma, S, df):
    """
    Inverse Wishart PDF untuk Σ|Y
    """
    N = Sigma.shape[0]
    
    # Log-determinant terms
    sign_Sigma, logdet_Sigma = np.linalg.slogdet(Sigma)
    sign_S, logdet_S = np.linalg.slogdet(S)
    
    log_det_term = -0.5 * (df + N + 1) * logdet_Sigma + 0.5 * df * logdet_S
    
    # Trace term
    Sigma_inv = linalg.inv(Sigma)
    trace_term = -0.5 * np.trace(Sigma_inv @ S)
    
    # Normalization constant (dapat diabaikan)
    log_pdf = log_det_term + trace_term
    
    return log_pdf

def reshape_true_parameters(mu_true, A_true, N, p):
    """
    Reshape true parameters dari simulasi ke format yang sesuai X_matrix
    
    Parameters:
    mu_true : (N,) konstanta
    A_true : (N, N, p) AR coefficients  
    N : jumlah variabel
    p : jumlah lag
    
    Returns:
    A_true_full : (K, N) dimana K = 1 + p*N
                  Format: [μ, A₁, A₂, ..., Aₚ] stacked by rows
    """
    K = 1 + p * N
    A_true_full = np.zeros((K, N))
    
    # Row 0: konstanta μ
    A_true_full[0, :] = mu_true
    
    # Rows untuk setiap lag
    for lag in range(p):
        start_row = 1 + lag * N
        end_row = 1 + (lag + 1) * N
        A_true_full[start_row:end_row, :] = A_true[:, :, lag]
    
    return A_true_full

# =============================================================================
# IMPLEMENTASI MENGGUNAKAN DATA SIMULASI
# =============================================================================

# Menggunakan data dari simulasi sebelumnya
# Pastikan Y_matrix dan X_matrix sudah ada dari code sebelumnya

print("VAR Likelihood Function Analysis")
print("="*50)

# 1. Likelihood dalam bentuk Normal-Wishart
A_hat, XtX_inv, S_hat, df = likelihood_normal_wishart_form(Y_matrix, X_matrix)

print(f"\nOLS Estimator A_hat:")
print(A_hat)

print(f"\nResidual Sum of Squares S_hat:")
print(S_hat)

# 2. Expected value dari Sigma dalam likelihood
Sigma_expected = S_hat / df
print(f"\nExpected Σ|Y (from likelihood):")
print(Sigma_expected)

print(f"\nTrue Sigma (from simulation):")
print(Sigma_true)

print(f"\nDifference:")
print(Sigma_expected - Sigma_true)

# 3. Likelihood value pada OLS estimates
ll_ols, A_ols, Sigma_ols = likelihood_at_ols(Y_matrix, X_matrix)

# 4. Reshape true parameters dan likelihood value pada true parameters
A_true_full = reshape_true_parameters(mu_true, A_true, N, p)
ll_true = log_likelihood_value(Y_matrix, X_matrix, A_true_full, Sigma_true)

print(f"\nParameter Comparison:")
print(f"A_true_full shape: {A_true_full.shape}")
print(f"A_hat shape: {A_hat.shape}")

print(f"\nLikelihood Comparison:")
print(f"Log-likelihood at OLS:  {ll_ols:.2f}")
print(f"Log-likelihood at TRUE: {ll_true:.2f}")
print(f"Difference:            {ll_ols - ll_true:.2f}")

# 5. Matric-variate normal PDF untuk A
log_mvn = matric_variate_normal_pdf(A_hat, A_hat, Sigma_expected, 
                                   linalg.inv(XtX_inv))
print(f"\nMatric-variate Normal log-PDF (A at mode): {log_mvn:.2f}")

# 6. Inverse Wishart PDF untuk Sigma  
log_iw = inverse_wishart_pdf(Sigma_expected, S_hat, df)
print(f"Inverse Wishart log-PDF (Σ at mode): {log_iw:.2f}")

print("\n" + "="*50)
print("Likelihood Function Analysis Completed!")
print("Key Results:")
print(f"- OLS performs {'better' if ll_ols > ll_true else 'worse'} than true parameters")
print(f"- Likelihood decomposed into Normal-Wishart form")
print(f"- Ready for Bayesian inference!")
print("="*50)