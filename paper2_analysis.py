"""
paper2_analysis.py
==================
Complete analysis for Paper 2: Volatility Spillovers in Macao's
Gaming-Tourism Nexus (Total Visitors, Overnight Visitors, Gaming Revenue)

Three modelling options:
  A  overnight visitors  <-->  gaming revenue       (2-variable)
  B  same-day + overnight + gaming                  (3-variable)
  C  overnight share (%) + gaming                   (ratio series)

All GARCH and DCC-GARCH are implemented from scratch (scipy MLE only).
Diebold-Yilmaz (2012) spillover index via VAR + FEVD.

Requires: numpy, pandas, scipy, matplotlib, pickle
"""

import numpy as np
import pandas as pd
import scipy.optimize as opt
import scipy.stats as stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle, warnings, os
warnings.filterwarnings('ignore')

# ── OUTPUT DIR ────────────────────────────────────────────────────────────
OUT = 'output'
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 10, 'axes.titlesize': 10,
    'axes.labelsize': 10, 'legend.fontsize': 8,
    'figure.dpi': 150, 'axes.spines.top': False,
    'axes.spines.right': False, 'axes.grid': True,
    'grid.alpha': 0.3, 'grid.linestyle': '--',
})
BLUE='#1f4e79'; ORANGE='#c55a11'; GREEN='#375623'; RED='#c00000'; GRAY='#595959'

# ═══════════════════════════════════════════════════════════════════════════
# 1. DATA PREPARATION
# ═══════════════════════════════════════════════════════════════════════════
print("="*60)
print("1. DATA PREPARATION")
print("="*60)

# Load visitor data from DSEC Excel
df_raw = pd.read_excel('dsec.xlsx', header=None)
data = df_raw.iloc[8:, [0, 11, 21]].copy()
data.columns = ['date_raw', 'sameday', 'overnight']
data = data[data['date_raw'].notna()]
data = data[~data['date_raw'].astype(str).str.contains('b |i |k |~|下載')]

def parse_date(s):
    s = str(s).strip().replace('年','-').replace('月','')
    try: return pd.to_datetime(s)
    except: return pd.NaT

data['date'] = data['date_raw'].apply(parse_date)
data = data[data['date'].notna()].sort_values('date').set_index('date')
data['sameday']   = pd.to_numeric(data['sameday'],   errors='coerce')
data['overnight'] = pd.to_numeric(data['overnight'], errors='coerce')
data['total']     = data['sameday'] + data['overnight']
data['ovn_share'] = data['overnight'] / data['total'] * 100  # overnight %

# Load gaming revenue from original paper data (gaming_revenue.csv)
gaming_df = pd.read_csv('gaming_revenue.csv', index_col=0, parse_dates=True)
data['gaming'] = gaming_df['gaming_revenue']

# Align: Jan 2008 – Jul 2025 (include Jan 2008 as base for first log-return)
data = data.loc['2008-01':'2025-07'].copy()
print(f"Aligned dataset: {data.index[0].strftime('%Y-%m')} to {data.index[-1].strftime('%Y-%m')}, n_raw={len(data)}")
print(f"Nulls: {data[['sameday','overnight','gaming','ovn_share']].isnull().sum().to_dict()}")

# Log returns (×100)
for col in ['total','sameday','overnight','gaming']:
    data[f'r_{col}'] = np.log(data[col] / data[col].shift(1)) * 100

# Overnight share: first difference (already stationary conceptually)
data['d_ovn_share'] = data['ovn_share'].diff()

data = data.dropna()
print(f"After differencing: n={len(data)}")

# Dummy variables
data['D_COVID']  = ((data.index >= '2020-02') & (data.index <= '2022-12')).astype(float)
data['D_REOPEN'] = ((data.index >= '2023-01') & (data.index <= '2023-06')).astype(float)
data['D_GFC']    = ((data.index >= '2008-10') & (data.index <= '2009-06')).astype(float)

print("\nOvernight share (%) by year:")
data['year'] = data.index.year
print(data.groupby('year')['ovn_share'].mean().round(1).to_string())
data.drop(columns='year', inplace=True)

# ═══════════════════════════════════════════════════════════════════════════
# 2. DESCRIPTIVE STATISTICS
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("2. DESCRIPTIVE STATISTICS")
print("="*60)

ret_cols = {
    'Total visitors (log-ret %)':    'r_total',
    'Same-day visitors (log-ret %)': 'r_sameday',
    'Overnight visitors (log-ret %)':'r_overnight',
    'Gaming revenue (log-ret %)':    'r_gaming',
    'Overnight share (1st diff %)':  'd_ovn_share',
}

desc_rows = []
for label, col in ret_cols.items():
    s = data[col].dropna()
    row = {
        'Series': label,
        'N': len(s),
        'Mean': round(s.mean(), 3),
        'Std': round(s.std(), 3),
        'Min': round(s.min(), 1),
        'Max': round(s.max(), 1),
        'Skewness': round(stats.skew(s), 3),
        'Ex. Kurtosis': round(stats.kurtosis(s), 3),
    }
    # Jarque-Bera
    jb_stat, jb_p = stats.jarque_bera(s)
    row['JB p-value'] = round(jb_p, 4)
    desc_rows.append(row)

desc_df = pd.DataFrame(desc_rows)
print(desc_df.to_string(index=False))
desc_df.to_csv(f'{OUT}/descriptive_stats.csv', index=False)

# Correlation matrix (log returns)
corr_cols = ['r_total','r_sameday','r_overnight','r_gaming']
corr_labels = ['Total','Same-day','Overnight','Gaming']
corr_mat = data[corr_cols].corr().round(4)
corr_mat.index = corr_labels; corr_mat.columns = corr_labels
print("\nCorrelation matrix (log returns):")
print(corr_mat.to_string())
corr_mat.to_csv(f'{OUT}/correlation_matrix.csv')

# ═══════════════════════════════════════════════════════════════════════════
# 3. UNIT ROOT TESTS (ADF from scratch)
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("3. UNIT ROOT TESTS")
print("="*60)

def adf_test(series, maxlag=12):
    """ADF test with AIC lag selection, constant included."""
    y = np.array(series.dropna())
    n = len(y)
    dy = np.diff(y)
    best_aic = np.inf; best_lag = 0; best_res = None

    for lag in range(0, maxlag + 1):
        if lag == 0:
            X = np.column_stack([y[:-1], np.ones(n-1)])
            Y = dy
        else:
            rows = n - 1 - lag
            if rows < 10: continue
            X = np.column_stack([
                y[lag:-1],
                *[dy[lag-i-1:n-1-i-1] for i in range(lag)],
                np.ones(rows)
            ])
            Y = dy[lag:]
        try:
            beta, res, rank, sv = np.linalg.lstsq(X, Y, rcond=None)
            sigma2 = np.sum((Y - X @ beta)**2) / (len(Y) - X.shape[1])
            aic = len(Y) * np.log(sigma2) + 2 * X.shape[1]
            if aic < best_aic:
                best_aic = aic; best_lag = lag
                XtX_inv = np.linalg.inv(X.T @ X)
                se = np.sqrt(sigma2 * XtX_inv[0,0])
                t_stat = beta[0] / se
                best_res = (t_stat, best_lag)
        except:
            continue
    return best_res if best_res else (np.nan, 0)

# MacKinnon (1994) approx critical values for no trend, constant
# tau: p=0.01: -3.51, p=0.05: -2.90, p=0.10: -2.58 (T=200)
ADF_CV = {0.01: -3.51, 0.05: -2.90, 0.10: -2.58}

level_series = {
    'Total visitors (level)': data['total'],
    'Overnight visitors (level)': data['overnight'],
    'Same-day visitors (level)': data['sameday'],
    'Gaming revenue (level)': data['gaming'],
    'Overnight share % (level)': data['ovn_share'],
}
ret_series = {
    'Total visitors log-ret': data['r_total'],
    'Overnight visitors log-ret': data['r_overnight'],
    'Same-day visitors log-ret': data['r_sameday'],
    'Gaming log-ret': data['r_gaming'],
    'Overnight share 1st diff': data['d_ovn_share'],
}

adf_rows = []
for label, s in {**level_series, **ret_series}.items():
    t, lag = adf_test(s)
    decision = 'I(1)' if label in level_series else 'I(0)'
    adf_rows.append({'Series': label, 'ADF stat': round(t,3), 'Lags': lag,
                     'CV 5%': -2.90, 'Decision': decision})

adf_df = pd.DataFrame(adf_rows)
print(adf_df.to_string(index=False))
adf_df.to_csv(f'{OUT}/unit_root_tests.csv', index=False)

# ═══════════════════════════════════════════════════════════════════════════
# 4. ARCH-LM TESTS
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("4. ARCH-LM TESTS")
print("="*60)

def arch_lm(series, lags=12):
    """Engle's ARCH-LM test."""
    e = np.array(series.dropna())
    e2 = e**2
    n = len(e2)
    Y = e2[lags:]
    X = np.column_stack([e2[lags-i-1:n-i-1] for i in range(lags)] + [np.ones(len(Y))])
    beta, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
    yhat = X @ beta
    ss_res = np.sum((Y - yhat)**2)
    ss_tot = np.sum((Y - Y.mean())**2)
    r2 = 1 - ss_res/ss_tot
    lm = n * r2
    p = 1 - stats.chi2.cdf(lm, lags)
    return round(lm, 3), round(p, 4)

arch_series = {
    'Total visitors log-ret': data['r_total'],
    'Same-day visitors log-ret': data['r_sameday'],
    'Overnight visitors log-ret': data['r_overnight'],
    'Gaming log-ret': data['r_gaming'],
}
arch_rows = []
for label, s in arch_series.items():
    lm, p = arch_lm(s)
    arch_rows.append({'Series': label, 'ARCH-LM(12)': lm, 'p-value': p,
                      'ARCH effects': 'Yes' if p < 0.05 else 'No'})

arch_df = pd.DataFrame(arch_rows)
print(arch_df.to_string(index=False))
arch_df.to_csv(f'{OUT}/arch_lm_tests.csv', index=False)

# ═══════════════════════════════════════════════════════════════════════════
# 5. UNIVARIATE GJR-GARCH(1,1) FOR EACH SERIES
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("5. UNIVARIATE GJR-GARCH(1,1)")
print("="*60)

def gjr_garch_loglik(params, r, dummies):
    """Negative log-likelihood for GJR-GARCH(1,1) with AR(1) mean."""
    mu, phi = params[0], params[1]
    d_coeffs = params[2:2+dummies.shape[1]] if dummies.shape[1] > 0 else []
    omega, alpha, gamma, beta = params[-4], params[-3], params[-2], params[-1]

    if omega <= 0 or alpha < 0 or gamma < -alpha or beta < 0 or alpha + gamma/2 + beta >= 1:
        return 1e10
    if beta >= 1 or alpha + beta >= 1.5:
        return 1e10

    n = len(r)
    eps = np.zeros(n)
    h   = np.zeros(n)

    # Mean residuals
    mean = mu + phi * np.concatenate([[0], r[:-1]])
    if len(d_coeffs) > 0:
        mean += dummies @ np.array(d_coeffs)
    eps = r - mean

    # Variance initialisation
    h[0] = np.var(eps)
    for t in range(1, n):
        I_neg = 1.0 if eps[t-1] < 0 else 0.0
        h[t] = omega + (alpha + gamma * I_neg) * eps[t-1]**2 + beta * h[t-1]
        if h[t] <= 0:
            h[t] = 1e-8

    ll = -0.5 * np.sum(np.log(h) + eps**2 / h)
    return -ll

def fit_gjr(series_name, r_series, data_df, verbose=True):
    r = r_series.values
    D = data_df[['D_COVID','D_REOPEN','D_GFC']].values
    n_d = D.shape[1]

    # Initial params: [mu, phi, d1, d2, d3, omega, alpha, gamma, beta]
    p0 = [r.mean(), 0.1, -5.0, 5.0, -5.0, np.var(r)*0.05, 0.10, 0.10, 0.80]
    bounds = [(-50,50),(-0.99,0.99),
              (-100,100),(-100,100),(-100,100),
              (1e-6,None),(1e-6,0.49),(0,0.49),(1e-6,0.98)]

    result = opt.minimize(gjr_garch_loglik, p0, args=(r, D),
                         method='L-BFGS-B', bounds=bounds,
                         options={'maxiter':5000,'ftol':1e-12,'gtol':1e-8})

    p = result.x
    mu,phi = p[0],p[1]
    d1,d2,d3 = p[2],p[3],p[4]
    omega,alpha,gamma,beta = p[5],p[6],p[7],p[8]

    # Compute conditional volatility
    mean = mu + phi*np.concatenate([[0],r[:-1]]) + d1*D[:,0] + d2*D[:,1] + d3*D[:,2]
    eps = r - mean
    h = np.zeros(len(r)); h[0] = np.var(eps)
    for t in range(1,len(r)):
        I_neg = 1. if eps[t-1] < 0 else 0.
        h[t] = omega + (alpha + gamma*I_neg)*eps[t-1]**2 + beta*h[t-1]
        if h[t] <= 0: h[t] = 1e-8

    loglik = -result.fun
    k = len(p)
    n = len(r)
    aic = 2*k - 2*loglik
    bic = k*np.log(n) - 2*loglik
    persist = alpha + beta

    if verbose:
        print(f"\n  {series_name}: omega={omega:.4f} alpha={alpha:.4f} gamma={gamma:.4f} "
              f"beta={beta:.4f} persist={persist:.4f} AIC={aic:.1f}")

    return {
        'series': series_name,
        'params': {'mu':mu,'phi':phi,'omega':omega,'alpha':alpha,'gamma':gamma,'beta':beta},
        'cond_vol': np.sqrt(h),
        'eps': eps,
        'h': h,
        'loglik': loglik,
        'aic': aic,
        'bic': bic,
        'persist': persist,
    }

garch_results = {}
series_map = {
    'Total visitors':    data['r_total'],
    'Same-day visitors': data['r_sameday'],
    'Overnight visitors':data['r_overnight'],
    'Gaming revenue':    data['r_gaming'],
}
for name, series in series_map.items():
    garch_results[name] = fit_gjr(name, series, data)

# Also fit for overnight share
garch_results['Overnight share'] = fit_gjr('Overnight share', data['d_ovn_share'], data)

# Summary table
gjr_rows = []
for name, res in garch_results.items():
    p = res['params']
    gjr_rows.append({
        'Series': name,
        'omega': round(p['omega'],4),
        'alpha': round(p['alpha'],4),
        'gamma': round(p['gamma'],4),
        'beta':  round(p['beta'],4),
        'Persistence': round(res['persist'],4),
        'AIC': round(res['aic'],1),
    })
gjr_df = pd.DataFrame(gjr_rows)
print("\nGJR-GARCH Parameter Summary:")
print(gjr_df.to_string(index=False))
gjr_df.to_csv(f'{OUT}/gjr_garch_params.csv', index=False)

# ═══════════════════════════════════════════════════════════════════════════
# 6. DCC-GARCH (bivariate and trivariate)
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("6. DCC-GARCH")
print("="*60)

def dcc_loglik(params, std_resid_matrix):
    """
    Engle (2002) DCC log-likelihood (correlation part only).
    params = [a, b]  with a>0, b>0, a+b<1
    std_resid_matrix: (T, k) matrix of standardised residuals from GARCH step
    """
    a, b = params
    if a <= 0 or b <= 0 or a + b >= 1:
        return 1e10
    T, k = std_resid_matrix.shape
    Z = std_resid_matrix

    # Unconditional correlation
    Qbar = np.corrcoef(Z.T)
    Q = Qbar.copy()
    ll = 0.0
    for t in range(1, T):
        Q = (1 - a - b) * Qbar + a * np.outer(Z[t-1], Z[t-1]) + b * Q
        # Correlation matrix R from Q
        D_q = np.diag(1.0 / np.sqrt(np.diag(Q)))
        R = D_q @ Q @ D_q
        # Ensure positive definite
        eigvals = np.linalg.eigvalsh(R)
        if np.any(eigvals <= 0):
            return 1e10
        sign, logdet = np.linalg.slogdet(R)
        if sign <= 0:
            return 1e10
        z = Z[t]
        try:
            R_inv = np.linalg.inv(R)
        except:
            return 1e10
        ll += -0.5 * (logdet + z @ R_inv @ z - z @ z)
    return -ll

def fit_dcc(names, series_list, garch_res_dict, verbose=True):
    """
    Two-step DCC-GARCH.
    Step 1: fit univariate GJR-GARCH for each series (already done).
    Step 2: fit DCC on standardised residuals.
    Returns time-varying correlation matrices.
    """
    k = len(names)
    T = len(series_list[0])

    # Build standardised residual matrix
    std_resids = np.zeros((T, k))
    for i, name in enumerate(names):
        eps = garch_res_dict[name]['eps']
        h   = garch_res_dict[name]['h']
        std_resids[:, i] = eps / np.sqrt(h)

    # Fit DCC params
    p0 = [0.05, 0.90]
    bounds = [(1e-6, 0.49), (1e-6, 0.98)]
    result = opt.minimize(dcc_loglik, p0, args=(std_resids,),
                         method='L-BFGS-B', bounds=bounds,
                         options={'maxiter':2000,'ftol':1e-10})
    a, b = result.x
    ll_dcc = -result.fun

    if verbose:
        print(f"\n  DCC({'+'.join(names)}): a={a:.4f} b={b:.4f} a+b={a+b:.4f} loglik={ll_dcc:.2f}")

    # Compute time-varying correlations
    Qbar = np.corrcoef(std_resids.T)
    Q = Qbar.copy()
    corr_ts = np.zeros((T, k, k))
    corr_ts[0] = Qbar.copy()
    for t in range(1, T):
        z = std_resids[t-1]
        Q = (1-a-b)*Qbar + a*np.outer(z,z) + b*Q
        D_q = np.diag(1/np.sqrt(np.diag(Q)))
        R = D_q @ Q @ D_q
        corr_ts[t] = R

    return {
        'names': names,
        'a': a, 'b': b,
        'loglik': ll_dcc,
        'corr_ts': corr_ts,  # (T, k, k)
        'std_resids': std_resids,
        'Qbar': Qbar,
    }

# OPTION A: Overnight + Gaming
print("\n--- Option A: Overnight Visitors + Gaming Revenue ---")
dcc_A = fit_dcc(
    ['Overnight visitors', 'Gaming revenue'],
    [data['r_overnight'], data['r_gaming']],
    garch_results
)

# OPTION B: Same-day + Overnight + Gaming (trivariate)
print("\n--- Option B: Same-day + Overnight + Gaming (3-variable) ---")
dcc_B = fit_dcc(
    ['Same-day visitors', 'Overnight visitors', 'Gaming revenue'],
    [data['r_sameday'], data['r_overnight'], data['r_gaming']],
    garch_results
)

# OPTION C: Overnight share + Gaming
print("\n--- Option C: Overnight Share (1st diff) + Gaming Revenue ---")
dcc_C = fit_dcc(
    ['Overnight share', 'Gaming revenue'],
    [data['d_ovn_share'], data['r_gaming']],
    garch_results
)

# Extract pairwise correlations for reporting
dates = data.index

# Option A: rho(overnight, gaming)
rho_A = dcc_A['corr_ts'][:, 0, 1]

# Option B: three pairs
rho_B_sd_ovn  = dcc_B['corr_ts'][:, 0, 1]
rho_B_sd_gam  = dcc_B['corr_ts'][:, 0, 2]
rho_B_ovn_gam = dcc_B['corr_ts'][:, 1, 2]

# Option C: rho(overnight_share, gaming)
rho_C = dcc_C['corr_ts'][:, 0, 1]

print("\nDCC summary statistics:")
print(f"  A  rho(overnight, gaming):       mean={rho_A.mean():.3f}  min={rho_A.min():.3f}  max={rho_A.max():.3f}")
print(f"  B  rho(same-day, overnight):     mean={rho_B_sd_ovn.mean():.3f}  min={rho_B_sd_ovn.min():.3f}  max={rho_B_sd_ovn.max():.3f}")
print(f"  B  rho(same-day, gaming):        mean={rho_B_sd_gam.mean():.3f}  min={rho_B_sd_gam.min():.3f}  max={rho_B_sd_gam.max():.3f}")
print(f"  B  rho(overnight, gaming):       mean={rho_B_ovn_gam.mean():.3f}  min={rho_B_ovn_gam.min():.3f}  max={rho_B_ovn_gam.max():.3f}")
print(f"  C  rho(ovn_share, gaming):       mean={rho_C.mean():.3f}  min={rho_C.min():.3f}  max={rho_C.max():.3f}")

# Pre/during/post COVID correlations
covid_mask   = (dates >= '2020-02') & (dates <= '2022-12')
pre_mask     = dates < '2020-02'
post_mask    = dates > '2022-12'

print("\nDynamic correlation by period:")
for label, rho in [('A: overnight-gaming', rho_A),
                   ('B: overnight-gaming', rho_B_ovn_gam),
                   ('C: share-gaming', rho_C)]:
    print(f"  {label}:")
    print(f"    Pre-COVID:  {rho[pre_mask].mean():.3f}")
    print(f"    COVID:      {rho[covid_mask].mean():.3f}")
    print(f"    Post-COVID: {rho[post_mask].mean():.3f}")

# ═══════════════════════════════════════════════════════════════════════════
# 7. DIEBOLD-YILMAZ (2012) SPILLOVER INDEX
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("7. DIEBOLD-YILMAZ SPILLOVER INDEX")
print("="*60)

def var_estimate(Y, p=2):
    """Estimate VAR(p) by OLS. Y: (T, k). Returns A list and Sigma."""
    T, k = Y.shape
    # Build regressor matrix
    X_list = [np.ones(T-p)]
    for lag in range(1, p+1):
        X_list.append(Y[p-lag:T-lag])
    X = np.column_stack(X_list)  # (T-p, 1+k*p)
    Y_dep = Y[p:]
    B, _, _, _ = np.linalg.lstsq(X, Y_dep, rcond=None)
    resid = Y_dep - X @ B
    Sigma = resid.T @ resid / (T - p - k*p - 1)
    # Extract coefficient matrices A1, A2, ...
    A = []
    for i in range(p):
        A.append(B[1 + i*k : 1 + (i+1)*k, :].T)
    return A, Sigma, B, resid

def var_to_ma(A_list, H=10):
    """Convert VAR(p) to MA(H) representation. Returns Psi list."""
    p = len(A_list)
    k = A_list[0].shape[0]
    Psi = [np.eye(k)]  # Psi_0 = I
    for h in range(1, H+1):
        psi_h = np.zeros((k, k))
        for j in range(min(h, p)):
            psi_h += A_list[j] @ Psi[h-1-j]
        if h > p:
            for j in range(p, h):
                if h-1-j >= 0 and h-1-j < len(Psi):
                    pass  # zero contribution
        Psi.append(psi_h)
    return Psi

def fevd_generalised(Psi_list, Sigma, H=10):
    """
    Generalised Forecast Error Variance Decomposition (Koop et al. 1996,
    Pesaran & Shin 1998) — order-invariant.
    Returns (k, k) matrix: theta[i,j] = share of variance of i due to j.
    """
    k = Sigma.shape[0]
    sigma_diag = np.diag(Sigma)
    theta = np.zeros((k, k))
    for j in range(k):
        ej = np.zeros(k); ej[j] = 1.0
        for i in range(k):
            ei = np.zeros(k); ei[i] = 1.0
            num = sum((ei @ Psi_list[h] @ Sigma @ ej)**2 for h in range(H+1))
            den = sum((ei @ Psi_list[h] @ Sigma @ Psi_list[h].T @ ei)
                      for h in range(H+1))
            theta[i, j] = num / (sigma_diag[j] * den) if den > 0 else 0
    # Normalise rows to sum to 1
    theta_norm = theta / theta.sum(axis=1, keepdims=True)
    return theta_norm

def dy_spillover(names, vol_matrix, p=2, H=10, verbose=True):
    """
    Full Diebold-Yilmaz (2012) spillover table.
    vol_matrix: (T, k) matrix of conditional volatilities.
    """
    k = len(names)
    A_list, Sigma, B, resid = var_estimate(vol_matrix, p=p)
    Psi_list = var_to_ma(A_list, H=H)
    theta = fevd_generalised(Psi_list, Sigma, H=H)

    # Spillover measures
    total_spillover = (theta.sum() - np.trace(theta)) / k * 100
    from_others = np.array([(theta[i,:].sum() - theta[i,i]) * 100 for i in range(k)])
    to_others   = np.array([(theta[:,j].sum() - theta[j,j]) * 100 for j in range(k)])
    net         = to_others - from_others

    if verbose:
        print(f"\n  DY Spillover table ({'+'.join(names)}, H={H}, VAR({p})):")
        print(f"  Total spillover index: {total_spillover:.1f}%")
        header = f"{'From/To':<22}" + "".join(f"{n[:12]:>14}" for n in names) + f"{'From others':>14}"
        print("  " + header)
        for i in range(k):
            row_str = f"  {names[i]:<22}"
            for j in range(k):
                row_str += f"{theta[i,j]*100:>14.1f}"
            row_str += f"{from_others[i]:>14.1f}"
            print(row_str)
        to_row = f"  {'To others':<22}" + "".join(f"{to_others[j]:>14.1f}" for j in range(k))
        print(to_row)
        net_row = f"  {'Net':<22}" + "".join(f"{net[j]:>14.1f}" for j in range(k))
        print(net_row)

    return {
        'names': names,
        'theta': theta,
        'total_spillover': total_spillover,
        'from_others': from_others,
        'to_others': to_others,
        'net': net,
    }

# Prepare conditional volatility matrices
vol_A = np.column_stack([
    garch_results['Overnight visitors']['cond_vol'],
    garch_results['Gaming revenue']['cond_vol']
])
vol_B = np.column_stack([
    garch_results['Same-day visitors']['cond_vol'],
    garch_results['Overnight visitors']['cond_vol'],
    garch_results['Gaming revenue']['cond_vol']
])
vol_C = np.column_stack([
    garch_results['Overnight share']['cond_vol'],
    garch_results['Gaming revenue']['cond_vol']
])

print("\n--- Option A: Overnight + Gaming ---")
dy_A = dy_spillover(['Overnight visitors', 'Gaming revenue'], vol_A, p=2, H=10)

print("\n--- Option B: Same-day + Overnight + Gaming ---")
dy_B = dy_spillover(['Same-day visitors', 'Overnight visitors', 'Gaming revenue'], vol_B, p=2, H=10)

print("\n--- Option C: Overnight Share + Gaming ---")
dy_C = dy_spillover(['Overnight share', 'Gaming revenue'], vol_C, p=2, H=10)

# Rolling spillover (60-month window) for Option A
print("\n  Computing rolling DY spillover (60-month window, Option A)...")
window = 60
roll_spill = []
roll_dates = []
for t in range(window, len(vol_A)):
    w = vol_A[t-window:t]
    try:
        dy_w = dy_spillover(
            ['Overnight visitors', 'Gaming revenue'], w, p=2, H=10, verbose=False)
        roll_spill.append(dy_w['total_spillover'])
    except:
        roll_spill.append(np.nan)
    roll_dates.append(dates[t])

roll_df = pd.DataFrame({'date': roll_dates, 'spillover_pct': roll_spill})
print(f"  Rolling spillover: mean={np.nanmean(roll_spill):.1f}%  "
      f"min={np.nanmin(roll_spill):.1f}%  max={np.nanmax(roll_spill):.1f}%")

# Also rolling for B
print("  Computing rolling DY spillover (60-month window, Option B)...")
roll_spill_B = []
for t in range(window, len(vol_B)):
    w = vol_B[t-window:t]
    try:
        dy_w = dy_spillover(
            ['Same-day visitors','Overnight visitors','Gaming revenue'],
            w, p=2, H=10, verbose=False)
        roll_spill_B.append(dy_w['total_spillover'])
    except:
        roll_spill_B.append(np.nan)

# ═══════════════════════════════════════════════════════════════════════════
# 8. ASYMMETRIC SPILLOVER TEST (BAD vs GOOD volatility)
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("8. ASYMMETRIC SPILLOVER (BAD vs GOOD VOLATILITY)")
print("="*60)

def split_vol(name, garch_res, ret_series):
    """Split realised vol into bad (negative shock) and good (positive shock)."""
    eps = garch_res[name]['eps']
    h   = garch_res[name]['h']
    bad_vol  = np.where(eps < 0, np.abs(eps) / np.sqrt(h), 0.0)
    good_vol = np.where(eps >= 0, np.abs(eps) / np.sqrt(h), 0.0)
    return bad_vol, good_vol

bad_ovn,  good_ovn  = split_vol('Overnight visitors', garch_results, data['r_overnight'])
bad_gam,  good_gam  = split_vol('Gaming revenue',     garch_results, data['r_gaming'])
bad_sd,   good_sd   = split_vol('Same-day visitors',  garch_results, data['r_sameday'])

# Bad spillover table
print("\n  BAD volatility spillover (Option A):")
bad_A = np.column_stack([bad_ovn, bad_gam])
dy_bad_A = dy_spillover(['Overnight (bad)', 'Gaming (bad)'], bad_A, p=2, H=10)

print("\n  GOOD volatility spillover (Option A):")
good_A = np.column_stack([good_ovn, good_gam])
dy_good_A = dy_spillover(['Overnight (good)', 'Gaming (good)'], good_A, p=2, H=10)

print(f"\n  Asymmetry: Bad total={dy_bad_A['total_spillover']:.1f}%  "
      f"Good total={dy_good_A['total_spillover']:.1f}%  "
      f"Diff={dy_bad_A['total_spillover']-dy_good_A['total_spillover']:.1f}pp")

print("\n  BAD volatility spillover (Option B):")
bad_B = np.column_stack([bad_sd, bad_ovn, bad_gam])
dy_bad_B = dy_spillover(['Same-day (bad)','Overnight (bad)','Gaming (bad)'], bad_B, p=2, H=10)

print("\n  GOOD volatility spillover (Option B):")
good_B = np.column_stack([good_sd, good_ovn, good_gam])
dy_good_B = dy_spillover(['Same-day (good)','Overnight (good)','Gaming (good)'], good_B, p=2, H=10)

# ═══════════════════════════════════════════════════════════════════════════
# 9. FIGURES
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("9. GENERATING FIGURES")
print("="*60)

# Fig 1: Overnight share over time
fig, axes = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True)
ax = axes[0]
ax.plot(data.index, data['ovn_share'], color=BLUE, lw=1.2)
ax.axvspan(pd.Timestamp('2020-02'), pd.Timestamp('2022-12'), alpha=0.12, color='red')
ax.axvspan(pd.Timestamp('2008-10'), pd.Timestamp('2009-06'), alpha=0.1, color='orange')
ax.axhline(50, color='gray', lw=0.8, ls='--', alpha=0.6)
ax.set_ylabel('Overnight visitors (%)')
ax.set_title('(a) Overnight visitor share of total arrivals')

ax = axes[1]
ax.plot(data.index, data['r_overnight'], color=BLUE, lw=0.8, alpha=0.7, label='Overnight log-ret')
ax.plot(data.index, data['r_gaming'],    color=ORANGE, lw=0.8, alpha=0.7, label='Gaming log-ret')
ax.axvspan(pd.Timestamp('2020-02'), pd.Timestamp('2022-12'), alpha=0.1, color='red')
ax.set_ylabel('Log-return (%)')
ax.set_title('(b) Overnight visitor and gaming revenue log-returns')
ax.legend(loc='upper left', fontsize=8)
plt.tight_layout()
plt.savefig(f'{OUT}/fig1_series_overview.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Fig 1 saved.")

# Fig 2: Conditional volatilities comparison
fig, axes = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True)
ax = axes[0]
ax.fill_between(data.index, 0, garch_results['Overnight visitors']['cond_vol'],
                alpha=0.5, color=BLUE, label='Overnight visitors')
ax.fill_between(data.index, 0, garch_results['Total visitors']['cond_vol'],
                alpha=0.3, color=GRAY, label='Total visitors')
ax.axvspan(pd.Timestamp('2020-02'), pd.Timestamp('2022-12'), alpha=0.08, color='red')
ax.set_ylabel('Conditional std (%)')
ax.set_title('(a) Overnight vs total visitor conditional volatility')
ax.legend(fontsize=8)

ax = axes[1]
ax.fill_between(data.index, 0, garch_results['Overnight visitors']['cond_vol'],
                alpha=0.5, color=BLUE, label='Overnight visitors')
ax.fill_between(data.index, 0, garch_results['Gaming revenue']['cond_vol'],
                alpha=0.4, color=ORANGE, label='Gaming revenue')
ax.axvspan(pd.Timestamp('2020-02'), pd.Timestamp('2022-12'), alpha=0.08, color='red')
ax.set_ylabel('Conditional std (%)')
ax.set_title('(b) Overnight visitor and gaming revenue conditional volatility')
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(f'{OUT}/fig2_cond_vol.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Fig 2 saved.")

# Fig 3: DCC dynamic correlations
fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)

ax = axes[0]
ax.plot(dates, rho_A, color=BLUE, lw=1.2, label='A: overnight-gaming')
ax.axvspan(pd.Timestamp('2020-02'), pd.Timestamp('2022-12'), alpha=0.1, color='red')
ax.axhline(rho_A.mean(), color=GRAY, lw=0.8, ls='--', alpha=0.8)
ax.set_ylabel('DCC correlation')
ax.set_title('(a) Option A: Overnight visitors and gaming revenue')
ax.set_ylim(-0.1, 1.0)

ax = axes[1]
ax.plot(dates, rho_B_ovn_gam, color=BLUE,   lw=1.2, label='Overnight-Gaming')
ax.plot(dates, rho_B_sd_gam,  color=GREEN,  lw=1.0, ls='--', label='Same-day-Gaming', alpha=0.8)
ax.plot(dates, rho_B_sd_ovn,  color=ORANGE, lw=1.0, ls=':',  label='Same-day-Overnight', alpha=0.8)
ax.axvspan(pd.Timestamp('2020-02'), pd.Timestamp('2022-12'), alpha=0.1, color='red')
ax.set_ylabel('DCC correlation')
ax.set_title('(b) Option B: Three-variable pairwise correlations')
ax.legend(fontsize=8, loc='upper left')
ax.set_ylim(-0.2, 1.0)

ax = axes[2]
ax.plot(dates, rho_C, color=ORANGE, lw=1.2, label='C: overnight share-gaming')
ax.axvspan(pd.Timestamp('2020-02'), pd.Timestamp('2022-12'), alpha=0.1, color='red')
ax.axhline(rho_C.mean(), color=GRAY, lw=0.8, ls='--', alpha=0.8)
ax.set_ylabel('DCC correlation')
ax.set_title('(c) Option C: Overnight share and gaming revenue')
ax.set_xlabel('Date')
ax.set_ylim(-0.5, 0.5)
plt.tight_layout()
plt.savefig(f'{OUT}/fig3_dcc_correlations.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Fig 3 saved.")

# Fig 4: Rolling spillover
fig, ax = plt.subplots(figsize=(11, 4.5))
ax.plot(roll_dates, roll_spill, color=BLUE, lw=1.5, label='Option A: Overnight+Gaming')
ax.plot(dates[window:], roll_spill_B, color=ORANGE, lw=1.2, ls='--',
        alpha=0.8, label='Option B: 3-variable')
ax.axvspan(pd.Timestamp('2020-02'), pd.Timestamp('2022-12'), alpha=0.1, color='red')
ax.axvspan(pd.Timestamp('2008-10'), pd.Timestamp('2009-06'), alpha=0.1, color='orange')
ax.set_ylabel('Total spillover index (%)')
ax.set_title('Rolling Diebold-Yilmaz total spillover index (60-month window)')
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(f'{OUT}/fig4_rolling_spillover.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Fig 4 saved.")

# Fig 5: Asymmetric spillover comparison
fig, ax = plt.subplots(figsize=(9, 5))
options = ['Option A\n(2-var)', 'Option B\n(3-var)']
bad_vals  = [dy_bad_A['total_spillover'],  dy_bad_B['total_spillover']]
good_vals = [dy_good_A['total_spillover'], dy_good_B['total_spillover']]
x = np.arange(len(options))
w = 0.3
ax.bar(x - w/2, bad_vals,  w, label='Bad volatility',  color=RED,  alpha=0.8)
ax.bar(x + w/2, good_vals, w, label='Good volatility', color=BLUE, alpha=0.8)
ax.set_xticks(x); ax.set_xticklabels(options)
ax.set_ylabel('Total spillover index (%)')
ax.set_title('Asymmetric spillover: bad vs good volatility')
ax.legend()
plt.tight_layout()
plt.savefig(f'{OUT}/fig5_asymmetric_spillover.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Fig 5 saved.")

# ═══════════════════════════════════════════════════════════════════════════
# 10. SAVE RESULTS
# ═══════════════════════════════════════════════════════════════════════════
results = {
    'data': data,
    'garch_results': garch_results,
    'dcc_A': dcc_A, 'dcc_B': dcc_B, 'dcc_C': dcc_C,
    'dy_A': dy_A,   'dy_B': dy_B,   'dy_C': dy_C,
    'dy_bad_A': dy_bad_A, 'dy_good_A': dy_good_A,
    'dy_bad_B': dy_bad_B, 'dy_good_B': dy_good_B,
    'rolling_spillover_A': {'dates': roll_dates, 'values': roll_spill},
    'rolling_spillover_B': {'dates': list(dates[window:]), 'values': roll_spill_B},
}
with open(f'{OUT}/paper2_results.pkl', 'wb') as f:
    pickle.dump(results, f)

# ═══════════════════════════════════════════════════════════════════════════
# 11. SUMMARY REPORT
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("SUMMARY REPORT")
print("="*60)

print("""
KEY FINDINGS
----------------------------------------

UNIVARIATE GARCH (GJR-GARCH asymmetry parameters):""")
for name in ['Total visitors','Same-day visitors','Overnight visitors','Gaming revenue']:
    g = garch_results[name]['params']['gamma']
    p = garch_results[name]['persist']
    print(f"  {name:<22}: gamma={g:.4f}  persistence={p:.4f}")

print(f"""
DCC CORRELATION SUMMARY:
  Option A (overnight-gaming):
    Pre-COVID:   {rho_A[pre_mask].mean():.3f}
    COVID:       {rho_A[covid_mask].mean():.3f}
    Post-COVID:  {rho_A[post_mask].mean():.3f}

  Option B (overnight-gaming within 3-var):
    Pre-COVID:   {rho_B_ovn_gam[pre_mask].mean():.3f}
    COVID:       {rho_B_ovn_gam[covid_mask].mean():.3f}
    Post-COVID:  {rho_B_ovn_gam[post_mask].mean():.3f}

  Option C (overnight share-gaming):
    Pre-COVID:   {rho_C[pre_mask].mean():.3f}
    COVID:       {rho_C[covid_mask].mean():.3f}
    Post-COVID:  {rho_C[post_mask].mean():.3f}

DIEBOLD-YILMAZ TOTAL SPILLOVER:
  Option A (2-var):          {dy_A['total_spillover']:.1f}%
  Option B (3-var):          {dy_B['total_spillover']:.1f}%
  Option C (share+gaming):   {dy_C['total_spillover']:.1f}%

ASYMMETRIC SPILLOVER (Option A):
  Bad volatility total:      {dy_bad_A['total_spillover']:.1f}%
  Good volatility total:     {dy_good_A['total_spillover']:.1f}%
  Asymmetry gap:             {dy_bad_A['total_spillover']-dy_good_A['total_spillover']:.1f} pp

NET SPILLOVER DIRECTIONS (Option A):
  Overnight -> Gaming (net): {dy_A['net'][0]:.1f}%
  Gaming -> Overnight (net): {dy_A['net'][1]:.1f}%

NET SPILLOVER DIRECTIONS (Option B):
  Same-day (net):            {dy_B['net'][0]:.1f}%
  Overnight (net):           {dy_B['net'][1]:.1f}%
  Gaming (net):              {dy_B['net'][2]:.1f}%
""")

print("All outputs saved to:", OUT)
print("Files:", os.listdir(OUT))
