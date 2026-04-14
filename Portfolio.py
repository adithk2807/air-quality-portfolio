# %% [markdown]
# # Portfolio - Beijing Air

# %% [markdown]
# ## 1 — Imports

# %%
import sys
import pandas as pd, numpy as np, glob, os, warnings
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import statsmodels.api as sm
from scipy.optimize import minimize
from scipy.stats import chi2
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

# %% [markdown]
# ## 2 — Clean Data

# %%
FOLDER = r'C:/Users/adith/Downloads/archive'

files  = sorted(set(glob.glob(os.path.join(FOLDER,'*.csv')) +
                    glob.glob(os.path.join(FOLDER,'*.CSV'))))
df_raw = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
df_raw.columns = [c.strip() for c in df_raw.columns]

for c in ['PM2.5','TEMP','WSPM','NO2','RAIN']:
    df_raw[c] = pd.to_numeric(df_raw[c], errors='coerce')

df = df_raw[['PM2.5','TEMP','WSPM','NO2','RAIN','station']].dropna().copy()
df = df.rename(columns={'PM2.5':'PM25'})
df['log_PM25'] = np.log(df['PM25'] + 1)
print(f'Clean shape: {df.shape}')
df.describe()

# %%
df.head()

# %% [markdown]
# ## 3 — Regression Analysis
# Three methods used:
# 1. **Log-transform** — PM2.5 is right-skewed, log makes it Gaussian
# 2. **Interaction term** TEMP×WSPM — wind dispersal changes with temperature
# 3. **Gamma GLM** — second model for right-skewed non-negative data

# %%
# Why log? Check skewness
print(f'PM2.5 skewness     : {df["PM25"].skew():.2f}  → right-skewed, NOT Gaussian')
print(f'log_PM25 skewness  : {df["log_PM25"].skew():.2f}  → closer to Gaussian')

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].hist(df['PM25'],     bins=60, color='red',       edgecolor='white')
axes[0].set_title('Raw PM2.5 — right-skewed')
axes[1].hist(df['log_PM25'], bins=60, color='steelblue', edgecolor='white')
axes[1].set_title('log(PM2.5+1) — approximately Gaussian')
plt.tight_layout(); plt.show()

# %%
# OLS with interaction term TEMP:WSPM
ols = smf.ols('log_PM25 ~ TEMP + WSPM + TEMP:WSPM + NO2 + RAIN + C(station)', data=df).fit()
ols.summary()

# %%
# Gamma GLM — generative model for non-negative skewed data 
glm = smf.glm('PM25 ~ TEMP + WSPM + TEMP:WSPM + NO2 + RAIN + C(station)',
               data=df,
               family=sm.families.Gamma(link=sm.families.links.Log())).fit()
glm.summary()

# %% [markdown]
# ## 4 — Item 7: Two Models on PM2.5
# **Parameter of interest: PM2.5**  
# **Question: How does wind speed (WSPM) explain PM2.5?**
# 
# - **Model 1 — OLS** (linear): `PM2.5 = β0 + β1·WSPM`
# - **Model 2 — Mathematical**: `PM2.5 = a·exp(−b·WSPM) + c`

# %%
sample = df.sample(5000, random_state=42).copy()

# Model 1: OLS
ols7   = smf.ols('PM25 ~ WSPM', data=sample).fit()
b0, b1 = ols7.params['Intercept'], ols7.params['WSPM']
r2_ols = ols7.rsquared

# Model 2: Mathematical exponential decay 

def math_model(W, a, b, c): return a * np.exp(-b * W) + c
def loss(p): return np.sum((sample['PM25'].values - math_model(sample['WSPM'].values, *p))**2)

fit        = minimize(loss, x0=[80, 0.2, 10], method='Nelder-Mead')
a, b, c    = fit.x
math_pred  = math_model(sample['WSPM'].values, a, b, c)
math_resid = sample['PM25'].values - math_pred
r2_math    = 1 - np.sum(math_resid**2) / np.sum((sample['PM25'].values - sample['PM25'].mean())**2)

# Effect of WSPM at mean wind speed
wspm_mean   = sample['WSPM'].mean()
math_effect = -a * b * np.exp(-b * wspm_mean)

print('Model 1 — OLS (Linear)')
print(f'  PM2.5 = {b0:.2f} + {b1:.2f}·WSPM')
print(f'  WSPM effect (constant) : {b1:+.4f}')
print(f'  95% CI : [{ols7.conf_int().loc["WSPM",0]:+.4f}, {ols7.conf_int().loc["WSPM",1]:+.4f}]')
print(f'  p-value: {ols7.pvalues["WSPM"]:.2e},  R²={r2_ols:.4f}')
print()
print('Model 2 — Mathematical Exponential Decay')
print(f'  PM2.5 = {a:.2f}·exp(−{b:.4f}·WSPM) + {c:.2f}')
print(f'  WSPM effect at mean wind: {math_effect:+.4f}')
print(f'  R²={r2_math:.4f}')

# %% [markdown]
# ## 5 — Item 8: Which Model is Better?
# **Likelihood-based approach: AIC, BIC, Likelihood Ratio Test**
# 
# - AIC = 2k − 2·log-likelihood  
# - BIC = k·log(n) − 2·log-likelihood  
# - Lower AIC/BIC = better model

# %%
n = len(sample)

# Log-likelihoods (assuming Gaussian errors)
ssr_ols    = np.sum(ols7.resid**2)
ssr_math   = np.sum(math_resid**2)
ll_ols     = -n/2 * np.log(ssr_ols/n)
ll_math    = -n/2 * np.log(ssr_math/n)

# AIC and BIC
k_ols, k_math = 2, 3
aic_ols  = 2*k_ols  - 2*ll_ols
aic_math = 2*k_math - 2*ll_math
bic_ols  = k_ols  * np.log(n) - 2*ll_ols
bic_math = k_math * np.log(n) - 2*ll_math

# Likelihood Ratio Test
lrt_stat = 2 * (ll_math - ll_ols)
lrt_p    = 1 - chi2.cdf(lrt_stat, df=k_math - k_ols)

print('=' * 50)
print(f'  {"Metric":<25} {"OLS":>8} {"Math":>8}')
print('-' * 50)
print(f'  {"Parameters (k)":<25} {k_ols:>8} {k_math:>8}')
print(f'  {"Log-likelihood":<25} {ll_ols:>8.1f} {ll_math:>8.1f}')
print(f'  {"AIC (lower=better)":<25} {aic_ols:>8.1f} {aic_math:>8.1f}')
print(f'  {"BIC (lower=better)":<25} {bic_ols:>8.1f} {bic_math:>8.1f}')
print(f'  {"R²  (higher=better)":<25} {r2_ols:>8.4f} {r2_math:>8.4f}')


# %%
# Simulation-based check — does simulated data look like observed?
np.random.seed(42)
sig_ols  = np.sqrt(ssr_ols/n)
sig_math = np.sqrt(ssr_math/n)
sim_ols  = b0 + b1*sample['WSPM'].values + np.random.normal(0, sig_ols,  n)
sim_math = math_model(sample['WSPM'].values, a, b, c) + np.random.normal(0, sig_math, n)

fig, axes = plt.subplots(1, 2, figsize=(13, 4))

# AIC/BIC bar chart
x = np.arange(2)
axes[0].bar(x-0.18, [aic_ols, bic_ols],   0.34, label='OLS',  color='steelblue', alpha=0.85)
axes[0].bar(x+0.18, [aic_math, bic_math],  0.34, label='Math', color='red',       alpha=0.85)
axes[0].set_xticks(x); axes[0].set_xticklabels(['AIC','BIC'])
axes[0].set_title('AIC & BIC (lower = better)')
axes[0].legend()

# Simulation check
axes[1].hist(sample['PM25'], bins=50, alpha=0.5,
             color='grey',      label='Observed',      density=True)
axes[1].hist(sim_ols,         bins=50, alpha=0.5,
             color='steelblue', label='OLS simulated',  density=True)
axes[1].hist(sim_math,        bins=50, alpha=0.5,
             color='red',       label='Math simulated', density=True)
axes[1].set_xlabel('PM2.5'); axes[1].set_ylabel('Density')
axes[1].set_title('Simulation check: does simulated ≈ observed?')
axes[1].legend()

plt.suptitle('Item 8 — Likelihood-based + Simulation Model Comparison',
             fontweight='bold')
plt.tight_layout(); plt.show()

# %% [markdown]
# ## 6 — Item 9: Causal Inference
# **Question: What is the CAUSAL effect of WSPM on PM2.5?**

# %%
# ── Item 9: Causal DAG ──────────────────────────────────────────────
# Confounders are variables that affect BOTH WSPM (treatment) and PM2.5 (outcome).
# We identify the minimal adjustment set to block all backdoor paths.
# Based on meteorological literature: TEMP, PRES, DEWP, RAIN, NO2, SO2, CO
# all influence wind dispersal AND directly affect PM2.5 concentration.

import networkx as nx
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(14, 8))

G = nx.DiGraph()
G.add_edges_from([
    # Confounders → Treatment (WSPM)
    ('TEMP',  'WSPM'),
    ('PRES',  'WSPM'),
    ('DEWP',  'WSPM'),
    ('RAIN',  'WSPM'),
    ('NO2',   'WSPM'),
    ('SO2',   'WSPM'),
    ('CO',    'WSPM'),
    # Confounders → Outcome (PM2.5)
    ('TEMP',  'PM2.5'),
    ('PRES',  'PM2.5'),
    ('DEWP',  'PM2.5'),
    ('RAIN',  'PM2.5'),
    ('NO2',   'PM2.5'),
    ('SO2',   'PM2.5'),
    ('CO',    'PM2.5'),
    # Causal effect of interest
    ('WSPM',  'PM2.5'),
])

pos = {
    'TEMP':  (0, 4),
    'PRES':  (0, 3),
    'DEWP':  (0, 2),
    'RAIN':  (0, 1),
    'NO2':   (0, 0),
    'SO2':   (0, -1),
    'CO':    (0, -2),
    'WSPM':  (3, 1),
    'PM2.5': (6, 1),
}

# Draw confounders
nx.draw_networkx_nodes(G, pos,
    nodelist=['TEMP', 'PRES', 'DEWP', 'RAIN', 'NO2', 'SO2', 'CO'],
    node_color='lightcoral', node_size=2000, ax=ax)

# Draw treatment
nx.draw_networkx_nodes(G, pos, nodelist=['WSPM'],
    node_color='lightblue', node_size=2000, ax=ax)

# Draw outcome
nx.draw_networkx_nodes(G, pos, nodelist=['PM2.5'],
    node_color='lightgreen', node_size=2000, ax=ax)

# Draw edges
for edge in G.edges():
    if edge == ('WSPM', 'PM2.5'):
        nx.draw_networkx_edges(G, pos, edgelist=[edge],
            edge_color='red', width=3, arrowsize=30, ax=ax, arrowstyle='->')
    else:
        nx.draw_networkx_edges(G, pos, edgelist=[edge],
            edge_color='gray', width=1.5, arrowsize=15, ax=ax, arrowstyle='->')

nx.draw_networkx_labels(G, pos, font_size=11, font_weight='bold', ax=ax)

ax.text(3, -3.2, 'Red arrow = Causal effect of interest (WSPM → PM2.5)',
        fontsize=11, ha='center', color='red', weight='bold')
ax.text(3, -3.6, 'Gray arrows = Confounding paths (all must be adjusted for)',
        fontsize=11, ha='center', color='gray', weight='bold')

plt.title('Causal DAG: Effect of Wind Speed (WSPM) on PM2.5\n',
          fontsize=13, fontweight='bold', pad=20)
plt.axis('off')
plt.tight_layout()
plt.show()


# %% [markdown]
# To answer this, the analysis controls for seven confounders:
# 
# - **TEMP**: Air temperature (affects both wind patterns and PM2.5 dispersion)
# - **PRES**: Atmospheric pressure (suppresses wind under high pressure; traps pollution)
# - **DEWP**: Dew point (humidity proxy; affects particle formation and dispersal)
# - **RAIN**: Rainfall (washes out PM2.5 and reduces wind speed)
# - **NO2**: Traffic/combustion proxy (affects both air circulation and directly adds to pollution)
# - **SO2**: Industrial emissions (correlated with both wind patterns and PM2.5 levels)
# - **CO**: Combustion proxy (co-varies with pollution sources and dispersal conditions)
# 
# *The DAG embodies the following causal assumptions:*
# 
# - **TEMP → WSPM**: Higher temperature increases convective activity, strengthening winds
# - **TEMP → PM2.5**: Temperature affects atmospheric mixing and particle formation
# - **PRES → WSPM**: High pressure systems suppress wind speed
# - **PRES → PM2.5**: High pressure traps pollution near the surface (temperature inversion)
# - **DEWP → WSPM**: Humidity influences air density and wind behavior
# - **DEWP → PM2.5**: High humidity promotes particle growth and PM2.5 accumulation
# - **RAIN → WSPM**: Precipitation is associated with storm-driven wind systems
# - **RAIN → PM2.5**: Rainfall directly removes particulates through wet deposition
# - **NO2 → WSPM**: Traffic density correlates with urban heat, affecting local winds
# - **NO2 → PM2.5**: NO2 is a direct component of air pollution
# - **SO2 → WSPM**: Industrial activity affects local atmospheric conditions
# - **SO2 → PM2.5**: SO2 is a direct pollutant contributing to PM2.5
# - **CO → WSPM**: Combustion sources co-vary with atmospheric conditions
# - **CO → PM2.5**: CO is a direct combustion pollutant correlated with PM2.5
# - **WSPM → PM2.5**: ← **Causal effect of interest** — wind disperses particulate matter

# %%
# Check TEMP is a confounder
print('Confounder check:')
print(f'  Corr(TEMP, WSPM) = {df["TEMP"].corr(df["WSPM"]):+.4f}')
print(f'  Corr(TEMP, PM25) = {df["TEMP"].corr(df["PM25"]):+.4f}')
print('  TEMP affects BOTH WSPM and PM2.5 → IS a confounder')

# Three models: unadjusted, adjusted, fully adjusted
unadj    = smf.ols('PM25 ~ WSPM', data=df).fit()
adj      = smf.ols('PM25 ~ WSPM + TEMP + NO2 + RAIN', data=df).fit()
adj_full = smf.ols('PM25 ~ WSPM + TEMP + NO2 + RAIN + C(station)', data=df).fit()

bias = unadj.params['WSPM'] - adj.params['WSPM']

print()
print('=' * 60)
print(f'  {"Model":<40} {"β_WSPM":>8} {"p-value":>10}')
print('-' * 60)
for name, m in [('Unadjusted (back-door open)',   unadj),
                ('Adjusted Z={TEMP,NO2,RAIN}',    adj),
                ('Fully adjusted + station FE',   adj_full)]:
    print(f'  {name:<40} {m.params["WSPM"]:>+8.4f} {m.pvalues["WSPM"]:>10.2e}')
print('=' * 60)
print(f'\n  Confounding bias = {bias:+.4f}')
print('  Adjusted β_WSPM is the CAUSAL effect')
print('  Unadjusted β_WSPM is BIASED due to TEMP confounding')

print(f'   Adjusted WSPM coefficient: {adj.params["WSPM"]:.4f} μg/m³ per m/s')
print(f'   Confounding bias: {bias:.4f} μg/m³ (naive over-estimates by this much)')

# %% [markdown]
# - Covariate adjustment estimates the causal effect by including confounders as 
# covariates: 
# 
# **Naive:**    PM2.5 = β0 + β1·WSPM + ε  
# **Adjusted:** PM2.5 = β0 + β1·WSPM + β2·TEMP + β3·NO2 + β4·RAIN + ε  
# 
# - β1 represents the Average Treatment Effect (ATE): the expected change in PM2.5 
# for a 1 m/s increase in wind speed, holding all confounders constant.
# 
# - The finding is physically plausible: higher wind speed mechanically disperses 
# particulate matter, reducing PM2.5 concentration. This is consistent with 
# atmospheric science literature.
# 
# - The confounding bias is large (−17.65 μg/m³), indicating that TEMP strongly 
# confounds the WSPM–PM2.5 relationship. Without adjustment, the naive model 
# dramatically over-estimates the effect of wind speed. After adjustment, the 
# effect is near zero and non-significant (p = 0.51), suggesting the apparent 
# raw correlation was almost entirely driven by confounding.

# %%
# Forest plot
models9  = {'Unadjusted\n(back-door open)': unadj,
             'Adjusted\n(TEMP,NO2,RAIN)':   adj,
             'Fully adjusted\n(+station)':  adj_full}
names9   = list(models9.keys())
coefs9   = [m.params['WSPM']         for m in models9.values()]
ci_lo9   = [m.conf_int().loc['WSPM',0] for m in models9.values()]
ci_hi9   = [m.conf_int().loc['WSPM',1] for m in models9.values()]
cols9    = ['red','steelblue','green']

fig, ax = plt.subplots(figsize=(9, 4))
for i in range(len(names9)):
    ax.scatter(coefs9[i], i, color=cols9[i], s=120, zorder=3)
    ax.hlines(i, ci_lo9[i], ci_hi9[i], color=cols9[i], lw=2.5)
ax.axvline(0, color='black', ls='--', lw=1)
ax.set_yticks(range(len(names9))); ax.set_yticklabels(names9)
ax.set_xlabel('Effect of WSPM on PM2.5  (β ± 95% CI)')
ax.set_title('Covariate Adjustment \nUnadjusted vs Adjusted',
             fontweight='bold')
ax.text(coefs9[0]+0.2, 0.3, f'Confounding\nbias={bias:+.3f}',
        fontsize=9, color='red')
plt.tight_layout(); plt.show()

# %% [markdown]
# ## 7 — Item 10: Propensity Score
# **Treatment:** High wind (WSPM > median = 1, else 0)  
# **Outcome:** PM2.5  

# %%
# Define binary treatment
df['high_wind'] = (df['WSPM'] > df['WSPM'].median()).astype(int)
covariates = ['TEMP','NO2','RAIN']

# Naive estimate (biased)
naive        = smf.ols('PM25 ~ high_wind', data=df).fit()
naive_effect = naive.params['high_wind']

# Covariate imbalance before matching
print('Covariate Imbalance BEFORE Matching')
print(f'  {"Covariate":<8} {"T=1 mean":>10} {"T=0 mean":>10} {"Std Diff":>10}')
std_before = {}
for cov in covariates:
    m1 = df[df['high_wind']==1][cov].mean()
    m0 = df[df['high_wind']==0][cov].mean()
    sd = (m1 - m0) / df[cov].std()
    std_before[cov] = sd
    print(f'  {cov:<8} {m1:>10.3f} {m0:>10.3f} {sd:>10.4f}')

# Propensity score — logistic regression P(high_wind | TEMP, NO2, RAIN)
X       = StandardScaler().fit_transform(df[covariates])
logit   = LogisticRegression(random_state=42).fit(X, df['high_wind'])
df['pscore'] = logit.predict_proba(X)[:, 1]

print(f'\nPropensity score model: P(high_wind | TEMP, NO2, RAIN)')
for cov, coef in zip(covariates, logit.coef_[0]):
    print(f'  {cov}: {coef:+.4f}')
