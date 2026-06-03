"""
Paper 2 figures v2: full DCC coverage, narrative-quality charts, no source lines.
"""
import numpy as np
import pandas as pd
import pickle, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings; warnings.filterwarnings('ignore')

OUT = 'output'
os.makedirs(OUT, exist_ok=True)

with open(f'{OUT}/paper2_results.pkl','rb') as f:
    r = pickle.load(f)

data      = r['data']
garch_res = r['garch_results']
dcc_A     = r['dcc_A']
dy_bad_A  = r['dy_bad_A']
dy_good_A = r['dy_good_A']
roll_A    = r['rolling_spillover_A']

dates = data.index
pre   = dates < '2020-02'
covid = (dates >= '2020-02') & (dates <= '2022-12')
post  = dates > '2022-12'
rho_A = dcc_A['corr_ts'][:, 0, 1]

# Verify
print(f'dates: {dates[0]} to {dates[-1]}, n={len(dates)}')
print(f'rho_A: n={len(rho_A)}, range=[{rho_A.min():.3f}, {rho_A.max():.3f}]')
print(f'rolling: {roll_A["dates"][0]} to {roll_A["dates"][-1]}, n={len(roll_A["dates"])}')

plt.rcParams.update({
    'font.family':     'DejaVu Sans',
    'font.size':       10,
    'axes.titlesize':  10,
    'axes.labelsize':  10,
    'legend.fontsize':  8,
    'axes.spines.top':  False,
    'axes.spines.right':False,
    'axes.grid':        True,
    'grid.alpha':       0.20,
    'grid.linestyle':   '--',
    'figure.dpi':       150,
})
BLUE   = '#1f4e79'
ORANGE = '#c55a11'
GRAY   = '#595959'
RED    = '#c00000'
GREEN  = '#375623'

def shade_events(ax):
    ax.axvspan(pd.Timestamp('2020-02'), pd.Timestamp('2022-12'),
               alpha=0.12, color='red',    zorder=0)
    ax.axvspan(pd.Timestamp('2008-10'), pd.Timestamp('2009-06'),
               alpha=0.10, color='orange', zorder=0)

def fmt_xaxis(ax):
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.set_xlim(dates[0], dates[-1])

# ── Figure 1: overview ──────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, sharex=True)
fig.set_figwidth(6.5)

ax = axes[0]
ax.fill_between(dates, data['ovn_share'], alpha=0.25, color=BLUE)
ax.plot(dates, data['ovn_share'], color=BLUE, lw=1.2, label='Overnight share')
shade_events(ax)
ax.axhline(50, color=GRAY, lw=0.8, ls='--', alpha=0.5, label='50% line')
ax.set_ylabel('Overnight share (%)')
ax.set_title('(a) Overnight visitors as share of total monthly arrivals')
ax.legend(fontsize=7)
ax.set_ylim(30, 65)

ax = axes[1]
ax.bar(dates, data['r_overnight'], color=np.where(data['r_overnight']<0, RED, BLUE),
       width=20, alpha=0.65, label='Overnight visitors')
ax.bar(dates, data['r_gaming'],    color=np.where(data['r_gaming']<0, '#8B0000', ORANGE),
       width=20, alpha=0.55, label='Gaming revenue')
ax.axhline(0, color='black', lw=0.5)
shade_events(ax)
ax.set_ylabel('Log-return (%)')
ax.set_title('(b) Monthly log-returns of overnight visitors and gaming revenue')
ax.legend(fontsize=7)

fmt_xaxis(axes[1])
plt.tight_layout()
plt.savefig(f'{OUT}/fig1_series_overview.png', dpi=150, bbox_inches='tight')
plt.close()
print('Fig 1 saved.')

# ── Figure 2: conditional volatility ────────────────────────────────────────
fig, axes = plt.subplots(2, 1, sharex=True)
fig.set_figwidth(6.5)

for i, (name, col, color) in enumerate([
    ('Overnight visitors', 'r_overnight', BLUE),
    ('Gaming revenue',     'r_gaming',    ORANGE),
]):
    ax = axes[i]
    cv = garch_res[name]['cond_vol']
    ax.fill_between(dates, 0, cv, alpha=0.40, color=color)
    ax.plot(dates, cv, color=color, lw=1.0, label='Cond. std \u03c3\u209c')
    ax.plot(dates, np.abs(data[col].values), color=GRAY,
            lw=0.5, alpha=0.40, label='|Log-return|')
    shade_events(ax)
    ax.set_ylabel('Cond. std (%)')
    ax.set_title(f'({chr(97+i)}) {name}')
    ax.legend(fontsize=7, loc='upper left')

fmt_xaxis(axes[1])
axes[1].set_xlabel('Year')
plt.tight_layout()
plt.savefig(f'{OUT}/fig2_cond_vol.png', dpi=150, bbox_inches='tight')
plt.close()
print('Fig 2 saved.')

# ── Figure 3: DCC correlation ──────────────────────────────────────────────
# FULL SERIES from 2008-03 through 2025-07 (209 obs)
fig, ax = plt.subplots()
fig.set_figwidth(6.5)

# Plot the full DCC series — confirmed 209 points starting 2008-03
ax.plot(dates, rho_A, color=BLUE, lw=1.6, zorder=5, label='DCC \u03c1\u209c')

# Horizontal zero line for reference
ax.axhline(0, color='black', lw=0.8, ls='-', alpha=0.4, zorder=4)

# Period mean lines
ax.axhline(rho_A[pre].mean(),   color=GREEN,  lw=1.0, ls='--', alpha=0.8,
           label=f'Pre-COVID mean ({rho_A[pre].mean():.3f})')
ax.axhline(rho_A[covid].mean(), color=RED,    lw=1.0, ls=':',  alpha=0.8,
           label=f'COVID mean ({rho_A[covid].mean():.3f})')
ax.axhline(rho_A[post].mean(),  color=ORANGE, lw=1.0, ls='-.', alpha=0.8,
           label=f'Post-COVID mean ({rho_A[post].mean():.3f})')

# Background shading — drawn AFTER the line so zorder matters
ax.axvspan(pd.Timestamp('2020-02'), pd.Timestamp('2022-12'),
           alpha=0.12, color='red',    zorder=2, label='COVID-19')
ax.axvspan(pd.Timestamp('2008-10'), pd.Timestamp('2009-06'),
           alpha=0.10, color='orange', zorder=2, label='GFC')

ax.set_ylabel('Dynamic conditional correlation \u03c1\u209c')
ax.set_xlabel('Year')
ax.set_title('Time-varying correlation between overnight visitors and gaming revenue')
fmt_xaxis(ax)
ax.set_ylim(min(rho_A.min() - 0.05, -0.25), max(rho_A.max() + 0.05, 0.35))
ax.legend(fontsize=7, loc='upper left', ncol=2)

# Annotate three regimes
ax.annotate('Pre-COVID\npositive link', xy=(pd.Timestamp('2015-01'), 0.22),
            fontsize=7, color=GREEN, ha='center')
ax.annotate('Structural\nbreak', xy=(pd.Timestamp('2021-01'), -0.08),
            fontsize=7, color=RED, ha='center')
ax.annotate('Post-COVID\nnegative', xy=(pd.Timestamp('2024-06'), -0.12),
            fontsize=7, color=ORANGE, ha='center')

plt.tight_layout()
plt.savefig(f'{OUT}/fig3_dcc_correlations.png', dpi=150, bbox_inches='tight')
plt.close()
print(f'Fig 3 saved. rho_A first={rho_A[0]:.4f}, at 2008-10={rho_A[7]:.4f}')

# ── Figure 4: rolling spillover ──────────────────────────────────────────────
fig, ax = plt.subplots()
fig.set_figwidth(6.5)

roll_dates  = pd.DatetimeIndex(roll_A['dates'])
roll_values = np.array(roll_A['values'], dtype=float)
full_mean   = np.nanmean(roll_values)

ax.fill_between(roll_dates, full_mean, roll_values,
                where=roll_values > full_mean,
                alpha=0.35, color=BLUE,   label='Above average integration')
ax.fill_between(roll_dates, full_mean, roll_values,
                where=roll_values <= full_mean,
                alpha=0.35, color=ORANGE, label='Below average integration')
ax.plot(roll_dates, roll_values, color=BLUE, lw=1.4, zorder=5)
ax.axhline(full_mean, color=GRAY, lw=1.0, ls='--',
           label=f'Full-sample mean ({full_mean:.1f}%)')

ax.axvspan(pd.Timestamp('2020-02'), pd.Timestamp('2022-12'),
           alpha=0.12, color='red', label='COVID-19')
ax.set_ylabel('Total Spillover Index (%)')
ax.set_xlabel('Year')
ax.set_title('Rolling Diebold-Yilmaz Total Spillover Index (60-month window)')
ax.legend(fontsize=7)
# Note: rolling starts Feb 2013 by construction (60-month burn-in)
ax.annotate('60-month window\nstarts Feb 2013', xy=(pd.Timestamp('2013-06'), 26),
            fontsize=7, color=GRAY, ha='left')

plt.tight_layout()
plt.savefig(f'{OUT}/fig4_rolling_spillover.png', dpi=150, bbox_inches='tight')
plt.close()
print('Fig 4 saved.')

# ── Figure 5: asymmetric spillover ──────────────────────────────────────────
fig, ax = plt.subplots()
fig.set_figwidth(5.5)

labels    = ['Total\nSpillover\nIndex', 'Overnight\n\u2192 Gaming', 'Gaming\n\u2192 Overnight']
bad_vals  = [
    dy_bad_A['total_spillover'],
    dy_bad_A['theta'][1,0]*100,
    dy_bad_A['theta'][0,1]*100,
]
good_vals = [
    dy_good_A['total_spillover'],
    dy_good_A['theta'][1,0]*100,
    dy_good_A['theta'][0,1]*100,
]

x = np.arange(len(labels))
w = 0.35
b1 = ax.bar(x - w/2, bad_vals,  w, label='Bad volatility',  color=RED,  alpha=0.85)
b2 = ax.bar(x + w/2, good_vals, w, label='Good volatility', color=BLUE, alpha=0.85)

for bar in list(b1) + list(b2):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.6,
            f'{bar.get_height():.1f}%', ha='center', va='bottom', fontsize=8)

# Asymmetry gap arrows for the first bar
ax.annotate('', xy=(x[0]-w/2, bad_vals[0]), xytext=(x[0]+w/2, good_vals[0]),
            arrowprops=dict(arrowstyle='<->', color='black', lw=1.2))
ax.text(x[0], (bad_vals[0]+good_vals[0])/2 + 2, f'+{bad_vals[0]-good_vals[0]:.1f}pp\ngap',
        ha='center', fontsize=7.5, color='black')

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel('Spillover index (%)')
ax.set_title('Bad vs good volatility spillovers')
ax.legend(fontsize=8)
ax.set_ylim(0, 75)
plt.tight_layout()
plt.savefig(f'{OUT}/fig5_asymmetric_spillover.png', dpi=150, bbox_inches='tight')
plt.close()
print('Fig 5 saved.')
print('All figures done.')
