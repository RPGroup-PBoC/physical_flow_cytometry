import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob
import flow
import scipy.stats
colors = flow.set_plotting_style()
# %matplotlib inline
###############
FIG_NO = 4
###############

# %% Load the Flow data set.
flow_files = glob.glob('../data/5000uM_IPTG_distributions/*wt*.csv')
flow_df = pd.DataFrame([], columns=['date', 'FITC-H', 'method'])
for f in flow_files:
    date = f.split('/')[-1].split('_')[0]
    _data = pd.read_csv(f, comment='#')

    # Gate and prune the data.
    gated = flow.gaussian_gate(_data, 0.4)
    _df = gated.loc[:, ['FITC-H']]
    _df.insert(0, 'date', date)
    _df.insert(0, 'method', 'flow')
    flow_df = flow_df.append(_df, ignore_index=True)
flow_df.rename(columns={'FITC-H': 'intensity'}, inplace=True)

# Load the microscopy dataset.
mic_files = glob.glob('../data/5000uM_IPTG_distributions/*microscopy*.csv')
mic_df = pd.DataFrame([], columns=['date', 'mean_intensity', 'method'])
for f in mic_files:
    date = f.split('/')[-1].split('_')[0]
    _data = pd.read_csv(f, comment='#')
    _data = _data.loc[(_data['rbs'] == 'RBS1027') & (_data['IPTG_uM'] == 5000)]
    # Filter by size and eccentricity.
    _data = _data.loc[(_data['area'] < 6) & (_data['eccentricity'] > 0.8),
                      ['mean_intensity']]
    _data.insert(0, 'date', date)
    _data.insert(0, 'method', 'microscopy')
    mic_df = mic_df.append(_data, ignore_index=True)
mic_df.rename(columns={'mean_intensity': 'intensity'}, inplace=True)

# Merge the data sets together.
data = pd.concat([flow_df, mic_df], axis=0)


# %%
# Plot the centered distributions
grouped = data.groupby(['method', 'date'])

fig, ax = flow.subplots(1, 2, figsize=(120, 60))
ax[1].set_ylabel('central moment value')
ax[0].set_ylabel('ECDF')
ax[0].set_xlabel('normalized intensity about mean')
ax[0].plot([], [], '-', color=colors[0], label='flow cytometry')
ax[0].plot([], [], '-', color=colors[1], label='microscopy')
ax[1].set_yscale('log')
ax[0].set_xlim([-0.5, 0.8])
ax[0].set_ylim([0, 1.05])

# Add panel labels.
fig.text(0, 0.95, '(A)', fontsize=8)
fig.text(0.5, 0.95, '(B)', fontsize=8)
_ = ax[0].legend(loc='center right', fontsize=6)
for g, d in grouped:
    if g[0] == 'flow':
        c = colors[0]
        alpha = 0.25

    else:
        c = colors[1]
        alpha = 0.75

    # Compute the ECDF and normalize from 0 to 1
    x, y = flow.ecdf(d['intensity'])
    x = (x - x.min()) / (x.max() - x.min())
    _ = ax[0].step(x - x.mean(), y, color=c, alpha=alpha, lw=1.5)

    # Compute the centered normalized moments.
    max_moments = 7
    moments = [scipy.stats.moment(x, i)
               for i in range(1, max_moments)]
    x_vals = np.arange(1, max_moments) + \
        np.random.normal(scale=0.05, size=len(moments))
    _ = ax[1].plot(x_vals, moments, 'o', color=c, alpha=0.5,
                   ms=4)
# ax[1].xaxis.grid(False)
ax[1].set_xticks([1, 2, 3, 4, 5, 6])
ax[1].set_xlim([1.5, 6.5])
_ = ax[1].set_xticklabels(['', 'variance', 'skewness', 'kurtosis', 'hyper-\nskewness',
                           'hyper-\nflatness'])
for t in ax[1].xaxis.get_ticklabels():
    t.set_rotation(45)
plt.tight_layout()
plt.savefig('../figs/fig{0}_moment_comparison.pdf'.format(FIG_NO),
            bbox_inches='tight')
