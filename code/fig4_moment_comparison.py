import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob
import flow
import scipy.stats
colors = flow.set_plotting_style()
#%matplotlib inline
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
# Plot the centered distributions.
grouped = data.groupby(['method', 'date'])
ax.set_ylabel('central moment value')
fig, ax = flow.subplots(1, 1, figsize=(100, 80))
ax.plot([], [], 'o', color=colors[0], label='flow cytometry')
ax.plot([], [], 'o', color=colors[1], label='microscopy')
ax.set_yscale('log')

_ = ax.legend(loc='upper left', fontsize=6)
for g, d in grouped:
    # Compute the ECDFs.
    x, y = flow.ecdf(d['intensity'])
    x = x - x.mean()
    if g[0] == 'flow':
        c = colors[0]
    else:
        c = colors[1]

    max_moments = 7
    moments = [scipy.stats.moment(d['intensity'], i)
               for i in range(1, max_moments)]
    x_vals = np.arange(1, max_moments) + np.random.normal(scale=0.05)
    _ = ax.plot(x_vals, moments, 'o', color=c, alpha=0.5,
                ms=4)
ax.xaxis.grid(False)
ax.set_xticks([1, 2, 3, 4, 5, 6])
ax.set_xlim([1.5, 6.5])
_ = ax.set_xticklabels(['', 'variance', 'skewness', 'kurtosis', 'hyperskewness',
                        'hyperflatness'])
for t in ax.xaxis.get_ticklabels():
    t.set_rotation(15)


# Compute the first three moments of the distributions.
