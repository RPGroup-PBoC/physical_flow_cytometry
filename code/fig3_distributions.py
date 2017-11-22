import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import pandas as pd
import flow
_ = flow.set_plotting_style()
%matplotlib inline

###########
FIG_NO = 3
###########

# Load example data sets.
flow_sets = np.sort(glob.glob('../data/flow/*.csv'))
flow_df = pd.DataFrame([], columns=['IPTG_uM', 'FITC-H', 'FSC-H', 'SSC-H'])
for f in flow_sets:
    _, _, _, RBS, IPTG = f.split('/')[-1].split('_')
    IPTG = float(IPTG.split('uM')[0])

    # Gate the data.
    _df = pd.read_csv(f, comment='#')
    gated = flow.gaussian_gate(_df, alpha=0.4)

    # Slice the gated data.
    _df = gated.loc[:, ['FITC-H', 'FSC-H', 'SSC-H']]
    _df.insert(0, 'IPTG_uM', IPTG)
    flow_df = flow_df.append(_df, ignore_index=True)

mic_sets = glob.glob('../data/microscopy/20161018*.csv')
mic_df = pd.DataFrame([], columns=['date', 'IPTG_uM', 'mean_intensity'])
for f in mic_sets:
    _df = pd.read_csv(f, comment='#')
    _df = _df[(_df['area'] < 6) & (_df['eccentricity'] > 0.85)]
    _df = _df.loc[:, ['date', 'IPTG_uM', 'rbs', 'mean_intensity']]

    mic_df = mic_df.append(_df, ignore_index=True)


mic_df = mic_df.loc[mic_df['rbs'] == 'RBS1027']
# set the bins.
flow_bins = np.logspace(3, 5, 100)
mic_bins = np.logspace(0, 4, 100)

# %% Set up a figure.
fig, ax = flow.subplots(2, 2, figsize=(120, 80))
_ax = ax.ravel()
for a in _ax:
    a.set_xscale('log')
    a.set_xlabel('intensity (a.u.)')

ax[0, 0].set_ylabel('frequency')
ax[0, 1].set_ylabel('frequency')
ax[1, 0].set_ylabel('ECDF')
ax[1, 1].set_ylabel('ECDF')
ax[0, 0].set_title('flow cytometry', backgroundcolor='#FFEDCE',
                   fontsize=8, y=1.01)
ax[0, 1].set_title('microscopy', backgroundcolor='#FFEDCE',
                   fontsize=8, y=1.01)

# Add panel labels.
fig.text(0, 0.95, '(A)')
fig.text(0.5, 0.95, '(B)')
grouped_flow = flow_df.groupby('IPTG_uM')
grouped_flow = flow_df.groupby('IPTG_uM')
colors = sns.color_palette('Blues', n_colors=14)
i = 0
for g, d in grouped_flow:
    hist, edges = np.histogram(d['FITC-H'], bins=flow_bins,
                               normed=True)
    x, y = flow.ecdf(d['FITC-H'])
    _ = ax[0, 0].step(edges[:-1], hist, color='k', linewidth=0.5, alpha=0.2)
    _ = ax[0, 0].fill_between(edges[:-1], 0, hist, color=colors[i], alpha=0.5,
                              step='pre')
    _ = ax[1, 0].step(x, y, color=colors[i])

    i += 1

# Microscopy distributions
grouped_mic = mic_df.groupby('IPTG_uM')
i = 0
for g, d in grouped_mic:
    hist, edges = np.histogram(d['mean_intensity'], bins=mic_bins,
                               normed=True)
    x, y = flow.ecdf(d['mean_intensity'])
    _ = ax[0, 1].step(edges[:-1], hist, color='k', linewidth=0.5, alpha=0.2)
    _ = ax[0, 1].fill_between(edges[:-1], 0, hist, color=colors[i], alpha=0.6,
                              step='pre')

    _ = ax[1, 1].step(x, y, color=colors[i])

    i += 1

ax[0, 0].ticklabel_format(style='sci', axis='y', scilimits=(0, 4))
ax[1, 0].set_xlim([1E3, 1E5])
ax[0, 1].set_xlim([10, 2.5E3])
ax[0, 0].yaxis.major.formatter._useMathText = True
plt.tight_layout()
plt.savefig(
    '../figs/fig{0}_distributions.pdf'.format(FIG_NO), bbox_inches='tight')
