import numpy as np
import matplotlib.pyplot as plt
import flow
import pandas as pd
import pymc3 as pm
import theano.tensor as tt
import matplotlib.gridspec as gridspec
colors = flow.set_plotting_style()
# %matplotlib inline

###########
FIG_NO = 2
###########
np.random.seed(42)

# Load the data sets.
flow_master = pd.read_csv('../data/flow_master.csv', comment='#')
mic_master = pd.read_csv('../data/microscopy_master.csv', comment='#')

# Only look at O1 and O2.
mic_master = mic_master.loc[mic_master['operator'] != 'O3']

flow_master = flow_master.loc[flow_master['operator'] != 'O3']


# Narrow down the flow master to only three data sets.
flow_pruned = []
grouped = flow_master.groupby('operator')
dates = {'O1': [20160825], 'O2': [20160807, 20160805, 20160809]}
for g, d in grouped:
    # chosen_dates = np.random.choice(d['date'].unique(), 3)
    # print(g, len(d), chosen_dates)
    for date in dates[g]:
        flow_pruned.append(d.loc[d['date'] == date])

flow_master.groupby(['IPTG_uM', 'date', 'operator', 'rbs']).count()

# Selected dates: O1 0825, 0823, 08, 27
# Selected dates: O2 0812, 0810, 0809
flow_date = pd.concat(flow_pruned, axis=0)

#%% Remove the auto and delta.
flow_samp = flow_date.loc[flow_date['rbs'] == 'RBS1027']
mic_samp = mic_master.loc[mic_master['rbs'] == 'RBS1027']

# Remove unnecessary columns
flow_samp = flow_samp.loc[:, ['operator',
                              'binding_energy', 'IPTG_uM', 'fold_change_A']]
flow_samp.rename(columns={'fold_change_A': 'fold_change'}, inplace=True)
flow_samp.insert(0, 'method', 'flow')
mic_samp = mic_samp.loc[:, ['operator',
                            'binding_energy', 'IPTG_uM', 'fold_change']]
mic_samp.insert(0, 'method', 'microscopy')


# %% Set up the MCMC Running each data set separately..
def theano_fc(IPTG, ep_a, ep_i, ep_ra, ep_ai=4.5, R=260, nns=4.6e6, n=2):
    numer = (1 + IPTG / tt.exp(ep_a))**n
    denom = numer + tt.exp(-ep_ai) * (1 + IPTG / tt.exp(ep_i))**n
    pact = numer / denom
    return (1 + pact * (R / nns) * tt.exp(-ep_ra))**-1


with pm.Model() as flow_model:
    # Define the priors.
    ep_a = pm.Normal('ep_a', mu=0, sd=10)
    ep_i = pm.Normal('ep_i', mu=0, sd=10)
    sigma = pm.HalfNormal('sigma', sd=10)

    # Define the expected values
    iptg = flow_samp['IPTG_uM'].values
    ep_ra = flow_samp['binding_energy'].values

    # Compute the probability of an active repressor
    mu = theano_fc(iptg, ep_a, ep_i, ep_ra)

    # Compute the likelihood.
    like = pm.Normal('like', mu=mu, sd=sigma,
                     observed=flow_samp['fold_change'].values, shape=2)
    flow_trace = pm.sample(draws=5000, tune=5000, njobs=4)


with pm.Model() as mic_model:
    # Define the priors.
    ep_a = pm.Normal('ep_a', mu=0, sd=10)
    ep_i = pm.Normal('ep_i', mu=0, sd=10)
    sigma = pm.HalfNormal('sigma', sd=10)

    # Define the expected values
    iptg = mic_samp['IPTG_uM'].values
    ep_ra = mic_samp['binding_energy'].values

    # Compute the probability of an active repressor
    mu = theano_fc(iptg, ep_a, ep_i, ep_ra)

    # Compute the likelihood.
    like = pm.Normal('like', mu=mu, sd=sigma,
                     observed=mic_samp['fold_change'].values, shape=2)
    mic_trace = pm.sample(draws=5000, tune=5000, njobs=4)


# %% Compute the fits and credible regions.
flow_mcmc_df = flow.trace_to_dataframe(flow_trace, flow_model)
flow_stats = flow.compute_statistics(flow_mcmc_df)
mic_mcmc_df = flow.trace_to_dataframe(mic_trace, mic_model)
mic_stats = flow.compute_statistics(mic_mcmc_df)


# %% Plotting.

# Compute the fits and cred regions.
def foldchange(IPTG, ep_a, ep_i, ep_ra, ep_ai=4.5, R=260, nns=4.6e6, n=2):
    numer = (1 + IPTG / np.exp(ep_a))**n
    denom = numer + np.exp(-ep_ai) * (1 + IPTG / np.exp(ep_i))**n
    pact = numer / denom
    return (1 + pact * (R / nns) * np.exp(-ep_ra))**-1


c_range = np.logspace(-2, 4, 500)
flow_cred_region = np.zeros((2, len(c_range), 2))
mic_cred_region = np.zeros((2, len(c_range), 2))
ep_ras = [-13.9, -15.3]
cred_dict = {}
for i, ep in enumerate(ep_ras):
    for j, c in enumerate(c_range):
        flow_val = foldchange(c, flow_mcmc_df['ep_a'], flow_mcmc_df['ep_i'],
                              ep)
        mic_val = foldchange(c, mic_mcmc_df['ep_a'], mic_mcmc_df['ep_i'], ep)
        flow_cred_region[i, j, :] = flow.compute_hpd(flow_val, mass_frac=0.95)
        mic_cred_region[i, j, :] = flow.compute_hpd(mic_val, mass_frac=0.95)

# %%

# Compute the fit with the modes.
c, ep_ra = np.meshgrid(c_range, [-13.9, -15.3])
flow_epa = flow_stats.loc[flow_stats['parameter'] == 'ep_a']['mode'].values
flow_epi = flow_stats.loc[flow_stats['parameter'] == 'ep_i']['mode'].values
mic_epa = mic_stats.loc[mic_stats['parameter'] == 'ep_a']['mode'].values
mic_epi = mic_stats.loc[mic_stats['parameter'] == 'ep_i']['mode'].values
flow_fit = foldchange(c, flow_epa, flow_epi, ep_ra)
mic_fit = foldchange(c, mic_epa, mic_epi, ep_ra)


#%% Generate the plot
plt.close('all')
fig = plt.figure(figsize=(100 / 25.4, 100 / 25.4))

# Set the axes
gs = gridspec.GridSpec(3, 2)
ax1 = fig.add_subplot(gs[0:2, :])
ax2 = fig.add_subplot(gs[2, 0])
ax3 = fig.add_subplot(gs[2, 1])

# fig, ax = flow.subplots(1, 3)
# ax1, ax2, ax3 = ax
# Add labels
ax2.set_yticks([])
ax3.set_yticks([])
ax1.set_xlabel('IPTG [M]')
ax1.set_ylabel('fold-change')
ax1.set_xscale('log')
ax2.set_ylabel('$\propto P\,(K_A\, |\, \mathrm{data})$')
ax2.set_xlabel('$K_A$ [µM]')
ax3.set_ylabel('$\propto P\,(K_I\, |\, \mathrm{data})$')
ax3.set_xlabel('$K_I$ [µM]')

# Add panels
fig.text(0, 0.95, '(A)', fontsize=8)
fig.text(0, 0.33, '(B)', fontsize=8)
fig.text(0.5, 0.33, '(C)', fontsize=8)


# Plot the data and fits.
_ = ax1.plot([], [], 'o', markerfacecolor='w', markeredgecolor='k', ms=5, markeredgewidth=1.5,
             label='flow cytometry data')
_ = ax1.plot([], [], 'ko', ms=6.5, label='microscopy data')
_ = ax1.plot([], [], 's', color=colors[4], label='operator O2')
_ = ax1.plot([], [], 's', color=colors[5], label='operator O1')
_ = ax1.plot([], [], '-k', label='flow cytometry fit')
_ = ax1.plot([], [], ':k', label='microscopy fit')
_ = ax1.legend(loc='upper left', fontsize=6)

# Plot the fits
for i, _ in enumerate(ep_ra):
    # Plot the best fit and credible region.j
    _ = ax1.plot(c_range / 1e6, flow_fit[i], color=colors[i + 4], alpha=0.5)
    _ = ax1.plot(c_range / 1e6, mic_fit[i], ':', color=colors[i + 4])

# Plot the data.
data = pd.concat([flow_samp, mic_samp], axis=0)
grouped = data.groupby(['method', 'binding_energy'])
glyph_colors = {-15.3: colors[5], -13.9: colors[4]}
flow_samp
for g, d in grouped:
    if g[0] == 'flow':
        fill = 'w'
        zorder = 100
    else:
        fill = glyph_colors[g[1]]
        zorder = 1

    _ = ax1.plot(d['IPTG_uM'] / 1e6, d['fold_change'], 'o', markerfacecolor=fill,
                 markeredgecolor=glyph_colors[g[1]], markersize=3.5,
                 markeredgewidth=1, zorder=zorder, alpha=0.75)

ax1.set_xlim([1E-8, 1E-2])
ax1.set_ylim([-0.05, 1.1])


# Plot the posterior distributions
ka_bins = np.linspace(50, 300, 100)
ki_bins = np.linspace(0.35, 0.88, 100)

flow_ka_hist, flow_ka_edges = np.histogram(
    np.exp(flow_mcmc_df['ep_a']), bins=ka_bins)
flow_ka_hist = flow_ka_hist / np.sum(flow_ka_hist)
flow_ki_hist, flow_ki_edges = np.histogram(
    np.exp(flow_mcmc_df['ep_i']), bins=ki_bins)
flow_ki_hist = flow_ki_hist / np.sum(flow_ki_hist)
mic_ka_hist, mic_ka_edges = np.histogram(
    np.exp(mic_mcmc_df['ep_a']), bins=ka_bins)
mic_ka_hist = mic_ka_hist / np.sum(mic_ka_hist)
mic_ki_hist, mic_ki_edges = np.histogram(
    np.exp(mic_mcmc_df['ep_i']), bins=ki_bins)
mic_ki_hist = mic_ki_hist / np.sum(mic_ki_hist)

ax2.step(ka_bins[:-1], mic_ka_hist, color=colors[2], lw=1,
         label='microscopy', alpha=0.75)
ax2.fill_between(ka_bins[:-1], 0, mic_ka_hist, color=colors[2], alpha=0.5,
                 step='pre')
ax2.step(ka_bins[:-1], flow_ka_hist, color=colors[0],
         lw=1, label='flow\ncytometry', alpha=0.75)
ax2.fill_between(ka_bins[:-1], 0, flow_ka_hist, color=colors[0], alpha=0.5,
                 step='pre')

ax2.legend(loc='upper right', fontsize=6, handlelength=1)

ax3.step(ki_bins[:-1], mic_ki_hist, color=colors[2], lw=1, alpha=0.75)
ax3.fill_between(ki_bins[:-1], 0, mic_ki_hist, color=colors[2], alpha=0.5,
                 step='pre')

ax3.step(ki_bins[:-1], flow_ki_hist, color=colors[0], lw=1, alpha=0.75)
ax3.fill_between(ki_bins[:-1], 0, flow_ki_hist, color=colors[0], alpha=0.5,
                 step='pre')

ax2.set_ylim([0, 0.12])
ax2.set_xlim([50, 300])
ax3.set_ylim([0, 0.09])

# Save the figure!
plt.tight_layout()
plt.savefig('../figs/fig{0}_parameter_estimation.pdf'.format(FIG_NO),
            bbox_inches='tight')
