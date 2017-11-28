import numpy as np
import matplotlib.pyplot as plt
import flow
import skimage.io
import skimage.filters
import skimage.morphology
import glob
import pandas as pd
colors = flow.set_plotting_style()
import imp
imp.reload(flow)
#% matplotlib inline
############
FIG_NO = 1
############

# Load a flow dataset.
flow_files = glob.glob('../data/flow/*.csv')
flow_data = pd.read_csv(flow_files[-1], comment="#")

# Gate the data.
gated = flow.gaussian_gate(flow_data, 0.4, x_val='FSC-A', y_val='SSC-A',
                           log=True)
gated.head()
# Show the raw data and the gated.
fig, ax = flow.subplots(1, 1, figsize=(35, 32))
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('FSC-A (a.u.)')
ax.set_ylabel('SSC-A (a.u.)')

# Set the legend with visible points.
ax.plot([], [], 'k.', label='discarded')
ax.plot([], [], '.', color=colors[0], label='selected')
ax.legend(loc='lower right', fontsize=6)

# Plot the data and the gates.
_ = ax.plot(flow_data['FSC-A'], flow_data['SSC-A'], 'k,', rasterized=True)
_ = ax.plot(gated['FSC-A'].values, gated['SSC-A'].values, ',', color=colors[0],
            rasterized=True)
ax.set_xlim([1E3, 1E5])
ax.set_xticks([])
ax.set_yticks([])
# plt.tight_layout()
plt.savefig(
    '../figs/fig{0}A_flow_cloud.pdf'.format(FIG_NO), bbox_inches='tight')

# %%  Show a representative segmentation.

im = skimage.io.imread('../data/example_image.ome.tif')
phase = im[:, :, 0]
mcherry = im[:, :, 2]

# Segment using LoG
selem = skimage.morphology.square(3)
filt = skimage.filters.median(mcherry, selem)
seg = flow.log_segmentation(filt, label=True)
blank_im = np.zeros_like(seg)
props = skimage.measure.regionprops(seg)
for prop in props:
    # Conversion to µm^2 using known pixel distance.
    area = prop.area * 0.16**2
    if (area < 6) & (area > 0.5) & (prop.eccentricity > 0.8):
        blank_im += (seg == prop.label)

# Shade the segmented cells.
contours = skimage.measure.find_contours(blank_im, 0)
fig, ax = flow.subplots(1, 1, figsize=(35, 32))
ax.imshow(phase, cmap=plt.cm.Greys_r)
for c in contours:
    _ = ax.plot(c[:, 1], c[:, 0], color='tomato', alpha=0.75, lw=0.1)
    _ = ax.fill(c[:, 1], c[:, 0], color='tomato', alpha=0.5)


# Add a scale bar for posterity.
_ = ax.set_frame_on(False)
ax.hlines(-10, 0, 10 / 0.16, lw=1)
_ = ax.set_xticks([])
_ = ax.set_yticks([])
plt.tight_layout()
plt.savefig(
    '../figs/fig{0}B_segmentation.pdf'.format(FIG_NO), bbox_inches='tight')
