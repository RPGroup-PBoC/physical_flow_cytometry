import numpy as np
import fcsparser
import pandas as pd
from _fit_bivariate_normal_AstroML import fit_bivariate_normal
import matplotlib.pyplot as plt
import seaborn as sns
import pymc3 as pm
import scipy.stats


# #######################
# Automated Gating
# #######################
def fit_2D_gaussian(df, x_val='FSC-H', y_val='SSC-H', log=False):
    '''
    This function hacks astroML fit_bivariate_normal to return the mean
    and covariance matrix when fitting a 2D gaussian fuction to the data
    contained in the x_val and y_val columns of the DataFrame df.

    Parameters
    ----------
    df : DataFrame.
        dataframe containing the data from which to fit the distribution
    x_val, y_val : str.
        name of the dataframe columns to be used in the function
    log : bool.
        indicate if the log of the data should be use for the fit or not

    Returns
    -------
    mu : tuple.
        (x, y) location of the best-fit bivariate normal
    cov : 2 x 2 array
        covariance matrix.
        cov[0, 0] = variance of the x_val column
        cov[1, 1] = variance of the y_val column
        cov[0, 1] = cov[1, 0] = covariance of the data
    '''
    if log:
        x = np.log10(df[x_val])
        y = np.log10(df[y_val])
    else:
        x = df[x_val]
        y = df[y_val]

    # Fit the 2D Gaussian distribution using atroML function
    mu, sigma_1, sigma_2, alpha = fit_bivariate_normal(x, y, robust=True)

    # compute covariance matrix from the standar deviations and the angle
    # that the fit_bivariate_normal function returns
    sigma_xx = ((sigma_1 * np.cos(alpha)) ** 2 +
                (sigma_2 * np.sin(alpha)) ** 2)
    sigma_yy = ((sigma_1 * np.sin(alpha)) ** 2 +
                (sigma_2 * np.cos(alpha)) ** 2)
    sigma_xy = (sigma_1 ** 2 - sigma_2 ** 2) * np.sin(alpha) * np.cos(alpha)

    # put elements of the covariance matrix into an actual matrix
    cov = np.array([[sigma_xx, sigma_xy], [sigma_xy, sigma_yy]])

    return mu, cov


# #################
def gauss_interval(df, mu, cov, x_val='FSC-H', y_val='SSC-H', log=False):
    '''
    Computes the of the statistic

    (x - µx)'Σ(x - µx)

    for each of the elements in df columns x_val and y_val.

    Parameters
    ----------
    df : DataFrame.
        dataframe containing the data from which to fit the distribution
    mu : array-like.
        (x, y) location of bivariate normal
    cov : 2 x 2 array
        covariance matrix
    x_val, y_val : str.
        name of the dataframe columns to be used in the function
    log : bool.
        indicate if the log of the data should be use for the fit or not.

    Returns
    -------
    statistic_gauss : array-like.
        array containing the result of the linear algebra operation:
        (x - µx)'sum(x - µx)
    '''
    # Determine that the covariance matrix is not singular
    det = np.linalg.det(cov)
    if det == 0:
        raise NameError("The covariance matrix can't be singular")

    # Compute the vector x defined as [[x - mu_x], [y - mu_y]]
    if log is True:
        x_vect = np.log10(np.array(df[[x_val, y_val]]))
    else:
        x_vect = np.array(df[[x_val, y_val]])

    x_vect[:, 0] = x_vect[:, 0] - mu[0]
    x_vect[:, 1] = x_vect[:, 1] - mu[1]

    # compute the inverse of the covariance matrix
    inv_sigma = np.linalg.inv(cov)

    # compute the operation
    interval_array = np.zeros(len(df))
    for i, x in enumerate(x_vect):
        interval_array[i] = np.dot(np.dot(x, inv_sigma), x.T)

    return interval_array


def gaussian_gate(df, alpha, x_val='FSC-H', y_val='SSC-H', log=False,
                  verbose=False):
    '''
    Function that applies an "unsupervised bivariate Gaussian gate" to the data
    over the channels x_val and y_val.

    Parameters
    ----------
    df : DataFrame.
        dataframe containing the data from which to fit the distribution
    alpha : float. [0, 1]
        fraction of data aimed to keep. Used to compute the chi^2 quantile
        function
    x_val, y_val : str.
        name of the dataframe columns to be used in the function
    log : bool.
        indicate if the log of the data should be use for the fit or not
    verbose : bool.
        indicate if the percentage of data kept should be print

    Returns
    -------
    df_thresh : DataFrame
        Pandas data frame to which the automatic gate was applied.
    '''

    # Perform sanity checks.
    if alpha < 0 or alpha > 1:
        return RuntimeError("`alpha` must be a float between 0 and 1.")

    data = df[[x_val, y_val]]
    # Fit the bivariate Gaussian distribution
    mu, cov = fit_2D_gaussian(data, log=log, x_val=x_val, y_val=y_val)

    # Compute the statistic for each of the pair of log scattering data
    interval_array = gauss_interval(data, mu, cov, log=log,
                                    x_val=x_val, y_val=y_val)

    # Find which data points fall inside the interval
    idx = interval_array <= scipy.stats.chi2.ppf(alpha, 2)

    # print the percentage of data kept
    if verbose:
        print('''
        with parameter alpha={0:0.2f}, percentage of data kept = {1:0.2f}
        '''.format(alpha, np.sum(idx) / len(df)))
    return df[idx]


# #######################################################
# Plotting Utilities
# #######################################################
def set_plotting_style(return_colors=True):
    """
    Sets the plotting style for a matplotlib backend. To use, simply call in the preamble of your
    notebook or script.

    >>> set_plotting_style()

    Parameters
    ----------
    return_colors: Bool
        If True, this will also return a palette of eight color-blind safe
        colors with the hideous yellow replaced by 'dusty purple.'
    """
    rc = {'axes.facecolor': '#E3DCD1',
          'font.family': 'Lucida Sans Unicode',
          'axes.labelsize': 8,
          'font.size': 8,
          'xtick.labelsize': 8,
          'ytick.labelsize': 8,
          'grid.linestyle': '-',
          'grid.linewidth': 0.5,
          'grid.alpha': 0.75,
          'grid.color': '#ffffff',
          'mathtext.fontset': 'stixsans',
          'mathtext.sf': 'sans',
          'legend.frameon': True,
          'legend.facecolor': '#FFEDCE',
          'figure.dpi': 150}
    plt.rc('text.latex', preamble=r'\usepackage{sfmath}')
    plt.rc('mathtext', fontset='stixsans', sf='sans')
    sns.set(rc=rc)
    colors = sns.color_palette('colorblind', n_colors=8)
    colors[4] = sns.xkcd_palette(['dusty purple'])[0]

    if return_colors:
        return colors


def subplots(nrows, ncols, figsize=(120, 80), **kwargs):
    R"""
    Generates a matplotlib subplot figure canvas with the size in mm.
    """
    _figsize = (figsize[0] / 25.4, figsize[1] / 25.4)
    fig, ax = plt.subplots(nrows, ncols, figsize=_figsize, **kwargs)
    return fig, ax

# ######################################################
# Statistics
# ######################################################


def ecdf(data):
    R"""
    Computes the empirical Cumulative Distribution Function of a given range
    of data.

    Parameters
    ----------
    data : 1d-array or pandas Series
        The data from which the ECDF should be computed

    Returns
    -------
    x, y : 1d-arrays
        The sorted data and the empirical CDF. These are of the same length as
        data.
    """
    n = len(data)
    x = np.sort(data)
    y = np.arange(0, n) / n
    return [x, y]


# ######################################################
# Image Processing Utilities
# ######################################################
def ome_split(im):
    """Splits an ome.tiff image into individual channels"""
    if len(np.shape(im)) != 3:
        raise RuntimeError('provided image must be a single image')
    ims = []
    for i in range(np.shape(im)[-1]):
        ims.append(im[:, :, i])
    return ims


def projection(im, mode='mean', median_filt=True):
    R"""
    Computes an average image from a provided array of images.

    Parameters
    ----------
    im : list or arrays of 2d-arrays
        Stack of images to be filtered.
    mode : string ('mean', 'median', 'min', 'max')
        Type of elementwise projection.
    median_filt : bool
        If True, each image will be median filtered before averaging.
        Median filtering is performed using a 3x3 square structural element.

    Returns
    -------
    im_avg : 2d-array
        Projected image with a type of int.
    """
    # Determine if the images should be median filtered.
    if median_filt is True:
        selem = skimage.morphology.square(3)
        im_filt = [scipy.ndimage.median_filter(i, footprint=selem) for i in im]
        im = im_filt
    # Get the image type
    im_type = im[0].dtype
    # Determine and perform the projection.
    if mode is 'mean':
        im_proj = np.mean(im, axis=0)
    elif mode is 'median':
        im_proj = np.median(im, axis=0)
    elif mode is 'min':
        im_proj = np.min(im, axis=0)
    elif mode is 'max':
        im_proj = np.max(im, axis=0)
    return im_proj.astype(im_type)


def generate_flatfield(im, im_dark, im_field, median_filt=True):
    """
    Corrects illumination of a given image using a dark image and an image of
    the flat illumination.

    Parameters
    ----------
    im : 2d-array
        Image to be flattened.
    im_field: 2d-array
        Average image of fluorescence illumination.
    median_filt : bool
        If True, the image to be corrected will be median filtered with a
        3x3 square structural element.

    Returns
    -------
    im_flat : 2d-array
        Image corrected for uneven fluorescence illumination. This is performed
        as

        im_flat = ((im - im_dark) / (im_field - im_dark)) *
                   mean(im_field - im_dark)

    Raises
    ------
    RuntimeError
        Thrown if bright image and dark image are approximately equal. This
        will result in a division by zero.
    """

    # Compute the mean field image.
    mean_diff = np.mean(im_field)

    if median_filt is True:
        selem = skimage.morphology.square(3)
        im_filt = scipy.ndimage.median_filter(im, footprint=selem)
    else:
        im_filt = im

    # Compute and return the flattened image.
    im_flat = (im_filt - im_dark) / (im_field - im_dark) * mean_diff
    return im_flat


def find_zero_crossings(im, selem, thresh):
    """
    This  function computes the gradients in pixel values of an image after
    applying a sobel filter to a given image. This  function is later used in
    the Laplacian of Gaussian cell segmenter (log_segmentation) function. The
    arguments are as follows.

    Parameters
    ----------
    im : 2d-array
        Image to be filtered.
    selem : 2d-array, bool
        Structural element used to compute gradients.
    thresh :  float
        Threshold to define gradients.

    Returns
    -------
    zero_cross : 2d-array
        Image with identified zero-crossings.

    Notes
    -----
    This function as well as `log_segmentation` were written by Justin Bois.
    http://bebi103.caltech.edu/
    """

    # apply a maximum and minimum filter to the image.
    im_max = scipy.ndimage.filters.maximum_filter(im, footprint=selem)
    im_min = scipy.ndimage.filters.minimum_filter(im, footprint=selem)

    # Compute the gradients using a sobel filter.
    im_filt = skimage.filters.sobel(im)

    # Find the zero crossings.
    zero_cross = (((im >= 0) & (im_min < 0)) | ((im <= 0) & (im_max > 0)))\
        & (im_filt >= thresh)

    return zero_cross


def log_segmentation(im, selem='default', thresh=0.0001, radius=2.0,
                     median_filt=True, clear_border=True, label=False):
    """
    This function computes the Laplacian of a gaussian filtered image and
    detects object edges as regions which cross zero in the derivative.

    Parameters
    ----------
    im :  2d-array
        Image to be processed. Must be a single channel image.
    selem : 2d-array, bool
        Structural element for identifying zero crossings. Default value is
        a 2x2 pixel square.
    radius : float
        Radius for gaussian filter prior to computation of derivatives.
    median_filt : bool
        If True, the input image will be median filtered with a 3x3 structural
        element prior to segmentation.
    selem : 2d-array, bool
        Structural element to be applied for laplacian calculation.
    thresh : float
        Threshold past which
    clear_border : bool
        If True, segmented objects touching the border will be removed.
        Default is True.
    label : bool
        If True, segmented objecs will be labeled. Default is False.

    Returns
    -------
    im_final : 2d-array
        Final segmentation mask. If label==True, the output will be a integer
        labeled image. If label==False, the output will be a bool.

    Notes
    -----
    We thank Justin Bois in his help writing this function.
    https://bebi103.caltech.edu
    """

    # Test that the provided image is only 2-d.
    if len(np.shape(im)) > 2:
        raise ValueError('image must be a single channel!')

    # Determine if the image should be median filtered.
    if median_filt is True:
        selem = skimage.morphology.square(3)
        im_filt = scipy.ndimage.median_filter(im, footprint=selem)
    else:
        im_filt = im
    # Ensure that the provided image is a float.
    if np.max(im) > 1.0:
        im_float = skimage.img_as_float(im_filt)
    else:
        im_float = im_filt

    # Compute the LoG filter of the image.
    im_LoG = scipy.ndimage.filters.gaussian_laplace(im_float, radius)

    # Define the structural element.
    if selem is 'default':
        selem = skimage.morphology.square(3)

    # Using find_zero_crossings, identify the edges of objects.
    edges = find_zero_crossings(im_LoG, selem, thresh)

    # Skeletonize the edges to a line with a single pixel width.
    skel_im = skimage.morphology.skeletonize(edges)

    # Fill the holes to generate binary image.
    im_fill = scipy.ndimage.morphology.binary_fill_holes(skel_im)

    # Remove small objects and objects touching border.
    im_final = skimage.morphology.remove_small_objects(im_fill)
    if clear_border is True:
        im_final = skimage.segmentation.clear_border(im_final, buffer_size=5)

    # Determine if the objects should be labeled.
    if label is True:
        im_final = skimage.measure.label(im_final)

    return im_final


# #####################################################
# MCMC Manipulation
# #####################################################

def _log_prior_trace(trace, model):
    """
    Computes the contribution of the log prior to the log posterior.

    Parameters
    ----------
    trace : PyMC3 trace object.
        Trace from the PyMC3 sampling.
    model : PyMC3 model object
        Model under which the sampling was performed

    Returns
    -------
    log_prior_vals : nd-array
        Array of log-prior values computed elementwise for each point in the
        trace.

    Notes
    -----
    This function was modified from one produced by Justin Bois.
    http://bebi103.caltech.edu
    """
    # Iterate through each trace.
    try:
        points = trace.points()
    except:
        points = trace

    # Get the unobserved variables.
    priors = [var.logp for var in model.unobserved_RVs if type(
        var) == pm.model.FreeRV]

    def logp_vals(pt):
        if len(model.unobserved_RVs) == 0:
            return pm.theanof.floatX(np.array([]), dtype='d')

        return np.array([logp(pt) for logp in priors])

    # Compute the logp for each value of the prior.
    log_prior = (logp_vals(pt) for pt in points)
    return np.stack(log_prior)


def _log_post_trace(trace, model):
    R"""
    Computes the log posterior of a PyMC3 sampling trace.

    Parameters
    ----------
    trace : PyMC3 trace object
        Trace from MCMC sampling
    model: PyMC3 model object
        Model under which the sampling was performed.

    Returns
    -------
    log_post : nd-array
        Array of log posterior values computed elementwise for each point in
        the trace

    Notes
    -----
    This function was modified from one produced by Justin Bois
    http://bebi103.caltech.edu
    """

    # Compute the log likelihood. Note this is improperly named in PyMC3.
    log_like = pm.stats._log_post_trace(trace, model).sum(axis=1)

    # Compute the log prior
    log_prior = _log_prior_trace(trace, model)

    return (log_prior.sum(axis=1) + log_like)


def trace_to_dataframe(trace, model):
    R"""
    Converts a PyMC3 sampling trace object to a pandas DataFrame

    Parameters
    ----------
    trace, model: PyMC3 sampling objects.
        The MCMC sampling trace and the model context.

    Returns
    -------
    df : pandas DataFrame
        A tidy data frame containing the sampling trace for each variable  and
        the computed log posterior at each point.
    """

    # Use the Pymc3 utilitity.
    df = pm.trace_to_dataframe(trace)

    # Include the log prop
    df['logp'] = _log_post_trace(trace, model)

    return df


def compute_statistics(df, varnames=None, logprob_name='logp'):
    R"""
    Computes the mode, hpd_min, and hpd_max from a pandas DataFrame. The value
    of the log posterior must be included in the DataFrame.
    """

    # Get the vars we care about.
    if varnames is None:
        varnames = [v for v in df.keys() if v is not 'logp']

    # Find the max of the log posterior.
    ind = np.argmax(df[logprob_name])
    if type(ind) is not int:
        ind = ind[0]

    # Instantiate the dataframe for the parameters.
    stat_df = pd.DataFrame([], columns=['parameter', 'mode', 'hpd_min',
                                        'hpd_max'])
    for v in varnames:
        mode = df.iloc[ind][v]
        hpd_min, hpd_max = compute_hpd(df[v].values, mass_frac=0.95)
        stat_dict = dict(parameter=v, mode=mode, hpd_min=hpd_min,
                         hpd_max=hpd_max)
        stat_df = stat_df.append(stat_dict, ignore_index=True)

    return stat_df


def compute_hpd(trace, mass_frac):
    R"""
    Returns highest probability density region given by
    a set of samples.

    Parameters
    ----------
    trace : array
        1D array of MCMC samples for a single variable
    mass_frac : float with 0 < mass_frac <= 1
        The fraction of the probability to be included in
        the HPD.  For hreple, `massfrac` = 0.95 gives a
        95% HPD.

    Returns
    -------
    output : array, shape (2,)
        The bounds of the HPD

    Notes
    -----
    We thank Justin Bois (BBE, Caltech) for developing this function.
    http://bebi103.caltech.edu/2015/tutorials/l06_credible_regions.html
    """
    # Get sorted list
    d = np.sort(np.copy(trace))

    # Number of total samples taken
    n = len(trace)

    # Get number of samples that should be included in HPD
    n_samples = np.floor(mass_frac * n).astype(int)

    # Get width (in units of data) of all intervals with n_samples samples
    int_width = d[n_samples:] - d[:n - n_samples]

    # Pick out minimal interval
    min_int = np.argmin(int_width)

    # Return interval
    return np.array([d[min_int], d[min_int + n_samples]])
