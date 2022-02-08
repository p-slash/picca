"""This module defines the class Dr16ExpectedFlux"""
import logging
import multiprocessing

import fitsio
from astropy.io import fits
import numpy as np
from scipy.interpolate import interp1d
import healpy
from pkg_resources import resource_filename

from picca.delta_extraction.astronomical_objects.forest import Forest
from picca.delta_extraction.astronomical_objects.pk1d_forest import Pk1dForest
from picca.delta_extraction.errors import ExpectedFluxError
from picca.delta_extraction.expected_flux import ExpectedFlux

accepted_options = ["input directory", "iter out prefix",
                    "num processors",
                    "var lss binning"]

defaults = {
    "iter out prefix": "delta_attributes",
}


class TrueContinuum(ExpectedFlux):
    """Class to the expected flux calculated used the true continuum unabsorbed contiuum
    for mocks. It uses var_lss pre-computed from mocks and the mean flux estimated from 

    Methods
    -------
    extract_deltas (from ExpectedFlux)
    __init__
    _initialize_arrays_lin
    _initialize_arrays_log
    _parse_config
    compute_delta_stack
    compute_expected_flux
    read_true_continuum
    read_var_lss
    populate_los_ids

    Attributes
    ----------
    los_ids: dict (from ExpectedFlux)
    A dictionary to store the mean expected flux fraction, the weights, and
    the inverse variance for each line of sight. Keys are the identifier for the
    line of sight and values are dictionaries with the keys "mean expected flux",
    and "weights" pointing to the respective arrays. If the given Forests are
    also Pk1dForests, then the key "ivar" must be available. Arrays have the same
    size as the flux array for the corresponding line of sight forest instance.


    get_stack_delta: scipy.interpolate.interp1d or None
    Interpolation function to compute the mean delta (from stacking all lines of
    sight). None for no info.

    get_var_lss: scipy.interpolate.interp1d
    Interpolation function to compute mapping functions var_lss. See equation 4 of
    du Mas des Bourboux et al. 2020 for details. Data for interpolation is read from a file.

    input_directory: str
    Directory where true continum data is store

    iter_out_prefix: str
    Prefix of the iteration files. These file contain the statistical properties
    of deltas at a given iteration step. Intermediate files will add
    '_iteration{num}.fits.gz' to the prefix for intermediate steps and '.fits.gz'
    for the final results.

    num_processors: int or None
    Number of processors to be used to compute the mean continua. None for no
    specified number (subprocess will take its default value).
    """

    def __init__(self, config):
        """Initialize class instance.

        Arguments
        ---------
        config: configparser.SectionProxy
        Parsed options to initialize class

        Raise
        -----
        ExpectedFluxError if Forest.wave_solution is not 'lin' or 'log'
        """
        self.logger = logging.getLogger(__name__)
        super().__init__()

        # load variables from config
        self.input_directory = None
        self.iter_out_prefix = None
        self.num_iterations = None
        self.num_processors = None
        self._parse_config(config)

        # initialize stack variables
        self.get_stack_delta = None
        self.get_stack_delta_weights = None
        #initialize mean cont variables
        self.get_mean_cont = None
        self.get_mean_cont_weight = None

        # read large scale structure variance
        self.get_var_lss = None
        self.read_var_lss()


    def _parse_config(self, config):
        """Parse the configuration options

        Arguments
        ---------
        config: configparser.SectionProxy
        Parsed options to initialize class

        Raises
        ------
        ExpectedFluxError if iter out prefix is not valid
        """
        self.input_directory = config.get("input directory")
        if self.input_directory is None:
            raise ExpectedFluxError(
                "Missing argument 'input directory' required "
                "by TrueContinuum")

        self.iter_out_prefix = config.get("iter out prefix")
        if self.iter_out_prefix is None:
            raise ExpectedFluxError(
                "Missing argument 'iter out prefix' required "
                "by TrueContinuum")
        if "/" in self.iter_out_prefix:
            raise ExpectedFluxError(
                "Error constructing TrueContinuum. "
                "'iter out prefix' should not incude folders. "
                f"Found: {self.iter_out_prefix}")

        self.num_processors = config.getint("num processors")

        self.var_lss_binning = config.get("var lss binning", 'log')

    def compute_delta_stack(self, forests, stack_from_deltas=False):
        """Compute a stack of the delta field as a function of wavelength

        Arguments
        ---------
        forests: List of Forest
        A list of Forest from which to compute the deltas.

        stack_from_deltas: bool - default: False
        Flag to determine whether to stack from deltas or compute them

        Raise
        -----
        ExpectedFluxError if Forest.wave_solution is not 'lin' or 'log'
        """
        if Forest.wave_solution == "log":
            num_bins = int((Forest.log_lambda_max - Forest.log_lambda_min) /
                           Forest.delta_log_lambda) + 1
            stack_log_lambda = (Forest.log_lambda_min +
                                np.arange(num_bins) * Forest.delta_log_lambda)

        elif Forest.wave_solution == "lin":
            num_bins = int((Forest.lambda_max - Forest.lambda_min) /
                           Forest.delta_lambda) + 1
            stack_lambda = (Forest.lambda_min +
                            np.arange(num_bins) * Forest.delta_lambda)
        else:
            raise ExpectedFluxError("Forest.wave_solution must be either "
                                    "'log' or 'linear'")

        stack_delta = np.zeros(num_bins)
        stack_weight = np.zeros(num_bins)

        for forest in forests:
            if stack_from_deltas:
                expected_flux = self.los_ids.get(forest.los_id).get("mean expected flux")
                delta = forest.flux/expected_flux
                weights = self.los_ids.get(forest.los_id).get("weights")
            else:
                # ignore forest if continuum could not be computed
                if forest.continuum is None:
                    continue
                delta = forest.flux / forest.continuum
                if Forest.wave_solution == "log":
                    var_lss = self.get_var_lss(forest.log_lambda)
                elif Forest.wave_solution == "lin":
                    var_lss = self.get_var_lss(forest.lambda_)
                else:
                    raise ExpectedFluxError(
                        "Forest.wave_solution must be either "
                        "'log' or 'linear'")

                var_pipe = 1. / forest.ivar
                variance = var_pipe + var_lss*forest.continuum**2
                weights = 1./ variance

            if Forest.wave_solution == "log":
                bins = ((forest.log_lambda - Forest.log_lambda_min) /
                        Forest.delta_log_lambda + 0.5).astype(int)
            elif Forest.wave_solution == "lin":
                bins = ((forest.lambda_ - Forest.lambda_min) /
                        Forest.delta_lambda + 0.5).astype(int)
            else:
                raise ExpectedFluxError("Forest.wave_solution must be either "
                                        "'log' or 'linear'")

            rebin = np.bincount(bins, weights=delta*weights)
            stack_delta[:len(rebin)] += rebin/rebin.mean()
            rebin = np.bincount(bins, weights=weights)
            stack_weight[:len(rebin)] += rebin/rebin.mean()

        w = stack_weight > 0
        stack_delta[w] /= stack_weight[w]

        if Forest.wave_solution == "log":
            self.get_stack_delta = interp1d(stack_log_lambda[stack_weight > 0.],
                                            stack_delta[stack_weight > 0.],
                                            kind="nearest",
                                            fill_value="extrapolate")
            self.get_stack_delta_weights = interp1d(
                stack_log_lambda[stack_weight > 0.],
                stack_weight[stack_weight > 0.],
                kind="nearest",
                fill_value=0.0,
                bounds_error=False)
        elif Forest.wave_solution == "lin":
            self.get_stack_delta = interp1d(stack_lambda[stack_weight > 0.],
                                            stack_delta[stack_weight > 0.],
                                            kind="nearest",
                                            fill_value="extrapolate")
            self.get_stack_delta_weights = interp1d(
                stack_lambda[stack_weight > 0.],
                stack_weight[stack_weight > 0.],
                kind="nearest",
                fill_value=0.0,
                bounds_error=False)
        else:
            raise ExpectedFluxError("Forest.wave_solution must be either "
                                    "'log' or 'linear'")

    def compute_expected_flux(self, forests, out_dir):
        """

        Arguments
        ---------
        forests: List of Forest
        A list of Forest from which to compute the deltas.

        out_dir: str
        Directory where iteration statistics will be saved

        Raise
        -----
        ExpectedFluxError if Forest.wave_solution is not 'lin' or 'log'
        """
        num_bins = (int(
            (Forest.log_lambda_max_rest_frame -
             Forest.log_lambda_min_rest_frame) / Forest.delta_log_lambda) + 1)
        self.log_lambda_rest_frame = (
            Forest.log_lambda_min_rest_frame + (np.arange(num_bins) + 0.5) *
            (Forest.log_lambda_max_rest_frame -
             Forest.log_lambda_min_rest_frame) / num_bins)

        context = multiprocessing.get_context('fork')
        for iteration in range(1):
            pool = context.Pool(processes=self.num_processors)
            self.logger.progress(
                f"Reading continum with {self.num_processors} processors"
            )

            forests = pool.map(self.read_true_continuum, forests)
            pool.close()

        if Forest.wave_solution == "log":
            self.compute_mean_cont_log(forests)
        elif Forest.wave_solution == "lin":
            self.compute_mean_cont_lin(forests)
        else:
            raise ExpectedFluxError("Forest.wave_solution must be "
                                            "either 'log' or 'linear'")

        # now loop over forests to populate los_ids
        self.populate_los_ids(forests)
        # now compute the mean delta
        self.compute_delta_stack(forests,stack_from_deltas=True)
        # Save delta atributes
        self.save_delta_attributes(out_dir)

    def read_true_continuum(self, forest):
        """Read the forest continuum and insert it into

        Arguments
        ---------
        forest: Forest
        A forest instance where the continuum will be computed

        Return
        ------
        forest: Forest
        The modified forest instance

        Raise
        -----
        ExpectedFluxError if Forest.wave_solution is not 'lin' or 'log'
        """
        in_nside = 16
        healpix = healpy.ang2pix(in_nside, np.pi / 2 - forest.dec, forest.ra,
                                 nest=True)
        filename_truth = (
            f"{self.input_directory}/{healpix//100}/{healpix}/truth-{in_nside}-"
            f"{healpix}.fits")
        hdul = fits.open(filename_truth)
        wmin = hdul["TRUE_CONT"].header["WMIN"]
        wmax = hdul["TRUE_CONT"].header["WMAX"]
        dwave = hdul["TRUE_CONT"].header["DWAVE"]
        twave = np.arange(wmin, wmax + dwave, dwave)
        true_cont = hdul["TRUE_CONT"].data
        hdul.close()
        indx = np.where(true_cont["TARGETID"]==forest.targetid)
        true_continuum = interp1d(twave, true_cont["TRUE_CONT"][indx])

        if Forest.wave_solution == "log":
            forest.continuum = true_continuum(10**forest.log_lambda)[0]
            mean_optical_depth = np.ones(forest.log_lambda.size)
            tau, gamma, lambda_rest_frame = 0.0023, 3.64, 1215.67
            w = 10.**forest.log_lambda / (1. + forest.z) <= lambda_rest_frame
            z = 10.**forest.log_lambda / lambda_rest_frame - 1.

        elif Forest.wave_solution == "lin":
            forest.continuum = true_continuum(forest.lambda_)[0]
            mean_optical_depth = np.ones(forest.lambda_.size)
            tau, gamma, lambda_rest_frame = 0.0023, 3.64, 1215.67
            w = forest.lambda_ / (1. + forest.z) <= lambda_rest_frame
            z = forest.lambda_ / lambda_rest_frame - 1.
        else:
            raise ExpectedFluxError("Forest.wave_solution must be either 'log' "
                                    "or 'lin'")

        # multiply by mean flux model for now, it might be changed for
        # computing the mean flux or reading it from file

        mean_optical_depth[w] *= np.exp(-tau * (1. + z[w])**gamma)
        forest.continuum *= mean_optical_depth
        forest.hpcont = healpix
        return forest

    def read_var_lss(self):
        """Read the LSS delta variance from files (written by the raw analysis)

        """

        if self.var_lss_binning == 'log':
            filename = 'deltas_lya_stats_log_bins.fits.gz'
        elif self.var_lss_binning == 'lin_2.4':
            filename = 'deltas_lya_stats_lin_bins_2.4.fits.gz'
        elif self.var_lss_binning == 'lin_3.2':
            filename = 'deltas_lya_stats_lin_bins_3.2.fits.gz'
        else:
            raise ValueError('Unknown var_lss_binning {}.'.format(self.var_lss_binning))

        var_lss_file = resource_filename('picca', 'delta_extraction')
        var_lss_file += '/expected_fluxes/var_lss/' + filename

        hdul = fits.open(var_lss_file)
        log_lambda = hdul[1].data['LAMBDA']
        flux_variance = hdul[1].data['VAR']
        mean_flux = hdul[1].data['MEANFLUX']
        hdul.close()

        if 'lin' in self.var_lss_binning:
            log_lambda = np.log10(log_lambda)

        var_lss=flux_variance/mean_flux**2
        self.get_var_lss = interp1d(log_lambda,
                                    var_lss,
                                    fill_value='extrapolate',
                                    kind='nearest')

    def compute_mean_cont_lin(self, forests):
        """Compute the mean quasar continuum over the whole sample assuming a
        linear wavelength solution. Then updates the value of self.get_mean_cont
        to contain it

        Arguments
        ---------
        forests: List of Forest
        A list of Forest from which to compute the deltas.
        """
        num_bins = self.lambda_rest_frame.size
        mean_cont = np.zeros(num_bins)
        mean_cont_weight = np.zeros(num_bins)

        for forest in forests:
            if forest.bad_continuum_reason is not None:
                continue
            bins = (
                (forest.lambda_ /
                 (1 + forest.z) - Forest.lambda_min_rest_frame) /
                (Forest.lambda_max_rest_frame - Forest.lambda_min_rest_frame) *
                num_bins).astype(int)

            var_lss = self.get_var_lss(forest.lambda_)
            var_pipe = 1. / forest.ivar
            variance = var_pipe + var_lss*forest.continuum**2
            weights = 1 / var_pipe
            cont = np.bincount(bins,
                               weights=forest.continuum * weights)
            mean_cont[:len(cont)] += cont

            cont = np.bincount(bins, weights=weights)
            mean_cont_weight[:len(cont)] += cont

        w = mean_cont_weight > 0
        mean_cont[w] /= mean_cont_weight[w]
        mean_cont /= mean_cont.mean()
        lambda_cont = self.lambda_rest_frame[w]

        # the new mean continuum is multiplied by the previous one to recover
        # <F/spectrum_dependent_fitting_function>

        self.get_mean_cont = interp1d(lambda_cont,
                                      mean_cont,
                                      fill_value="extrapolate")
        self.get_mean_cont_weight = interp1d(lambda_cont,
                                             mean_cont_weight,
                                             fill_value=0.0,
                                             bounds_error=False)

    def compute_mean_cont_log(self, forests):
        """Compute the mean quasar continuum over the whole sample assuming a
        log-linear wavelength solution. Then updates the value of
        self.get_mean_cont to contain it

        Arguments
        ---------
        forests: List of Forest
        A list of Forest from which to compute the deltas.
        """
        num_bins = self.log_lambda_rest_frame.size
        mean_cont = np.zeros(num_bins)
        mean_cont_weight = np.zeros(num_bins)

        for forest in forests:
            if forest.bad_continuum_reason is not None:
                continue
            bins = ((forest.log_lambda - Forest.log_lambda_min_rest_frame -
                     np.log10(1 + forest.z)) /
                    (Forest.log_lambda_max_rest_frame -
                     Forest.log_lambda_min_rest_frame) * num_bins).astype(int)

            var_lss = self.get_var_lss(forest.log_lambda)
            var_pipe = 1. / forest.ivar
            variance = var_pipe + var_lss*forest.continuum**2
            weights = 1 / var_pipe
            cont = np.bincount(bins, weights= forest.continuum * weights)
            mean_cont[:len(cont)] += cont

            cont = np.bincount(bins, weights=weights)
            mean_cont_weight[:len(cont)] += cont

        w = mean_cont_weight > 0
        mean_cont[w] /= mean_cont_weight[w]
        mean_cont /= mean_cont.mean()
        log_lambda_cont = self.log_lambda_rest_frame[w]

        self.get_mean_cont = interp1d(log_lambda_cont,
                                      mean_cont,
                                      fill_value="extrapolate")
        self.get_mean_cont_weight = interp1d(log_lambda_cont,
                                             mean_cont_weight,
                                             fill_value=0.0,
                                             bounds_error=False)


    def populate_los_ids(self, forests):
        """Populate the dictionary los_ids with the mean expected flux, weights,
        and inverse variance arrays for each line-of-sight.

        Arguments
        ---------
        forests: List of Forest
        A list of Forest from which to compute the deltas.
        """

        for forest in forests:
            if forest.bad_continuum_reason is not None:
                self.logger.info(f"Rejected forest with los_id {forest.los_id} "
                                 f"due to {forest.bad_continuum_reason}")
                continue

            # get the variance functions and statistics
            if Forest.wave_solution == "log":
                var_lss = self.get_var_lss(forest.log_lambda)
            elif Forest.wave_solution == "lin":
                var_lss = self.get_var_lss(forest.lambda_)
            else:
                raise ExpectedFluxError("Forest.wave_solution must be either "
                                        "'log' or 'lin'")

            mean_expected_flux = forest.continuum
            var_pipe = 1. / forest.ivar
            variance = var_pipe + var_lss*mean_expected_flux**2
            weights = 1. / variance

            if isinstance(forest, Pk1dForest):
                ivar = forest.ivar / mean_expected_flux**2

                self.los_ids[forest.los_id] = {
                    "mean expected flux": mean_expected_flux,
                    "weights": weights,
                    "ivar": ivar,
                    "continuum": forest.continuum,
                }
            else:
                self.los_ids[forest.los_id] = {
                    "mean expected flux": mean_expected_flux,
                    "weights": weights,
                    "continuum": forest.continuum,
                }

    def save_delta_attributes(self,out_dir):
        """Save the statistical properties of deltas at a given iteration
        step

        Arguments
        ---------
        iteration: int
        Iteration number. -1 for final iteration

        out_dir: str
        Directory where data will be saved

        Raise
        -----
        ExpectedFluxError if Forest.wave_solution is not 'lin' or 'log'
        """
        iter_out_file = self.iter_out_prefix + ".fits.gz"

        with fitsio.FITS(out_dir + iter_out_file, 'rw',
                         clobber=True) as results:
            header = {}
            header["FITORDER"] = -1
            if Forest.wave_solution == "log":
                # TODO: update this once the TODO in compute continua is fixed
                num_bins = int((Forest.log_lambda_max - Forest.log_lambda_min) /
                               Forest.delta_log_lambda) + 1
                stack_log_lambda = (
                    Forest.log_lambda_min +
                    np.arange(num_bins) * Forest.delta_log_lambda)
                results.write([
                    stack_log_lambda,
                    self.get_stack_delta(stack_log_lambda),
                    self.get_stack_delta_weights(stack_log_lambda)
                ],
                              names=['loglam', 'stack', 'weight'],
                              header=header,
                              extname='STACK_DELTAS')


                results.write([
                    self.log_lambda_rest_frame,
                    self.get_mean_cont(self.log_lambda_rest_frame),
                    self.get_mean_cont_weight(self.log_lambda_rest_frame),
                ],
                              names=['loglam_rest', 'mean_cont', 'weight'],
                              extname='CONT')

            elif Forest.wave_solution == "lin":
                num_bins = int((Forest.lambda_max - Forest.lambda_min) /
                               Forest.delta_lambda) + 1
                stack_lambda = (Forest.lambda_min +
                                np.arange(num_bins) * Forest.delta_lambda)

                results.write([
                    stack_lambda,
                    self.get_stack_delta(stack_lambda),
                    self.get_stack_delta_weights(stack_lambda)
                ],
                              names=['lam', 'stack', 'weight'],
                              header=header,
                              extname='STACK')


                results.write([
                    self.lambda_rest_frame,
                    self.get_mean_cont(self.lambda_rest_frame),
                    self.get_mean_cont_weight(self.lambda_rest_frame),
                ],
                              names=['lam_rest', 'mean_cont', 'weight'],
                              extname='CONT')

            else:
                raise ExpectedFluxError("Forest.wave_solution must be either "
                                        "'log' or 'lin'")