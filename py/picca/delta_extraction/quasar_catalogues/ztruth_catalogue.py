"""This module defines the class ZtruthCatalogue to read z_truth
files from DESI
"""
from astropy.table import Table
import numpy as np

from picca.delta_extraction.errors import QuasarCatalogueError
from picca.delta_extraction.quasar_catalogue import QuasarCatalogue
from picca.delta_extraction.userprint import userprint

class ZtruthCatalogue(QuasarCatalogue):
    """Reads the z_truth catalogue from DESI

    Methods
    -------
    __init__
    _parse_config


    Attributes
    ----------
    catalogue: astropy.table.Table (from QuasarCatalogue)
    The quasar catalogue

    max_num_spec: int or None (from QuasarCatalogue)
    Maximum number of spectra to read. None for no maximum

    z_min: float (from QuasarCatalogue)
    Minimum redshift. Quasars with redshifts lower than z_min will be
    discarded

    z_max: float (from QuasarCatalogue)
    Maximum redshift. Quasars with redshifts higher than or equal to
    z_max will be discarded

    filename: str
    Filename of the z_truth catalogue

    """
    def __init__(self, config):
        """Initialize class instance

        Arguments
        ---------
        config: configparser.SectionProxy
        Parsed options to initialize class
        """
        super().__init__(config)

        # load variables from config
        self.filename = None
        self._parse_config(config)

        # read DRQ Catalogue
        catalogue = self.read_catalogue()

        self.catalogue = catalogue

        super().trim_catalogue()

    def _parse_config(self, config):
        """Parse the configuration options

        Arguments
        ---------
        config: configparser.SectionProxy
        Parsed options to initialize class

        Raise
        -----
        DataError upon missing required variables
        """
        self.filename = config.get("catalogue")
        if self.filename is None:
            raise QuasarCatalogueError("Missing argument 'catalogue' required by ZtruthCatalogue")

    def read_catalogue(self):
        """Read the z_truth catalogue

        Returns
        -------
        catalogue: Astropy.table.Table
        Table with the catalogue
        """
        userprint('Reading catalogue from ', self.filename)
        catalogue = Table.read(self.filename, ext=1)

        keep_columns = ['RA', 'DEC', 'Z', 'TARGETID', 'FIBER', 'SPECTROGRAPH']

        ## Sanity checks
        userprint('')
        w = np.ones(len(catalogue), dtype=bool)
        userprint(f"start                 : nb object in cat = {np.sum(w)}")

        ## Redshift range
        w &= catalogue['Z'] >= self.z_min
        userprint(f"and z >= {self.z_min}        : nb object in cat = {np.sum(w)}")
        w &= catalogue['Z'] < self.z_max
        userprint(f"and z < {self.z_max}         : nb object in cat = {np.sum(w)}")

        catalogue.keep_columns(keep_columns)
        w = np.where(w)[0]
        catalogue = catalogue[w]

        return catalogue