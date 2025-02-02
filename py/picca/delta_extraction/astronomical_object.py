"""This module defines the class AstronomicalObjectError
from which all astronomical objects must inherit
"""
import healpy
import numpy as np

from picca.delta_extraction.errors import AstronomicalObjectError

class AstronomicalObject:
    """Base class from which all astronomical ojects must inherit.

    Methods
    -------
    __init__
    __gt__
    __eq__
    get_header

    Attributes
    ----------
    dec: float
    Declination (in rad)

    healpix: int
    Healpix number associated with (ra, dec)

    los_id: longint
    Line-of-sight id

    ra: float
    Right ascention (in rad)

    z: float
    Redshift
    """
    def __init__(self, **kwargs):
        """Initialize instance

        Arguments
        ---------
        **kwargs: dict
        Dictionary contiaing the information

        Raise
        -----
        AstronomicalObjectError if there are missing variables
        """
        self.dec = kwargs.get("dec")
        if self.dec is None:
            raise AstronomicalObjectError("Error constructing AstronomicalObject. "
                                          "Missing variable 'dec'")

        self.los_id = kwargs.get("los_id")
        if self.los_id is None:
            raise AstronomicalObjectError("Error constructing AstronomicalObject. "
                                          "Missing variable 'los_id'")

        self.ra = kwargs.get("ra")
        if self.ra is None:
            raise AstronomicalObjectError("Error constructing AstronomicalObject. "
                                          "Missing variable 'ra'")

        self.z = kwargs.get("z")
        if self.z is None:
            raise AstronomicalObjectError("Error constructing AstronomicalObject. "
                                          "Missing variable 'z'")

        self.healpix = healpy.ang2pix(16, np.pi / 2 - self.dec, self.ra)

    def __gt__(self, other):
        """Comparare two astronomical_objects

        Arguments
        ---------
        other: AstronomicalObject
        Comparison object

        Return
        ------
        True if this object is 'greater than' the other object
        """
        is_greater = False
        if self.healpix > other.healpix:
            is_greater = True
        elif self.healpix == other.healpix:
            if self.ra > other.ra:
                is_greater = True
            if self.ra == other.ra:
                if self.dec > other.dec:
                    is_greater = True
                if self.dec == other.dec:
                    if self.z > other.z:
                        is_greater = True
        return is_greater

    def __eq__(self, other):
        """Comparare two astronomical_objects

        Arguments
        ---------
        other: AstronomicalObject
        Comparison object

        Returns
        -------
        True if the objects are equal
        """
        return (self.healpix == other.healpix and self.ra == other.ra and
                self.dec == other.dec and self.z == other.z)

    def get_header(self):
        """Return line-of-sight data to be saved as a fits file header

        Return
        ------
        header : list of dict
        A list of dictionaries containing 'name', 'value' and 'comment' fields
        """
        header = [
            {
                'name': 'LOS_ID',
                'value': self.los_id,
                'comment': 'Picca line-of-sight id'
            },
            {
                'name': 'RA',
                'value': self.ra,
                'comment': 'Right Ascension [rad]'
            },
            {
                'name': 'DEC',
                'value': self.dec,
                'comment': 'Declination [rad]'
            },
            {
                'name': 'Z',
                'value': self.z,
                'comment': 'Redshift'
            },
        ]

        return header
