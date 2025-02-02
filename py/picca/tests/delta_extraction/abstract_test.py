"""This file contains an abstract class to define functions common to all tests"""
import unittest
import re
import os
import numpy as np
import astropy.io.fits as fits

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

class AbstractTest(unittest.TestCase):
    """Abstract test class to define functions used in all tests

    Methods
    -------
    compare_ascii
    compare_fits
    setUp
    """
    def setUp(self):
        """ Check that the results folder exists and create it
            if it does not."""
        if not os.path.exists("{}/results/".format(THIS_DIR)):
            os.makedirs("{}/results/".format(THIS_DIR))
        if not os.path.exists("{}/results/Log/".format(THIS_DIR)):
            os.makedirs("{}/results/Log/".format(THIS_DIR))

    def compare_ascii(self, orig_file, new_file):
        """Compare two ascii files to check that they are equal

        Arguments
        ---------
        orig_file: str
        Control file

        new_file: str
        New file

        expand_dir: bool - Default: False
        If set to true, replace the instances of the string 'THIS_DIR' by
        its value
        """
        orig = open(orig_file, 'r')
        new = open(new_file, 'r')

        try:
            for orig_line, new_line in zip(orig.readlines(),
                                           new.readlines()):
                # this is necessary to remove the system dependent bits of
                # the paths
                if "py/picca/tests/delta_extraction" in orig_line:
                    orig_line = re.sub(r"\/[^ ]*\/py\/picca\/tests\/delta_extraction\/",
                                       "", orig_line)
                    new_line = re.sub(r"\/[^ ]*\/py\/picca\/tests\/delta_extraction\/",
                                      "", new_line)

                if not orig_line == new_line:
                    print("Lines not equal")
                    print("Original line " + orig_line)
                    print("New line " + new_line)
                    self.assertTrue(orig_line == new_line)
        finally:
            orig.close()
            new.close()

    def compare_fits(self, orig_file, new_file):
        """Compare two fits files to check that they are equal

        Arguments
        ---------
        orig_file: str
        Control file

        new_file: str
        New file
        """
        # open fits files
        orig_hdul = fits.open(orig_file)
        new_hdul = fits.open(new_file)

        try:
            # compare them
            self.assertTrue(len(orig_hdul), len(new_hdul))
            # loop over HDUs
            for hdu_index, _ in enumerate(orig_hdul):
                # check header
                orig_header = orig_hdul[hdu_index].header
                new_header = new_hdul[hdu_index].header
                for key in orig_header:
                    self.assertTrue(key in new_header)
                    if not key in ["CHECKSUM", "DATASUM"]:
                        if orig_header[key] != new_header[key]:
                            print(f"\nOriginal file: {orig_file}")
                            print(f"New file: {new_file}")
                            print(f"\n For header {orig_header['EXTNAME']}")
                            print(f"Different values found for key {key}: "
                                  f"orig: {orig_header[key]}, new: {new_header[key]}")
                        self.assertTrue((orig_header[key] == new_header[key]) or
                                        (np.isclose(orig_header[key], new_header[key])))
                for key in new_header:
                    if key not in orig_header:
                        print(f"\nOriginal file: {orig_file}")
                        print(f"New file: {new_file}")
                        print(f"key {key} missing in orig header")
                        if key in ["MEANSNR", "BLINDING"]:
                            continue
                    self.assertTrue(key in orig_header)
                # check data
                orig_data = orig_hdul[hdu_index].data
                new_data = new_hdul[hdu_index].data
                if orig_data is None:
                    self.assertTrue(new_data is None)
                else:
                    for col in orig_data.dtype.names:
                        if not col in new_data.dtype.names:
                            print(f"\nOriginal file: {orig_file}")
                            print(f"New file: {new_file}")
                            print(f"Column {col} in HDU {orig_header['EXTNAME']} "
                                  "missing in new file")
                        self.assertTrue(col in new_data.dtype.names)
                        if not np.allclose(orig_data[col],
                                     new_data[col],
                                     equal_nan=True):
                            print(f"\nOriginal file: {orig_file}")
                            print(f"New file: {new_file}")
                            print(f"Different values found for column {col} in"
                                  f"HDU {orig_header['EXTNAME']}")
                            print("original new isclose original-new\n")
                            for new, orig in zip(new_data[col], orig_data[col]):
                                print(f"{orig} {new} "
                                      f"{np.isclose(orig, new, equal_nan=True)} "
                                      f"{orig-new}")
                        self.assertTrue(np.allclose(orig_data[col],
                                                    new_data[col],
                                                    equal_nan=True))
                    for col in new_data.dtype.names:
                        if col not in orig_data.dtype.names:
                            print(f"Column {col} missing in orig header")
                            if col in ["num_pixels", "valid_fit"]:
                                continue
                        self.assertTrue(col in orig_data.dtype.names)
        finally:
            orig_hdul.close()
            new_hdul.close()

if __name__ == '__main__':
    pass
