#!/usr/bin/env python

from __future__ import division, print_function

import argparse
import glob
import os
from array import array
from multiprocessing import Pool

import fitsio
import scipy as sp

from picca import constants
from picca.data import delta
from picca.pk1d import (compute_cor_reso, compute_Pk_noise, compute_Pk_raw,
                        fill_masked_pixels, rebin_diff_noise, split_forest,
                        compute_cor_reso_matrix)
from picca.utils import print


if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Compute the 1D power spectrum')

    parser.add_argument('--out-dir', type=str, default=None, required=True,
        help='Output directory')

    parser.add_argument('--out-format', type=str, default='fits', required=False,
        help='Output format:  fits (only one currently supported)')

    parser.add_argument('--in-dir', type=str, default=None, required=True,
        help='Directory to delta files')

    parser.add_argument('--in-format', type=str, default='fits', required=False,
        help=' Input format used for input files: ascii or fits')

    parser.add_argument('--SNR-min',type=float,default=2.,required=False,
        help='Minimal mean SNR per pixel ')

    parser.add_argument('--reso-max',type=float,default=85.,required=False,
        help='Maximal resolution in km/s ')

    parser.add_argument('--lambda-obs-min',type=float,default=3600.,required=False,
        help='Lower limit on observed wavelength [Angstrom]' )

    parser.add_argument('--nb-part',type=int,default=3,required=False,
        help='Number of parts in forest')

    parser.add_argument('--nb-pixel-min',type=int,default=75,required=False,
        help='Minimal number of pixels in a part of forest')

    parser.add_argument('--nb-pixel-masked-max',type=int,default=40,required=False,
        help='Maximal number of masked pixels in a part of forest')

    parser.add_argument('--no-apply-filling', action='store_true', default=False, required=False,
        help='Dont fill masked pixels')

    parser.add_argument('--noise-estimate', type=str, default='mean_diff', required=False,
        help='Estimate of Pk_noise pipeline/diff/mean_diff/rebin_diff/mean_rebin_diff')

    parser.add_argument('--forest-type', type=str, default='Lya', required=False,
        help='Forest used: Lya, SiIV, CIV')

    parser.add_argument('--debug', action='store_true', default=False, required=False,
        help='Fill root histograms for debugging')

    parser.add_argument('--pixel-correction', default='default', required=False,
        help='Which type of pixel correction to apply')

    parser.add_argument('--nproc', type=int, default=1, required=False,
        help='Number of processors')
    parser.add_argument('--res-estimate', default='Gaussian', required=False,
        help='Resolution correction estimated by: Gaussian, matrix (leave at default for perfect models)')
    parser.add_argument('--linear-binning', action='store_true', default = False,
            help='should the deltas be computed on linearly sampled wavelength bins')
    parser.add_argument('--output-in-angstrom', action='store_true', default = False,
            help='does not convert the power to velocity units when computed from linear binning')

    parser.add_argument('--use-desi-new-defaults', action='store_true', default = False,
        help='use changes put into picca to allow resolution treatment for the P1d more properly with DESI mocks (e.g. different sampling)')

    parser.add_argument('--nb-noise-exp', default = 100, type=int,
        help='number of pipeline noise realizations to generate per spectrum')

    parser.add_argument('--noise-overestim-factor', default = 1.0, type=float,
        help='factor by which the pipeline noise is too large')



    args = parser.parse_args()
    if args.use_desi_new_defaults:
        #add whatever we want the default for desi to be here
        args.linear_binning = True
        args.output_in_angstrom = True
        args.res_estimate = 'matrix'


    noiseless_fullres=False #stores if the read in spectra didn't contain ivar and reso info (if this is True everything will be treated as noiseless, infinite resolution)

    ###note that after some point all ll variables can be either in loglambda or lambda

    # Read deltas
    if (args.in_format=='fits') :
        fi = glob.glob(args.in_dir+"/*.fits.gz")
    elif (args.in_format=='ascii') :
        fi = glob.glob(args.in_dir+"/*.txt")

    data = {}
    ndata = 0
    print(args.in_dir)
    print(fi)
    # initialize randoms
    sp.random.seed(4)
    skipmsgprinted=False
    # define per file function
    # need to check if this is fine with the randoms
    def process_file(i):
        global ndata, data, noiseless_fullres,skipmsgprinted
        f=fi[i]
        if os.path.exists(args.out_dir + '/Pk1D-' + str(i) + '.fits.gz'):
            if not skipmsgprinted:
                print("skipped analysis for existing outputs")
                skipmsgprinted=True
            return 1#skip existing files

        if i%1==0:
            print("\rread {} of {} {}".format(i,len(fi),ndata),end="")

        # read fits or ascii file
        if (args.in_format=='fits') :
            hdus = fitsio.FITS(f)
            try:
                dels = [delta.from_fitsio(h,Pk1D_type=True) for h in hdus[1:]]
            except ValueError:
                print("\nPk1d_type=True didn't work on read in, maybe perfect model? Trying without!")
                dels = [delta.from_fitsio(h,Pk1D_type=False) for h in hdus[1:]]
                for d in dels:
                    d.iv=sp.ones(d.de.shape)*1e10
                    d.mean_SNR=1e5
                    d.mean_reso=1e-3
                    d.diff = sp.zeros(d.de.shape)
                    if args.linear_binning:
                        d.dlambda = sp.median(sp.diff(10**d.ll))  #(d.ll[-1]-d.ll[0])/(len(d.ll)-1) #both of those should give the same result, but the first is more explicite, second one should be faster, but this shouldn't be a dominant effect
                    else:
                        d.dll = sp.mean(sp.diff(d.ll))  #(d.ll[-1]-d.ll[0])/(len(d.ll)-1) #both of those should give the same result, but the first is more explicite, second one should be faster, but this shouldn't be a dominant effect
                    d.dll_resmat=d.dll
                noiseless_fullres=True
        elif (args.in_format=='ascii') :
            ascii_file = open(f,'r')
            dels = [delta.from_ascii(line) for line in ascii_file]
        if args.linear_binning != (dels[0].linear_binning==True):   #a bit hacky, but that way both None and False (encoded into fits as F or '        ') are caught...
            raise Exception("inconsistent settings for linear wavelength binning and delta files")

        ndata+=len(dels)
        print ("\n ndata =  ",ndata)
        out = None

        # loop over deltas
        for d in dels:

            # Selection over the SNR and the resolution
            if (d.mean_SNR<=args.SNR_min or d.mean_reso>=args.reso_max) : continue

            # first pixel in forest
            for first_pixel,first_pixel_ll in enumerate(d.ll):
                if 10.**first_pixel_ll>args.lambda_obs_min : break

            # minimum number of pixel in forest
            nb_pixel_min = args.nb_pixel_min
            if ((len(d.ll)-first_pixel)<nb_pixel_min) : continue

            # Split in n parts the forest
            nb_part_max = (len(d.ll)-first_pixel)//nb_pixel_min
            nb_part = min(args.nb_part,nb_part_max)
            if d.dll_resmat is None:
                if args.linear_binning:
                    d.ll=10**d.ll
                    d.dll_resmat = sp.median(sp.diff(d.ll))
                    d.dll = d.dll_resmat  #overwrite the d.dll entries whatever they are with the true pixelization
                else:
                    d.dll_resmat = 1 * sp.median(10 ** -d.ll) / sp.log(10.)  #converts 1 angstrom to whatever the relevant log lambda is at current lambda
            else:
                if args.linear_binning:
                    d.dll = d.dlambda
                    d.ll=10**d.ll

            ###note that beginning here, all ll arrays will be either lambda or log lambda binned depending on input and dll will be the corresponding pixel size

            if args.res_estimate == 'Gaussian':
                m_z_arr,ll_arr,de_arr,diff_arr,iv_arr, dll_res_arr = split_forest(nb_part,d.dll,d.ll,d.de,d.diff,d.iv,first_pixel,dll_reso=d.dll_resmat,linear_binning=args.linear_binning)
            elif args.res_estimate == 'matrix':
                m_z_arr, ll_arr, de_arr, diff_arr, iv_arr, reso_mat_arr, dll_res_arr = split_forest(nb_part, d.dll, d.ll, d.de, d.diff, d.iv, first_pixel, reso_matrix=d.reso_matrix, dll_reso=d.dll_resmat, linear_binning=args.linear_binning)
            for f in range(nb_part):
                # rebin diff spectrum (note that this has not been adapted to linear binning yet and might need changes)
                if (args.noise_estimate=='rebin_diff' or args.noise_estimate=='mean_rebin_diff'):
                    diff_arr[f]=rebin_diff_noise(d.dll,ll_arr[f],diff_arr[f])

                # Fill masked pixels with 0.
                if args.res_estimate == 'Gaussian':
                    ll_new,delta_new,diff_new,iv_new,nb_masked_pixel = fill_masked_pixels(d.dll,ll_arr[f],de_arr[f],diff_arr[f],iv_arr[f],args.no_apply_filling)
                elif args.res_estimate == 'matrix':
                    #for resolution matrix the filling is not yet fully implemented, so far just the mean is taken here
                    ll_new,delta_new,diff_new,iv_new,nb_masked_pixel = fill_masked_pixels(d.dll,ll_arr[f],de_arr[f],diff_arr[f],iv_arr[f],args.no_apply_filling)
                    reso_mat_new=sp.mean(reso_mat_arr[f],axis=1)
                dll_reso=dll_res_arr[f]


                if (nb_masked_pixel> args.nb_pixel_masked_max) : continue
                # if (args.out_format=='root' and  args.debug): compute_mean_delta(ll_new,delta_new,iv_new,d.zqso)

                lam_lya = constants.absorber_IGM["LYA"]
                if args.linear_binning:
                    z_abs = ll_new/lam_lya -1.0
                else:
                    z_abs = 10.**ll_new/lam_lya - 1.0

                mean_z_new = sp.mean(z_abs)

                # Compute Pk_raw
                k,Pk_raw = compute_Pk_raw(d.dll,delta_new,linear_binning=args.linear_binning)

                # Compute Pk_noise
                run_noise = True
                Pk_noise,Pk_diff = compute_Pk_noise(d.dll,iv_new*args.noise_overestim_factor**2,diff_new,run_noise,linear_binning=args.linear_binning,nb_noise_exp=args.nb_noise_exp)

                # Compute resolution correction

                if args.linear_binning:  #it's weird to compute this here manually, could be cleaned up
                    delta_pixel = d.dlambda
                else:
                    delta_pixel = d.dll*sp.log(10.)*constants.speed_light/1000.

                if args.res_estimate == 'Gaussian' and not noiseless_fullres:
                    cor_reso = compute_cor_reso(delta_pixel, d.mean_reso,k, pixel_correction=args.pixel_correction)
                elif  args.res_estimate == 'matrix' and not noiseless_fullres:
                    #this assumes pixelization of resolution matrix and spectrum to be the same (which it is for real data and the new linearly gridded mocks)
                    cor_reso = compute_cor_reso_matrix(reso_mat_new, k, len(ll_new), delta_pixel, pixel_correction=args.pixel_correction,linear_binning=args.linear_binning)
                else:
                    #this is for computing a pixelization correction only
                    cor_reso = compute_cor_reso(delta_pixel, d.mean_reso,k, pixel_correction=args.pixel_correction,infres=True)


                # Compute 1D Pk
                if (args.noise_estimate=='pipeline'):
                    Pk = (Pk_raw - Pk_noise)/cor_reso
                elif (args.noise_estimate=='diff' or args.noise_estimate=='rebin_diff'):
                    Pk = (Pk_raw - Pk_diff)/cor_reso
                elif (args.noise_estimate=='mean_diff' or args.noise_estimate=='mean_rebin_diff'):
                    selection = (k>0) & (k<0.02)
                    if (args.noise_estimate=='mean_rebin_diff'):
                        selection = (k>0.003) & (k<0.02)
                    Pk_mean_diff = sp.mean(Pk_diff[selection])
                    Pk = (Pk_raw - Pk_mean_diff)/cor_reso
                elif noiseless_fullres:
                    Pk = Pk_raw / cor_reso

                #to convert linearly binned data back to velocity space
                if args.linear_binning and not args.output_in_angstrom:
                     Pk*=constants.speed_light/1000/sp.mean(ll_new)  #note again that ll_new is actually lambda here
                     k/=constants.speed_light/1000/sp.mean(ll_new)

                # save in fits format

                if (args.out_format=='fits'):
                    hd = [ {'name':'RA','value':d.ra,'comment':"QSO's Right Ascension [degrees]"},
                        {'name':'DEC','value':d.dec,'comment':"QSO's Declination [degrees]"},
                        {'name':'Z','value':d.zqso,'comment':"QSO's redshift"},
                        {'name':'MEANZ','value':m_z_arr[f],'comment':"Absorbers mean redshift"},
                        {'name':'MEANRESO','value':d.mean_reso,'comment':'Mean resolution [km/s]'},
                        {'name':'MEANSNR','value':d.mean_SNR,'comment':'Mean signal to noise ratio'},
                        {'name':'NBMASKPIX','value':nb_masked_pixel,'comment':'Number of masked pixels in the section'},
                        {'name':'PLATE','value':d.plate,'comment':"Spectrum's plate id"},
                        {'name':'MJD','value':d.mjd,'comment':'Modified Julian Date,date the spectrum was taken'},
                        {'name': 'FIBER', 'value': d.fid, 'comment': "Spectrum's fiber number"},
                        {'name': 'LIN_BIN', 'value': args.linear_binning, 'comment': "analysis was performed on delta with linear binned lambda"},
                        {'name': 'THING_ID', 'value': d.thid, 'comment': "thingid"}
                    ]

                    cols=[k,Pk_raw,Pk_noise,Pk_diff,cor_reso,Pk]
                    names=['k','Pk_raw','Pk_noise','Pk_diff','cor_reso','Pk']
                    comments=['Wavenumber', 'Raw power spectrum', "Noise's power spectrum", 'Noise coadd difference power spectrum',\
                              'Correction resolution function', 'Corrected power spectrum (resolution and noise)']
                    baseunit='AA' if (args.linear_binning and args.output_in_angstrom) else 'km/s'
                    units = ['({})^-1'.format(baseunit)]
                    units.extend([baseunit]*3+['']+[baseunit])
                    try:
                        out.write(cols,names=names,header=hd,comments=comments,units=units)
                    except AttributeError:
                        out = fitsio.FITS(args.out_dir+f'/Pk1D-{i}.fits.gz','rw',clobber=True)    #note that the former naming convention only had a number instead of plate here
                        out.write(cols,names=names,header=hd,comment=comments,units=units)
        if (args.out_format=='fits' and out is not None):
            out.close()
        return 0
    if args.nproc>1:
        pool=Pool(args.nproc)
        pool.map(process_file,range(len(fi)))
    else:
        for i,f in enumerate(fi):
            process_file(i)
    # define per file functionloop over input files


# Store root file results
    # if (args.out_format=='root'):
    #      storeFile.Write()


    print ("all done ")
