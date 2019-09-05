from __future__ import print_function

import scipy as sp
from scipy.fftpack import fft, fftfreq
import scipy.interpolate as spint

from picca import constants
from picca.utils import print



def split_forest(nb_part,dll,ll,de,diff,iv,first_pixel,reso_matrix=None,dll_reso=None):

    ll_limit=[ll[first_pixel]]
    nb_bin= (len(ll)-first_pixel)//nb_part
    lam_lya = constants.absorber_IGM["LYA"]

    m_z_arr = []
    ll_arr = []
    de_arr = []
    diff_arr = []
    iv_arr = []
    dll_reso_arr = []
    reso_matrix_arr = []

    ll_c = ll.copy()
    de_c = de.copy()
    diff_c = diff.copy()
    iv_c = iv.copy()
    if reso_matrix is not None:
        reso_matrix_c=reso_matrix.copy()
    for p in range(1,nb_part) :
        ll_limit.append(ll[nb_bin*p+first_pixel])

    ll_limit.append(ll[len(ll)-1]+0.1*dll)
    m_z_init = sp.mean([10**ll_c[-1],10**ll_c[0]])/lam_lya -1.0

    for p in range(nb_part) :

        selection = (ll_c>= ll_limit[p]) & (ll_c<ll_limit[p+1])

        ll_part = ll_c[selection]
        de_part = de_c[selection]
        diff_part = diff_c[selection]
        iv_part = iv_c[selection]
        if reso_matrix is not None:
            reso_matrix_part = reso_matrix_c[:, selection]

        m_z = sp.mean([10**ll_part[-1],10**ll_part[0]])/lam_lya -1.0

        m_z_arr.append(m_z)
        ll_arr.append(ll_part)
        de_arr.append(de_part)
        diff_arr.append(diff_part)
        iv_arr.append(iv_part)
        if dll_reso is not None:
            dll_reso_arr.append(dll_reso*(1+m_z_init)/(1+m_z))
        if reso_matrix is not None:
            reso_matrix_arr.append(reso_matrix_part)

    out=[m_z_arr, ll_arr, de_arr, diff_arr, iv_arr]
    if reso_matrix is not None:
        out.append(reso_matrix_arr)
    if dll_reso is not None:
        out.append(dll_reso_arr)

    return out

def rebin_diff_noise(dll,ll,diff):

    crebin = 3
    if (diff.size < crebin):
        print("Warning: diff.size too small for rebin")
        return diff
    dll2 = crebin*dll

    # rebin not mixing pixels separated by masks
    bin2 = sp.floor((ll-ll.min())/dll2+0.5).astype(int)

    # rebin regardless of intervening masks
    # nmax = diff.size//crebin
    # bin2 = sp.zeros(diff.size)
    # for n in range (1,nmax +1):
    #     bin2[n*crebin:] += sp.ones(diff.size-n*crebin)

    cdiff2 = sp.bincount(bin2.astype(int),weights=diff)
    civ2 = sp.bincount(bin2.astype(int))
    w = (civ2>0)
    if (len(civ2) == 0) :
        print( "Error: diff size = 0 ",diff)
    diff2 = cdiff2[w]/civ2[w]*sp.sqrt(civ2[w])
    diffout = sp.zeros(diff.size)
    nmax = len(diff)//len(diff2)
    for n in range (nmax+1) :
        lengthmax = min(len(diff),(n+1)*len(diff2))
        diffout[n*len(diff2):lengthmax] = diff2[:lengthmax-n*len(diff2)]
        sp.random.shuffle(diff2)

    return diffout


def fill_masked_pixels(dll,ll,delta,diff,iv,no_apply_filling,linear_binning=False):


    if no_apply_filling : return ll,delta,diff,iv,0


    ll_idx = ll.copy()
    if not linear_binning:
        ll_idx -= ll[0]
        ll_idx /= dll
        ll_idx += 0.5
    else:
        ll_idx = 10**ll_idx
        ll_idx -= 10**ll[0]
        ll_idx /= dll            #dll for linear binning needs to be a delta-lambda
        ll_idx += 0.5
    index =sp.array(ll_idx,dtype=int)
    index_all = range(index[-1]+1)
    index_ok = sp.in1d(index_all, index)

    delta_new = sp.zeros(len(index_all))
    delta_new[index_ok]=delta
    if not linear_binning:
        ll_new = sp.array(index_all,dtype=float)
        ll_new *= dll
        ll_new += ll[0]
    else:
        ll_new = sp.array(index_all,dtype=float)
        ll_new *= dll
        ll_new += 10 ** ll[0]
        ll_new = sp.log10(ll_new)

    diff_new = sp.zeros(len(index_all))
    diff_new[index_ok]=diff

    iv_new = sp.ones(len(index_all))
    iv_new *=0.0
    iv_new[index_ok]=iv

    nb_masked_pixel=len(index_all)-len(index)

    return ll_new,delta_new,diff_new,iv_new,nb_masked_pixel

def compute_Pk_raw(dll,delta,ll,linear_binning=False):   #MW: why does this function depend on ll at all? Every computation is based on dll only, so the explicit dependence on dll is not necessary

    #   Length in km/s
    if not linear_binning:
        length_lambda = dll*constants.speed_light/1000.*sp.log(10.)*len(delta)
    else:
        length_lambda = dll*len(delta)
    # make 1D FFT
    nb_pixels = len(delta)
    nb_bin_FFT = nb_pixels//2 + 1
    fft_a = fft(delta)

    # compute power spectrum
    fft_a = fft_a[:nb_bin_FFT]
    Pk = (fft_a.real ** 2 + fft_a.imag ** 2) * length_lambda / nb_pixels ** 2
    k = 2 * sp.pi * np.abs(fftfreq(nb_pixels, length_lambda / nb_pixels))
    #k=k[:nb_bin_FFT]
    k = sp.arange(nb_bin_FFT,dtype=float)*2*sp.pi/length_lambda

    return k,Pk


def compute_Pk_noise(dll,iv,diff,ll,run_noise,linear_binning=False):

    nb_pixels = len(iv)
    nb_bin_FFT = nb_pixels//2 + 1

    nb_noise_exp = 10
    Pk = sp.zeros(nb_bin_FFT)
    err = sp.zeros(nb_pixels)
    w = iv>0
    err[w] = 1.0/sp.sqrt(iv[w])

    if (run_noise) :
        for _ in range(nb_noise_exp): #iexp unused, but needed
            delta_exp= sp.zeros(nb_pixels)
            delta_exp[w] = sp.random.normal(0.,err[w])
            _,Pk_exp = compute_Pk_raw(dll,delta_exp,ll,linear_binning) #k_exp unused, but needed
            Pk += Pk_exp

        Pk /= float(nb_noise_exp)

    _,Pk_diff = compute_Pk_raw(dll,diff,ll,linear_binning) #k_diff unused, but needed

    return Pk,Pk_diff

def compute_cor_reso(delta_pixel, mean_reso, k, delta_pixel2=None, pixel_correction=None,infres=False):

    nb_bin_FFT = len(k)
    cor = sp.ones(nb_bin_FFT)

    if pixel_correction == 'default':  # default correction
        cor *= sp.sinc(k * delta_pixel / (2 * sp.pi))**2 #use numpy function directly
        #sp.ones(nb_bin_FFT)
        #sinc[k > 0.] = (sp.sin(k[k > 0.] * delta_pixel / 2.0) /
        #                (k[k > 0.] * delta_pixel / 2.0))**2
        #cor *= sinc

    if not infres:
        cor *= sp.exp(-(k*mean_reso)**2)
    return cor

def compute_cor_reso_matrix(dll_resmat, reso_matrix, k, delta_pixel, delta_pixel2, ll, pixel_correction=None,linear_binning=False):
    """
    Perform the resolution + pixelization correction assuming general resolution kernel
     as e.g. DESI resolution matrix
    """
    if len(reso_matrix.shape)==1:
        #assume you got a mean reso_matrix
        reso_matrix=reso_matrix[sp.newaxis,:]

    W2arr=[]
    for resmat in reso_matrix:
        r=sp.append(resmat, sp.zeros(ll.size-resmat.size))
        k_resmat,W2=compute_Pk_raw(dll_resmat, r, ll, linear_binning=linear_binning) #this assumes a pixel scale of 1 Angstrom inside the reso matrix
        W2/=W2[0]
        #this interpolates to the final k_binning if different
        W2int=spint.interp1d(k_resmat,W2,bounds_error=False)

        W2arr.append(W2int(k))
        #print ('k: {}'.format(' '.join(['{:.3f}'.format(ki) for ki in k]) ))

    Wres2=sp.mean(W2arr,axis=0)

    cor = sp.ones(len(k))
    cor *= Wres2
    nb_bin_FFT = len(k)
    if pixel_correction == 'default':  # default correction
        cor *= sp.sinc(k * delta_pixel / (2 * sp.pi))**2 #use numpy function directly
    elif pixel_correction == 'inverse':  #this is assuming that the reso_matrix is regridded by averaging square blocks of the initial matrix which would add a sinc**4 (i.e. the FFT of a triangular function [or rect convolved with rect])
        cor /= sp.sinc(k * delta_pixel / (2 * sp.pi))**2 #use numpy function directly
    return cor

class Pk1D :

    def __init__(self,ra,dec,zqso,mean_z,plate,mjd,fiberid,msnr,mreso,
                 k,Pk_raw,Pk_noise,cor_reso,Pk,nb_mp,Pk_diff=None):

        self.ra = ra
        self.dec = dec
        self.zqso = zqso
        self.mean_z = mean_z
        self.mean_snr = msnr
        self.mean_reso = mreso
        self.nb_mp = nb_mp

        self.plate = plate
        self.mjd = mjd
        self.fid = fiberid
        self.k = k
        self.Pk_raw = Pk_raw
        self.Pk_noise = Pk_noise
        self.cor_reso = cor_reso
        self.Pk = Pk
        self.Pk_diff = Pk_diff


    @classmethod
    def from_fitsio(cls,hdu):

        """
        read Pk1D from fits file
        """

        hdr = hdu.read_header()

        ra = hdr['RA']
        dec = hdr['DEC']
        zqso = hdr['Z']
        mean_z = hdr['MEANZ']
        mean_reso = hdr['MEANRESO']
        mean_SNR = hdr['MEANSNR']
        plate = hdr['PLATE']
        mjd = hdr['MJD']
        fid = hdr['FIBER']
        nb_mp = hdr['NBMASKPIX']

        data = hdu.read()
        k = data['k'][:]
        Pk = data['Pk'][:]
        Pk_raw = data['Pk_raw'][:]
        Pk_noise = data['Pk_noise'][:]
        cor_reso = data['cor_reso'][:]
        Pk_diff = data['Pk_diff'][:]

        return cls(ra,dec,zqso,mean_z,plate,mjd,fid, mean_SNR, mean_reso,k,Pk_raw,Pk_noise,cor_reso, Pk,nb_mp,Pk_diff)
