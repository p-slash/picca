from __future__ import print_function
import numpy as np
import scipy as sp

from picca import constants
from picca.utils import print


def exp_diff(file,ll) :

    nexp_per_col = file[0].read_header()['NEXP']//2
    fltotodd  = np.zeros(ll.size)
    ivtotodd  = np.zeros(ll.size)
    fltoteven = np.zeros(ll.size)
    ivtoteven = np.zeros(ll.size)

    if (nexp_per_col)<2 :
        print("DBG : not enough exposures for diff")

    for iexp in range (nexp_per_col) :
        for icol in range (2):
            llexp = file[4+iexp+icol*nexp_per_col]["loglam"][:]
            flexp = file[4+iexp+icol*nexp_per_col]["flux"][:]
            ivexp = file[4+iexp+icol*nexp_per_col]["ivar"][:]
            mask  = file[4+iexp+icol*nexp_per_col]["mask"][:]
            bins = sp.searchsorted(ll,llexp)

            # exclude masks 25 (COMBINEREJ), 23 (BRIGHTSKY)?
            if iexp%2 == 1 :
                civodd=np.bincount(bins,weights=ivexp*(mask&2**25==0))
                cflodd=np.bincount(bins,weights=ivexp*flexp*(mask&2**25==0))
                fltotodd[:civodd.size-1] += cflodd[:-1]
                ivtotodd[:civodd.size-1] += civodd[:-1]
            else :
                civeven=np.bincount(bins,weights=ivexp*(mask&2**25==0))
                cfleven=np.bincount(bins,weights=ivexp*flexp*(mask&2**25==0))
                fltoteven[:civeven.size-1] += cfleven[:-1]
                ivtoteven[:civeven.size-1] += civeven[:-1]

    w=ivtotodd>0
    fltotodd[w]/=ivtotodd[w]
    w=ivtoteven>0
    fltoteven[w]/=ivtoteven[w]

    alpha = 1
    if (nexp_per_col%2 == 1) :
        n_even = (nexp_per_col-1)//2
        alpha = np.sqrt(4.*n_even*(n_even+1))/nexp_per_col
    diff = 0.5 * (fltoteven-fltotodd) * alpha ### CHECK THE * alpha (Nathalie)

    return diff


def exp_diff_desi(file,mask_targetid,method):
    argsort = np.flip(np.argsort(np.mean(file["IV"][mask_targetid],axis=1)))

    ivar_mean = np.mean(file["IV"][mask_targetid][:,:],axis=1)
    argmin_ivar = np.argmin(ivar_mean)
    argsort = np.arange(ivar_mean.size)
    argsort[-1],argsort[argmin_ivar] = argsort[argmin_ivar],argsort[-1]

    teff_lya = file["TEFF_LYA"][mask_targetid][argsort]
    flux = file["FL"][mask_targetid][argsort,:]
    ivar = file["IV"][mask_targetid][argsort,:]

    if(method == "diff_eboss"):
        return(exp_diff_desi_eboss(flux,ivar,teff_lya,False))
    if(method == "diff_eboss_even"):
        return(exp_diff_desi_eboss(flux,ivar,teff_lya,True))
    if(method == "diff_eboss_corr"):
        return(exp_diff_desi_eboss_corr(flux,ivar,teff_lya,False))
    if(method == "diff_eboss_corr_even"):
        return(exp_diff_desi_eboss_corr(flux,ivar,teff_lya,True))
    if(method == "diff_desi_array"):
        return(exp_diff_desi_array(flux,ivar,teff_lya,False))
    if(method == "diff_desi_array_even"):
        return(exp_diff_desi_array(flux,ivar,teff_lya,True))
    if(method == "diff_desi_mean_array"):
        return(exp_diff_desi_mean_array(flux,ivar,teff_lya,False))
    if(method == "diff_desi_mean_array_even"):
        return(exp_diff_desi_mean_array(flux,ivar,teff_lya,True))
    if(method == "diff_desi_time"):
        return(exp_diff_desi_time(flux,ivar,teff_lya,False))
    if(method == "diff_desi_time_even"):
        return(exp_diff_desi_time(flux,ivar,teff_lya,True))


def exp_diff_desi_eboss(flux,ivar,teff_lya,use_only_even):
    n_exp = len(flux)
    if (n_exp < 2):
        print("Not enough exposures for diff, spectra rejected")
        return None
    if(use_only_even):
        if (n_exp%2 == 1):
            print("Odd number of exposures discarded")
            return None
    fltotodd  = np.zeros(flux.shape[1])
    ivtotodd  = np.zeros(flux.shape[1])
    fltoteven = np.zeros(flux.shape[1])
    ivtoteven = np.zeros(flux.shape[1])
    for iexp in range (2* (n_exp//2)) :
        flexp = flux[iexp]
        ivexp = ivar[iexp]
        if iexp%2 == 1 :
            fltotodd += flexp * ivexp
            ivtotodd += ivexp
        else :
            fltoteven += flexp * ivexp
            ivtoteven += ivexp

    w_odd=ivtotodd>0
    fltotodd[w_odd]/=ivtotodd[w_odd]
    w_even=ivtoteven>0
    fltoteven[w_even]/=ivtoteven[w_even]

    alpha = 1
    if (n_exp%2 == 1) :
        n_even = n_exp//2
        alpha = np.sqrt(4.*n_even*(n_even+1))/n_exp
    diff = 0.5 * (fltoteven-fltotodd) * alpha

    return diff

def exp_diff_desi_eboss_corr(flux,ivar,teff_lya,use_only_even):
    n_exp = len(flux)
    if (n_exp < 2):
        print("Not enough exposures for diff, spectra rejected")
        return None
    if(use_only_even):
        if (n_exp%2 == 1):
            print("Odd number of exposures discarded")
            return None
    fltotodd  = np.zeros(flux.shape[1])
    ivtotodd  = np.zeros(flux.shape[1])
    fltoteven = np.zeros(flux.shape[1])
    ivtoteven = np.zeros(flux.shape[1])
    for iexp in range (2* (n_exp//2)) :
        flexp = flux[iexp]
        ivexp = ivar[iexp]
        if iexp%2 == 1 :
            fltotodd += flexp * ivexp
            ivtotodd += ivexp
        else :
            fltoteven += flexp * ivexp
            ivtoteven += ivexp

    w_odd=ivtotodd>0
    fltotodd[w_odd]/=ivtotodd[w_odd]
    w_even=ivtoteven>0
    fltoteven[w_even]/=ivtoteven[w_even]

    alpha = 1
    if (n_exp%2 == 1) :
        n_even = n_exp//2
        alpha = np.sqrt((2*n_even)/(n_exp))
    diff = 0.5 * (fltoteven-fltotodd) * alpha

    return diff


def exp_diff_desi_array(flux,ivar,teff_lya,use_only_even):
    n_exp = len(flux)
    if (n_exp < 2):
        print("Not enough exposures for diff, spectra rejected")
        return None
    if(use_only_even):
        if (n_exp%2 == 1):
            print("Odd number of exposures discarded")
            return None
    ivtot  = np.zeros(flux.shape[1])
    fltotodd  = np.zeros(flux.shape[1])
    ivtotodd  = np.zeros(flux.shape[1])
    fltoteven = np.zeros(flux.shape[1])
    ivtoteven = np.zeros(flux.shape[1])
    for iexp in range (2* (n_exp//2)) :
        flexp = flux[iexp]
        ivexp = ivar[iexp]
        if iexp%2 == 1 :
            fltotodd += flexp * ivexp
            ivtotodd += ivexp
        else :
            fltoteven += flexp * ivexp
            ivtoteven += ivexp
    for iexp in range(n_exp):
        ivtot += ivar[iexp]
    w_odd=ivtotodd>0
    fltotodd[w_odd]/=ivtotodd[w_odd]
    w_even=ivtoteven>0
    fltoteven[w_even]/=ivtoteven[w_even]
    w=w_odd&w_even&(ivtot>0)
    alpha_array  = np.ones(flux.shape[1])
    alpha_array[w] = (1/np.sqrt(ivtot[w]))/(0.5 * np.sqrt((1/ivtoteven[w]) + (1/ivtotodd[w])))
    diff = 0.5 * (fltoteven-fltotodd) * alpha_array
    return diff

def exp_diff_desi_mean_array(flux,ivar,teff_lya,use_only_even):
    n_exp = len(flux)
    if (n_exp < 2):
        print("Not enough exposures for diff, spectra rejected")
        return None
    if(use_only_even):
        if (n_exp%2 == 1):
            print("Odd number of exposures discarded")
            return None
    ivtot  = np.zeros(flux.shape[1])
    fltotodd  = np.zeros(flux.shape[1])
    ivtotodd  = np.zeros(flux.shape[1])
    fltoteven = np.zeros(flux.shape[1])
    ivtoteven = np.zeros(flux.shape[1])
    for iexp in range (2* (n_exp//2)) :
        flexp = flux[iexp]
        ivexp = ivar[iexp]
        if iexp%2 == 1 :
            fltotodd += flexp * ivexp
            ivtotodd += ivexp
        else :
            fltoteven += flexp * ivexp
            ivtoteven += ivexp
    for iexp in range(n_exp):
        ivtot += ivar[iexp]
    w_odd=ivtotodd>0
    fltotodd[w_odd]/=ivtotodd[w_odd]
    w_even=ivtoteven>0
    fltoteven[w_even]/=ivtoteven[w_even]
    w=w_odd&w_even&(ivtot>0)
    alpha_array  = np.ones(flux.shape[1])
    alpha_array[w] = (1/np.sqrt(ivtot[w]))/(0.5 * np.sqrt((1/ivtoteven[w]) + (1/ivtotodd[w])))
    alpha = np.nanmean((1/np.sqrt(ivtot[w]))) / np.nanmean((0.5 * np.sqrt((1/ivtoteven[w]) + (1/ivtotodd[w]))))
    diff = 0.5 * (fltoteven-fltotodd) * alpha
    return diff


def exp_diff_desi_time(flux,ivar,teff_lya,use_only_even):
    n_exp = len(flux)
    if (n_exp < 2):
        print("Not enough exposures for diff, spectra rejected")
        return None
    if(use_only_even):
        if (n_exp%2 == 1):
            print("Odd number of exposures discarded")
            return None
    fltotodd  = np.zeros(flux.shape[1])
    ivtotodd  = np.zeros(flux.shape[1])
    fltoteven = np.zeros(flux.shape[1])
    ivtoteven = np.zeros(flux.shape[1])
    t_even = 0
    t_odd = 0
    t_exp = np.sum(teff_lya)
    for iexp in range (2* (n_exp//2)) :
        flexp = flux[iexp]
        ivexp = ivar[iexp]
        teff_lya_exp = teff_lya[iexp]
        if iexp%2 == 1 :
            fltotodd += flexp * ivexp
            ivtotodd += ivexp
            t_odd += teff_lya_exp
        else :
            fltoteven += flexp * ivexp
            ivtoteven += ivexp
            t_even += teff_lya_exp

    w_odd=ivtotodd>0
    fltotodd[w_odd]/=ivtotodd[w_odd]
    w_even=ivtoteven>0
    fltoteven[w_even]/=ivtoteven[w_even]

    alpha = 2 * np.sqrt((t_odd*t_even)/(t_exp*(t_odd + t_even)))
    diff = 0.5 * (fltoteven-fltotodd) * alpha
    return diff


# def exp_diff_desi_test(file,mask_targetid) :
#     argsort = np.flip(np.argsort(file["TEFF_LYA"][mask_targetid][:]))
#     flux = file["FL"][mask_targetid][argsort,:]
#     ivar = file["IV"][mask_targetid][argsort,:]
#     teff_lya = file["TEFF_LYA"][mask_targetid][argsort]
#     n_exp = len(flux)
#     if (n_exp<2) :
#         print("DBG : not enough exposures for diff, spectra rejected")
#         return None
#     fltotodd  = np.zeros(flux.shape[1])
#     ivtotodd  = np.zeros(flux.shape[1])
#     fltoteven = np.zeros(flux.shape[1])
#     ivtoteven = np.zeros(flux.shape[1])
#     t_even = 0
#     t_odd = 0
#     t_exp = np.sum(teff_lya)
#     t_last = teff_lya[-1]
#     t_exp_2 = 0
#     ivtot  = np.zeros(flux.shape[1])
#     for i in range(n_exp):
#         t_exp_2 += teff_lya[i]
#     for iexp in range (2* (n_exp//2)) :
#         flexp = flux[iexp]
#         ivexp = ivar[iexp]
#         teff_lya_exp = teff_lya[iexp]
#         if iexp%2 == 1 :
#             fltotodd += flexp * ivexp
#             ivtotodd += ivexp
#             t_odd += teff_lya_exp
#         else :
#             fltoteven += flexp * ivexp
#             ivtoteven += ivexp
#             t_even += teff_lya_exp
#
#     w_odd=ivtotodd>0
#     fltotodd[w_odd]/=ivtotodd[w_odd]
#     w_even=ivtoteven>0
#     fltoteven[w_even]/=ivtoteven[w_even]
#
#     alpha = 1
#
#     for iexp in range(n_exp):
#         ivtot += ivar[iexp]
#     w=w_odd&w_even&(ivtot>0)
#
#     n_even = n_exp//2
#     # print("n_exp",n_exp)
#     # print("exp",t_exp)
#     # print("sum",t_odd + t_even + t_last)
#     # print("last",t_last)
#     # print("alphas")
#     # alpha_time = 2 * np.sqrt((t_odd*t_even)/(t_exp*(t_odd + t_even)))
#     # print("alpha_time",alpha_time)
#     # alpha_time_false = 2* np.sqrt((t_even *(t_exp-t_even))/t_exp**2)
#     # print("alpha_time_false",alpha_time_false)
#     # alpha_eboss = 2 * np.sqrt(n_even*(n_even+1)) / (2*n_even+1)
#     # print("alpha_eboss",alpha_eboss)
#     # alpha_eboss_corr = np.sqrt((2*n_even)/(2*n_even+1))
#     # print("alpha_eboss_corr",alpha_eboss_corr)
#     # alpha_array  = np.ones(flux.shape[1])
#     # alpha_array[w] = (1/np.sqrt(ivtot[w]))/(0.5 * np.sqrt((1/ivtoteven[w]) + (1/ivtotodd[w])))
#     # alpha_desi_mean = np.nanmean((1/np.sqrt(ivtot[w]))) / np.nanmean((0.5 * np.sqrt((1/ivtoteven[w]) + (1/ivtotodd[w]))))
#     # print("alpha_desi_mean",alpha_desi_mean)
#     # alpha_desi_mean = np.nanmean(alpha_array)
#     # print("alpha_desi_mean 2",alpha_desi_mean)
#
#     if (n_exp%2 == 1) :
#         alpha_time_oddeven = np.sqrt((t_exp - t_last)/t_exp)
#         print("alpha_time_oddeven",alpha_time_oddeven)
#         alpha = alpha_time_oddeven
#     diff = 0.5 * (fltoteven-fltotodd) * alpha
#     # import matplotlib.pyplot as plt
#     # if(n_exp == 5):
#     #     wave = 10 ** file["LL"]
#     #     plt.figure()
#     #     plt.plot(wave,alpha_array)
#     #     plt.plot(wave,np.full(alpha_array.shape,alpha_time))
#     #     plt.plot(wave,np.full(alpha_array.shape,alpha_desi_mean))
#     #     plt.legend(["Array","Time","mean ratio Array"])
#     #     plt.savefig(f"test_5_{np.random.randint(100)}")
#
#
#     return diff
#
# def exp_diff_desi_time(file,mask_targetid) :
#     argsort = np.flip(np.argsort(file["TEFF_LYA"][mask_targetid][:]))
#     flux = file["FL"][mask_targetid][argsort,:]
#     ivar = file["IV"][mask_targetid][argsort,:]
#     teff_lya = file["TEFF_LYA"][mask_targetid][argsort]
#
#     n_exp = len(flux)
#     if (n_exp<2) :
#         print("DBG : not enough exposures for diff, spectra rejected")
#         return None
#     fltotodd  = np.zeros(flux.shape[1])
#     ivtotodd  = np.zeros(flux.shape[1])
#     fltoteven = np.zeros(flux.shape[1])
#     ivtoteven = np.zeros(flux.shape[1])
#     t_even = 0
#     t_odd = 0
#     t_exp = np.sum(teff_lya)
#     t_last = teff_lya[-1]
#     for iexp in range (2* (n_exp//2)) :
#         flexp = flux[iexp]
#         ivexp = ivar[iexp]
#         teff_lya_exp = teff_lya[iexp]
#         if iexp%2 == 1 :
#             fltotodd += flexp * ivexp
#             ivtotodd += ivexp
#             t_odd += teff_lya_exp
#         else :
#             fltoteven += flexp * ivexp
#             ivtoteven += ivexp
#             t_even += teff_lya_exp
#
#     w=ivtotodd>0
#     fltotodd[w]/=ivtotodd[w]
#     w=ivtoteven>0
#     fltoteven[w]/=ivtoteven[w]
#
#     alpha = 1
#     if (n_exp%2 == 1) :
#         # n_even = (n_exp-1)//2
#         # alpha_N_old = np.sqrt(4.*n_even*(n_exp-n_even))/n_exp
#         # alpha_N = np.sqrt(4.*t_even*(t_exp-t_even))/t_exp
#         # alpha_C_new = np.sqrt((t_exp - t_last)/t_exp)
#         alpha_N_new = np.sqrt((t_exp - t_last)*(t_exp+t_last))/t_exp
#         alpha = alpha_N_new
#     diff = 0.5 * (fltoteven-fltotodd) * alpha
#
#     return diff
#
#
# def exp_diff_desi_noodd(file,mask_targetid) :
#     argsort = np.flip(np.argsort(np.mean(file["IV"][mask_targetid],axis=1)))
#     flux = file["FL"][mask_targetid][argsort,:]
#     ivar = file["IV"][mask_targetid][argsort,:]
#     n_exp = len(flux)
#     if (n_exp < 2):
#         print("Not enough exposures for diff, spectra rejected")
#         return None
#     if (n_exp%2 == 1):
#         print("Odd number of exposures discarded")
#         return None
#     ivtot  = np.zeros(flux.shape[1])
#     fltotodd  = np.zeros(flux.shape[1])
#     ivtotodd  = np.zeros(flux.shape[1])
#     fltoteven = np.zeros(flux.shape[1])
#     ivtoteven = np.zeros(flux.shape[1])
#     for iexp in range (2* (n_exp//2)) :
#         if iexp%2 == 1 :
#             fltotodd += flux[iexp] * ivar[iexp]
#             ivtotodd += ivar[iexp]
#         else :
#             fltoteven += flux[iexp] * ivar[iexp]
#             ivtoteven += ivar[iexp]
#     for iexp in range(n_exp):
#         ivtot += ivar[iexp]
#     w=ivtotodd>0
#     fltotodd[w]/=ivtotodd[w]
#     w=ivtoteven>0
#     fltoteven[w]/=ivtoteven[w]
#     alpha = 1.0
#     diff = 0.5 * (fltoteven-fltotodd) * alpha
#     return diff
#
#
# def exp_diff_desi_old(file,mask_targetid) :
#     argsort = np.flip(np.argsort(np.mean(file["IV"][mask_targetid],axis=1)))
#     flux = file["FL"][mask_targetid][argsort,:]
#     ivar = file["IV"][mask_targetid][argsort,:]
#
#     n_exp = len(flux)
#     if (n_exp<2) :
#         print("DBG : not enough exposures for diff, spectra rejected")
#         return None
#     fltotodd  = np.zeros(flux.shape[1])
#     ivtotodd  = np.zeros(flux.shape[1])
#     fltoteven = np.zeros(flux.shape[1])
#     ivtoteven = np.zeros(flux.shape[1])
#     for iexp in range (2* (n_exp//2)) :
#         flexp = flux[iexp]
#         ivexp = ivar[iexp]
#         if iexp%2 == 1 :
#             fltotodd += flexp * ivexp
#             ivtotodd += ivexp
#         else :
#             fltoteven += flexp * ivexp
#             ivtoteven += ivexp
#
#     w=ivtotodd>0
#     fltotodd[w]/=ivtotodd[w]
#     w=ivtoteven>0
#     fltoteven[w]/=ivtoteven[w]
#
#     alpha = 1
#     if (n_exp%2 == 1) :
#         n_even = (n_exp-1)//2
#         alpha = np.sqrt(4.*n_even*(n_even+1))/n_exp
#     diff = 0.5 * (fltoteven-fltotodd) * alpha
#
#     return diff
#
#
# def exp_diff_desi_new(file,mask_targetid) :
#     argsort = np.flip(np.argsort(np.mean(file["IV"][mask_targetid],axis=1)))
#     flux = file["FL"][mask_targetid][argsort,:]
#     ivar = file["IV"][mask_targetid][argsort,:]
#     n_exp = len(flux)
#     if (n_exp < 2):
#         print("Not enough exposures for diff, spectra rejected")
#         return None
#     ivtot  = np.zeros(flux.shape[1])
#     fltotodd  = np.zeros(flux.shape[1])
#     ivtotodd  = np.zeros(flux.shape[1])
#     fltoteven = np.zeros(flux.shape[1])
#     ivtoteven = np.zeros(flux.shape[1])
#     for iexp in range (2* (n_exp//2)) :
#         if iexp%2 == 1 :
#             fltotodd += flux[iexp] * ivar[iexp]
#             ivtotodd += ivar[iexp]
#         else :
#             fltoteven += flux[iexp] * ivar[iexp]
#             ivtoteven += ivar[iexp]
#     for iexp in range(n_exp):
#         ivtot += ivar[iexp]
#     w_odd=ivtotodd>0
#     fltotodd[w_odd]/=ivtotodd[w_odd]
#     w_even=ivtoteven>0
#     fltoteven[w_even]/=ivtoteven[w_even]
#     w=w_odd&w_even&(ivtot>0)
#     alpha  = np.ones(flux.shape[1])
#     alpha[w] = (1/np.sqrt(ivtot[w]))/(0.5 * np.sqrt((1/ivtoteven[w]) + (1/ivtotodd[w])))
#     diff = 0.5 * (fltoteven-fltotodd) * alpha
#     return diff
#
# def exp_diff_desi_new_even(file,mask_targetid) :
#     argsort = np.flip(np.argsort(np.mean(file["IV"][mask_targetid],axis=1)))
#     flux = file["FL"][mask_targetid][argsort,:]
#     ivar = file["IV"][mask_targetid][argsort,:]
#     n_exp = len(flux)
#     if (n_exp < 2):
#         print("Not enough exposures for diff, spectra rejected")
#         return None
#     fltotodd  = np.zeros(flux.shape[1])
#     ivtotodd  = np.zeros(flux.shape[1])
#     fltoteven = np.zeros(flux.shape[1])
#     ivtoteven = np.zeros(flux.shape[1])
#     for iexp in range (2* (n_exp//2)) :
#         if iexp%2 == 1 :
#             fltotodd += flux[iexp] * ivar[iexp]
#             ivtotodd += ivar[iexp]
#         else :
#             fltoteven += flux[iexp] * ivar[iexp]
#             ivtoteven += ivar[iexp]
#     w_odd=ivtotodd>0
#     fltotodd[w_odd]/=ivtotodd[w_odd]
#     w_even=ivtoteven>0
#     fltoteven[w_even]/=ivtoteven[w_even]
#     alpha = np.ones(flux.shape[1])
#     if (n_exp%2 == 1) :
#         ivtot  = np.zeros(flux.shape[1])
#         for iexp in range(n_exp):
#             ivtot += ivar[iexp]
#         w=w_odd&w_even&(ivtot>0)
#         alpha[w] = (1/np.sqrt(ivtot[w]))/(0.5 * np.sqrt((1/ivtoteven[w]) + (1/ivtotodd[w])))
#     diff = 0.5 * (fltoteven-fltotodd) * alpha
#     return diff
#


def spectral_resolution(wdisp,with_correction=None,fiber=None,ll=None) :

    reso = wdisp*constants.speed_light/1000.*1.0e-4*np.log(10.)

    if (with_correction):
        wave = np.power(10.,ll)
        corrPlateau = 1.267 - 0.000142716*wave + 1.9068e-08*wave*wave;
        corrPlateau[wave>6000.0] = 1.097

        fibnum = fiber%500
        if(fibnum<100):
            corr = 1. + (corrPlateau-1)*.25 + (corrPlateau-1)*.75*(fibnum)/100.
        elif (fibnum>400):
            corr = 1. + (corrPlateau-1)*.25 + (corrPlateau-1)*.75*(500-fibnum)/100.
        else:
            corr = corrPlateau
        reso *= corr
    return reso

def spectral_resolution_desi(reso_matrix, ll) :

    reso= np.clip(reso_matrix,1.0e-6,1.0e6)   #note that the following is not strictly speaking right, as the resolution matrix has been convolved with a rectangle along both rows and cols
    rms_in_pixel = (np.sqrt(1.0/2.0/sp.log(reso[len(reso)//2][:]/reso[len(reso)//2-1][:]))
                    + np.sqrt(4.0/2.0/sp.log(reso[len(reso)//2][:]/reso[len(reso)//2-2][:]))
                    + np.sqrt(1.0/2.0/sp.log(reso[len(reso)//2][:]/reso[len(reso)//2+1][:]))
                    + np.sqrt(4.0/2.0/sp.log(reso[len(reso)//2][:]/reso[len(reso)//2+2][:]))
                    )/4.0

    return rms_in_pixel#reso_in_km_per_s
