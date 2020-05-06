"""This module defines a set of functions to manage reading of data.

This module provides a class (Metadata) and several functions:
    - read_dlas
    - read_absorbers
    - read_drq
    - read_dust_map
    - read_data
    - read_from_spec
    - read_from_mock_1D
    - read_from_pix
    - read_from_spcframe
    - read_from_spplate
    - read_from_desi
    - read_deltas
    - read_objects
See the respective documentation for details
"""
import fitsio
import numpy as np
import scipy as sp
import healpy
import glob
import sys
import time
import os.path
import copy

from picca.utils import userprint
from picca.data import Forest, Delta, QSO
from picca.prep_Pk1D import exp_diff, spectral_resolution
from picca.prep_Pk1D import spectral_resolution_desi

## use a metadata class to simplify things
class Metadata(object):
    """Class defined to organize the storage of metadata.

    Attributes:
        thingid: integer or None
            Thingid of the observation.
        ra: float or None
            Right-ascension of the quasar (in radians).
        dec: float or None
            Declination of the quasar (in radians).
        z_qso: float or None
            Redshift of the quasar.
        plate: integer or None
            Plate number of the observation.
        mjd: integer or None
            Modified Julian Date of the observation.
        fiberid: integer or None
            Fiberid of the observation.
        order: 0 or 1 or None
            Order of the log10(lambda) polynomial for the continuum fit

    Methods:
        __init__
    """
    def __init__(self):
        """Initialize instance."""
        self.thingid = None
        self.ra = None
        self.dec = None
        self.z_qso = None
        self.plate = None
        self.mjd = None
        self.fiberid = None
        self.order = None

def read_dlas(filename):
    """Read the DLA catalog from a fits file.

    ASCII or DESI files can be converted using:
        utils.eBOSS_convert_DLA()
        utils.desi_convert_DLA()

    Args:
        filename: str
            File containing the DLAs

    Returns:
        A dictionary with the DLA's information. Keys are the THING_ID
        associated with the DLA. Values is a tuple with its redshift and
        column density.
    """
    columns_list = ['THING_ID', 'Z', 'NHI']
    hdul = fitsio.FITS(filename)
    cat = {col: hdul['DLACAT'][col][:] for col in columns_list}
    hdul.close()

    # sort the items in the dictionary according to THING_ID and redshift
    w = np.argsort(cat['Z'])
    for key in cat.keys():
        cat[key] = cat[key][w]
    w = np.argsort(cat['THING_ID'])
    for key in cat.keys():
        cat[key] = cat[key][w]

    # group DLAs on the same line of sight together
    dlas = {}
    for thingid in np.unique(cat['THING_ID']):
        w = (thingid == cat['THING_ID'])
        dlas[thingid] = [(zabs, nhi)
                         for zabs, nhi in zip(cat['Z'][w], cat['NHI'][w])]
    number_dlas = np.sum([len(dla) for dla in dlas.values()])

    userprint('\n')
    userprint(' In catalog: {} DLAs'.format(number_dlas))
    userprint(' In catalog: {} forests have a DLA'.format(len(dlas)))
    userprint('\n')

    return dlas

def read_absorbers(filename):
    """Read the absorbers catalog from an ascii file.

    Args:
        filename: str
            File containing the absorbers

    Returns:
        A dictionary with the absorbers's information. Keys are the THING_ID
        associated with the DLA. Values is a tuple with its redshift and
        column density.
    """
    file = open(filename)
    absorbers = {}
    number_absorbers = 0
    col_names = None
    for line in file.readlines():
        cols = line.split()
        if len(cols) == 0:
            continue
        if cols[0][0] == "#":
            continue
        if cols[0] == "ThingID":
            col_names = cols
            continue
        if cols[0][0] == "-":
            continue
        thingid = int(cols[col_names.index("ThingID")])
        if thingid not in absorbers:
            absorbers[thingid] = []
        lambda_absorber = float(cols[col_names.index("lambda")])
        absorbers[thingid].append(lambda_absorber)
        number_absorbers += 1
    file.close()

    userprint("")
    userprint(" In catalog: {} absorbers".format(number_absorbers))
    userprint(" In catalog: {} forests have absorbers".format(len(absorbers)))
    userprint("")

    return absorbers

def read_drq(drq,zmin,zmax,keep_bal,bi_max=None):
    hdul = fitsio.FITS(drq)

    ## Redshift
    try:
        z_qso = hdul[1]['Z'][:]
    except:
        userprint("Z not found (new DRQ >= DRQ14 style), using Z_VI (DRQ <= DRQ12)")
        z_qso = hdul[1]['Z_VI'][:]

    ## Info of the primary observation
    thingid = hdul[1]['THING_ID'][:]
    ra = hdul[1]['RA'][:].astype('float64')
    dec = hdul[1]['DEC'][:].astype('float64')
    plate = hdul[1]['PLATE'][:]
    mjd = hdul[1]['MJD'][:]
    fiberid = hdul[1]['FIBERID'][:]

    ## Sanity
    userprint('')
    w = sp.ones(ra.size,dtype=bool)
    userprint(" start               : nb object in cat = {}".format(w.sum()) )
    w &= thingid>0
    userprint(" and thingid>0          : nb object in cat = {}".format(w.sum()) )
    w &= ra!=dec
    userprint(" and ra!=dec         : nb object in cat = {}".format(w.sum()) )
    w &= ra!=0.
    userprint(" and ra!=0.          : nb object in cat = {}".format(w.sum()) )
    w &= dec!=0.
    userprint(" and dec!=0.         : nb object in cat = {}".format(w.sum()) )
    w &= z_qso>0.
    userprint(" and z>0.            : nb object in cat = {}".format(w.sum()) )

    ## Redshift range
    if not zmin is None:
        w &= z_qso>=zmin
        userprint(" and z>=zmin         : nb object in cat = {}".format(w.sum()) )
    if not zmax is None:
        w &= z_qso<zmax
        userprint(" and z<zmax          : nb object in cat = {}".format(w.sum()) )

    ## BAL visual
    if not keep_bal and bi_max==None:
        try:
            bal_flag = hdul[1]['BAL_FLAG_VI'][:]
            w &= bal_flag==0
            userprint(" and BAL_FLAG_VI == 0  : nb object in cat = {}".format(ra[w].size) )
        except:
            userprint("BAL_FLAG_VI not found\n")
    ## BAL CIV
    if bi_max is not None:
        try:
            bi = hdul[1]['BI_CIV'][:]
            w &= bi<=bi_max
            userprint(" and BI_CIV<=bi_max  : nb object in cat = {}".format(ra[w].size) )
        except:
            userprint("--bi-max set but no BI_CIV field in HDU")
            sys.exit(1)
    userprint("")

    ra = ra[w]*sp.pi/180.
    dec = dec[w]*sp.pi/180.
    z_qso = z_qso[w]
    thingid = thingid[w]
    plate = plate[w]
    mjd = mjd[w]
    fiberid = fiberid[w]
    hdul.close()

    return ra,dec,z_qso,thingid,plate,mjd,fiberid

def read_dust_map(drq, Rv = 3.793):
    hdul = fitsio.FITS(drq)
    thingid = hdul[1]['THING_ID'][:]
    ext  = hdul[1]['EXTINCTION'][:][:,1]/Rv
    hdul.close()

    return dict(zip(thingid, ext))

target_mobj = 500
nside_min = 8

def read_data(in_dir,drq,mode,zmin = 2.1,zmax = 3.5,nspec=None,log=None,keep_bal=False,bi_max=None,order=1, best_obs=False, single_exp=False, pk1d=None):

    userprint("mode: "+mode)
    ra,dec,z_qso,thingid,plate,mjd,fiberid = read_drq(drq,zmin,zmax,keep_bal,bi_max=bi_max)

    if nspec != None:
        ## choose them in a small number of pixels
        pixs = healpy.ang2pix(16, sp.pi / 2 - dec, ra)
        s = sp.argsort(pixs)
        ra = ra[s][:nspec]
        dec = dec[s][:nspec]
        z_qso = z_qso[s][:nspec]
        thingid = thingid[s][:nspec]
        plate = plate[s][:nspec]
        mjd = mjd[s][:nspec]
        fiberid = fiberid[s][:nspec]

    if mode == "pix":
        try:
            fin = in_dir + "/master.fits.gz"
            hdul = fitsio.FITS(fin)
        except IOError:
            try:
                fin = in_dir + "/master.fits"
                hdul = fitsio.FITS(fin)
            except IOError:
                try:
                    fin = in_dir + "/../master.fits"
                    hdul = fitsio.FITS(fin)
                except:
                    userprint("error reading master")
                    sys.exit(1)
        nside = hdul[1].read_header()['NSIDE']
        hdul.close()
        pixs = healpy.ang2pix(nside, sp.pi / 2 - dec, ra)
    elif mode in ["spec","corrected-spec","spcframe","spplate","spec-mock-1D"]:
        nside = 256
        pixs = healpy.ang2pix(nside, sp.pi / 2 - dec, ra)
        mobj = sp.bincount(pixs).sum()/len(np.unique(pixs))

        ## determine nside such that there are 1000 objs per pixel on average
        userprint("determining nside")
        while mobj<target_mobj and nside >= nside_min:
            nside //= 2
            pixs = healpy.ang2pix(nside, sp.pi / 2 - dec, ra)
            mobj = sp.bincount(pixs).sum()/len(np.unique(pixs))
        userprint("nside = {} -- mean #obj per pixel = {}".format(nside,mobj))
        if log is not None:
            log.write("nside = {} -- mean #obj per pixel = {}\n".format(nside,mobj))

    elif mode=="desi":
        nside = 8
        userprint("Found {} qsos".format(len(z_qso)))
        data = read_from_desi(nside,in_dir,thingid,ra,dec,z_qso,plate,mjd,fiberid,order, pk1d=pk1d)
        return data,len(data),nside,"RING"

    else:
        userprint("I don't know mode: {}".format(mode))
        sys.exit(1)

    data ={}
    ndata = 0

    if mode=="spcframe":
        pix_data = read_from_spcframe(in_dir,thingid, ra, dec, z_qso, plate, mjd, fiberid, order, mode=mode, log=log, best_obs=best_obs, single_exp=single_exp)
        ra = [d.ra for d in pix_data]
        ra = sp.array(ra)
        dec = [d.dec for d in pix_data]
        dec = sp.array(dec)
        pixs = healpy.ang2pix(nside, sp.pi / 2 - dec, ra)
        for i, p in enumerate(pixs):
            if p not in data:
                data[p] = []
            data[p].append(pix_data[i])

        return data, len(pixs),nside,"RING"

    if mode in ["spplate","spec","corrected-spec"]:
        if mode == "spplate":
            pix_data = read_from_spplate(in_dir,thingid, ra, dec, z_qso, plate, mjd, fiberid, order, log=log, best_obs=best_obs)
        else:
            pix_data = read_from_spec(in_dir,thingid, ra, dec, z_qso, plate, mjd, fiberid, order, mode=mode,log=log, pk1d=pk1d, best_obs=best_obs)
        ra = [d.ra for d in pix_data]
        ra = sp.array(ra)
        dec = [d.dec for d in pix_data]
        dec = sp.array(dec)
        pixs = healpy.ang2pix(nside, sp.pi / 2 - dec, ra)
        for i, p in enumerate(pixs):
            if p not in data:
                data[p] = []
            data[p].append(pix_data[i])

        return data, len(pixs), nside, "RING"

    upix = np.unique(pixs)

    for i, pix in enumerate(upix):
        w = pixs == pix
        ## read all hiz qsos
        if mode == "pix":
            t0 = time.time()
            pix_data = read_from_pix(in_dir,pix,thingid[w], ra[w], dec[w], z_qso[w], plate[w], mjd[w], fiberid[w], order, log=log)
            read_time=time.time()-t0
        elif mode == "spec-mock-1D":
            t0 = time.time()
            pix_data = read_from_mock_1D(in_dir,thingid[w], ra[w], dec[w], z_qso[w], plate[w], mjd[w], fiberid[w], order, mode=mode,log=log)
            read_time=time.time()-t0
        if not pix_data is None:
            userprint("{} read from pix {}, {} {} in {} secs per spectrum".format(len(pix_data),pix,i,len(upix),read_time/(len(pix_data)+1e-3)))
        if not pix_data is None and len(pix_data)>0:
            data[pix] = pix_data
            ndata += len(pix_data)

    return data,ndata,nside,"RING"

def read_from_spec(in_dir,thingid,ra,dec,z_qso,plate,mjd,fiberid,order,mode,log=None,pk1d=None,best_obs=None):
    drq_dict = {t:(r,d,z) for t,r,d,z in zip(thingid,ra,dec,z_qso)}

    ## if using multiple observations,
    ## then replace thingid, plate, mjd, fiberid
    ## by what's available in spAll
    if not best_obs:
        fi = glob.glob(in_dir.replace("spectra/","").replace("lite","").replace("full","")+"/spAll-*.fits")
        if len(fi) > 1:
            userprint("ERROR: found multiple spAll files")
            userprint("ERROR: try running with --bestobs option (but you will lose reobservations)")
            for f in fi:
                userprint("found: ",fi)
            sys.exit(1)
        if len(fi) == 0:
            userprint("ERROR: can't find required spAll file in {}".format(in_dir))
            userprint("ERROR: try runnint with --best-obs option (but you will lose reobservations)")
            sys.exit(1)

        spAll = fitsio.FITS(fi[0])
        userprint("INFO: reading spAll from {}".format(fi[0]))
        thingid_spall = spAll[1]["THING_ID"][:]
        plate_spall = spAll[1]["PLATE"][:]
        mjd_spall = spAll[1]["MJD"][:]
        fiberid_spall = spAll[1]["FIBERID"][:]
        qual_spall = spAll[1]["PLATEQUALITY"][:].astype(str)
        zwarn_spall = spAll[1]["ZWARNING"][:]

        w = sp.in1d(thingid_spall, thingid) & (qual_spall == "good")
        ## Removing spectra with the following ZWARNING bits set:
        ## SKY, LITTLE_COVERAGE, UNPLUGGED, BAD_TARGET, NODATA
        ## https://www.sdss.org/dr14/algorithms/bitmasks/#ZWARNING
        for zwarnbit in [0,1,7,8,9]:
            w &= (zwarn_spall&2**zwarnbit)==0
        userprint("INFO: # unique objs: ",len(thingid))
        userprint("INFO: # spectra: ",w.sum())
        thingid = thingid_spall[w]
        plate = plate_spall[w]
        mjd = mjd_spall[w]
        fiberid = fiberid_spall[w]
        spAll.close()

    ## to simplify, use a list of all metadata
    allmeta = []
    ## Used to preserve original order and pass unit tests.
    t_list = []
    t_set = set()
    for t,p,m,f in zip(thingid,plate,mjd,fiberid):
        if t not in t_set:
            t_list.append(t)
            t_set.add(t)
        r,d,z = drq_dict[t]
        meta = Metadata()
        meta.thingid = t
        meta.ra = r
        meta.dec = d
        meta.z_qso = z
        meta.plate = p
        meta.mjd = m
        meta.fiberid = f
        meta.order = order
        allmeta.append(meta)

    pix_data = []
    thingids = {}


    for meta in allmeta:
        t = meta.thingid
        if not t in thingids:
            thingids[t] = []
        thingids[t].append(meta)

    userprint("reading {} thingids".format(len(thingids)))

    for t in t_list:
        t_delta = None
        for meta in thingids[t]:
            r,d,z,p,m,f = meta.ra,meta.dec,meta.z_qso,meta.plate,meta.mjd,meta.fiberid
            try:
                fin = in_dir + "/{}/{}-{}-{}-{:04d}.fits".format(p,mode,p,m,f)
                hdul = fitsio.FITS(fin)
            except IOError:
                log.write("error reading {}\n".format(fin))
                continue
            log.write("{} read\n".format(fin))
            log_lambda = hdul[1]["loglam"][:]
            flux = hdul[1]["flux"][:]
            ivar = hdul[1]["ivar"][:]*(hdul[1]["and_mask"][:]==0)

            if pk1d is not None:
                # compute difference between exposure
                exposures_diff = exp_diff(hdul,log_lambda)
                # compute spectral resolution
                wdisp =  hdul[1]["wdisp"][:]
                reso = spectral_resolution(wdisp,True,f,log_lambda)
            else:
                exposures_diff = None
                reso = None
            deltas = Forest(log_lambda,flux,ivar, t, r, d, z, p, m, f,order,exposures_diff=exposures_diff,reso=reso)
            if t_delta is None:
                t_delta = deltas
            else:
                t_delta += deltas
            hdul.close()
        if t_delta is not None:
            pix_data.append(t_delta)
    return pix_data

def read_from_mock_1D(in_dir,thingid,ra,dec,z_qso,plate,mjd,fiberid, order,mode,log=None):
    pix_data = []

    try:
        fin = in_dir
        hdul = fitsio.FITS(fin)
    except IOError:
        log.write("error reading {}\n".format(fin))

    for t,r,d,z,p,m,f in zip(thingid,ra,dec,z_qso,plate,mjd,fiberid):
        hdu = hdul["{}".format(t)]
        log.write("file: {} hdus {} read  \n".format(fin, hdu))
        lamb = hdu["wavelength"][:]
        log_lambda = sp.log10(lamb)
        flux = hdu["flux"][:]
        error =hdu["error"][:]
        ivar = 1.0/error**2

        # compute difference between exposure
        exposures_diff = np.zeros(len(lamb))
        # compute spectral resolution
        wdisp =  hdu["psf"][:]
        reso = spectral_resolution(wdisp)

        # compute the mean expected flux
        f_mean_tr = hdu.read_header()["MEANFLUX"]
        cont = hdu["continuum"][:]
        mef = f_mean_tr * cont
        d = Forest(log_lambda,flux,ivar, t, r, d, z, p, m, f,order, exposures_diff,reso, mef)
        pix_data.append(d)

    hdul.close()

    return pix_data


def read_from_pix(in_dir,pix,thingid,ra,dec,z_qso,plate,mjd,fiberid,order,log=None):
        try:
            fin = in_dir + "/pix_{}.fits.gz".format(pix)
            hdul = fitsio.FITS(fin)
        except IOError:
            try:
                fin = in_dir + "/pix_{}.fits".format(pix)
                hdul = fitsio.FITS(fin)
            except IOError:
                userprint("error reading {}".format(pix))
                return None

        ## fill log
        if log is not None:
            for t in thingid:
                if t not in hdul[0][:]:
                    log.write("{} missing from pixel {}\n".format(t,pix))
                    userprint("{} missing from pixel {}".format(t,pix))

        pix_data=[]
        thingid_list=list(hdul[0][:])
        thingid2idx = {t:thingid_list.index(t) for t in thingid if t in thingid_list}
        loglam  = hdul[1][:]
        flux = hdul[2].read()
        ivar = hdul[3].read()
        andmask = hdul[4].read()
        for (t, r, d, z, p, m, f) in zip(thingid, ra, dec, z_qso, plate, mjd, fiberid):
            try:
                idx = thingid2idx[t]
            except:
                ## fill log
                if log is not None:
                    log.write("{} missing from pixel {}\n".format(t,pix))
                userprint("{} missing from pixel {}".format(t,pix))
                continue
            d = Forest(loglam,flux[:,idx],ivar[:,idx]*(andmask[:,idx]==0), t, r, d, z, p, m, f,order)

            if log is not None:
                log.write("{} read\n".format(t))
            pix_data.append(d)
        hdul.close()
        return pix_data

def read_from_spcframe(in_dir, thingid, ra, dec, z_qso, plate, mjd, fiberid, order, mode=None, log=None, best_obs=False, single_exp = False):

    if not best_obs:
        userprint("ERROR: multiple observations not (yet) compatible with spframe option")
        userprint("ERROR: rerun with the --best-obs option")
        sys.exit(1)

    allmeta = []
    for t,r,d,z,p,m,f in zip(thingid,ra,dec,z_qso,plate,mjd,fiberid):
        meta = Metadata()
        meta.thingid = t
        meta.ra = r
        meta.dec = d
        meta.z_qso = z
        meta.plate = p
        meta.mjd = m
        meta.fiberid = f
        meta.order = order
        allmeta.append(meta)
    platemjd = {}
    for i in range(len(thingid)):
        pm = (plate[i], mjd[i])
        if not pm in platemjd:
            platemjd[pm] = []
        platemjd[pm].append(allmeta[i])

    pix_data={}
    userprint("reading {} plates".format(len(platemjd)))

    for pm in platemjd:
        p,m = pm
        exps = []
        spplate = in_dir+"/{0}/spPlate-{0}-{1}.fits".format(p,m)
        userprint("INFO: reading plate {}".format(spplate))
        hdul=fitsio.FITS(spplate)
        head = hdul[0].read_header()
        hdul.close()
        iexp = 1
        for c in ["B1", "B2", "R1", "R2"]:
            card = "NEXP_{}".format(c)
            if card in head:
                nexp = head["NEXP_{}".format(c)]
            else:
                continue
            for _ in range(nexp):
                str_iexp = str(iexp)
                if iexp<10:
                    str_iexp = '0'+str_iexp

                card = "EXPID"+str_iexp
                if not card in head:
                    continue

                exps.append(head["EXPID"+str_iexp][:11])
                iexp += 1

        userprint("INFO: found {} exposures in plate {}-{}".format(len(exps), p,m))

        if len(exps) == 0:
            continue

        exp_num = [e[3:] for e in exps]
        exp_num = np.unique(exp_num)
        sp.random.shuffle(exp_num)
        exp_num = exp_num[0]
        for exp in exps:
            if single_exp:
                if not exp_num in exp:
                    continue
            t0 = time.time()
            ## find the spectrograph number:
            spectro = int(exp[1])
            assert spectro == 1 or spectro == 2

            spcframe = fitsio.FITS(in_dir+"/{}/spCFrame-{}.fits".format(p, exp))

            flux = spcframe[0].read()
            ivar = spcframe[1].read()*(spcframe[2].read()==0)
            llam = spcframe[3].read()

            ## now convert all those fluxes into forest objects
            for meta in platemjd[pm]:
                if spectro == 1 and meta.fiberid > 500: continue
                if spectro == 2 and meta.fiberid <= 500: continue
                index =(meta.fiberid-1)%500
                t = meta.thingid
                r = meta.ra
                d = meta.dec
                z = meta.z_qso
                f = meta.fiberid
                order = meta.order
                d = Forest(llam[index],flux[index],ivar[index], t, r, d, z, p, m, f, order)
                if t in pix_data:
                    pix_data[t] += d
                else:
                    pix_data[t] = d
                if log is not None:
                    log.write("{} read from exp {} and mjd {}\n".format(t, exp, m))
            nread = len(platemjd[pm])

            userprint("INFO: read {} from {} in {} per spec. Progress: {} of {} \n".format(nread, exp, (time.time()-t0)/(nread+1e-3), len(pix_data), len(thingid)))
            spcframe.close()

    data = list(pix_data.values())
    return data

def read_from_spplate(in_dir, thingid, ra, dec, z_qso, plate, mjd, fiberid, order, log=None, best_obs=False):

    drq_dict = {t:(r,d,z) for t,r,d,z in zip(thingid,ra,dec,z_qso)}

    ## if using multiple observations,
    ## then replace thingid, plate, mjd, fiberid
    ## by what's available in spAll

    if not best_obs:
        fi = glob.glob(in_dir+"/spAll-*.fits")
        if len(fi) > 1:
            userprint("ERROR: found multiple spAll files")
            userprint("ERROR: try running with --bestobs option (but you will lose reobservations)")
            for f in fi:
                userprint("found: ",fi)
            sys.exit(1)
        if len(fi) == 0:
            userprint("ERROR: can't find required spAll file in {}".format(in_dir))
            userprint("ERROR: try runnint with --best-obs option (but you will lose reobservations)")
            sys.exit(1)

        spAll = fitsio.FITS(fi[0])
        userprint("INFO: reading spAll from {}".format(fi[0]))
        thingid_spall = spAll[1]["THING_ID"][:]
        plate_spall = spAll[1]["PLATE"][:]
        mjd_spall = spAll[1]["MJD"][:]
        fiberid_spall = spAll[1]["FIBERID"][:]
        qual_spall = spAll[1]["PLATEQUALITY"][:].astype(str)
        zwarn_spall = spAll[1]["ZWARNING"][:]

        w = sp.in1d(thingid_spall, thingid)
        userprint("INFO: Found {} spectra with required THING_ID".format(w.sum()))
        w &= qual_spall == "good"
        userprint("INFO: Found {} spectra with 'good' plate".format(w.sum()))
        ## Removing spectra with the following ZWARNING bits set:
        ## SKY, LITTLE_COVERAGE, UNPLUGGED, BAD_TARGET, NODATA
        ## https://www.sdss.org/dr14/algorithms/bitmasks/#ZWARNING
        bad_zwarnbit = {0:'SKY',1:'LITTLE_COVERAGE',7:'UNPLUGGED',8:'BAD_TARGET',9:'NODATA'}
        for zwarnbit,zwarnbit_str in bad_zwarnbit.items():
            w &= (zwarn_spall&2**zwarnbit)==0
            userprint("INFO: Found {} spectra without {} bit set: {}".format(w.sum(), zwarnbit, zwarnbit_str))
        userprint("INFO: # unique objs: ",len(thingid))
        userprint("INFO: # spectra: ",w.sum())
        thingid = thingid_spall[w]
        plate = plate_spall[w]
        mjd = mjd_spall[w]
        fiberid = fiberid_spall[w]
        spAll.close()

    ## to simplify, use a list of all metadata
    allmeta = []
    for t,p,m,f in zip(thingid,plate,mjd,fiberid):
        r,d,z = drq_dict[t]
        meta = Metadata()
        meta.thingid = t
        meta.ra = r
        meta.dec = d
        meta.z_qso = z
        meta.plate = p
        meta.mjd = m
        meta.fiberid = f
        meta.order = order
        allmeta.append(meta)

    pix_data = {}
    platemjd = {}
    for p,m,meta in zip(plate,mjd,allmeta):
        pm = (p,m)
        if not pm in platemjd:
            platemjd[pm] = []
        platemjd[pm].append(meta)


    userprint("reading {} plates".format(len(platemjd)))

    for pm in platemjd:
        p,m = pm
        spplate = in_dir + "/{0}/spPlate-{0}-{1}.fits".format(str(p).zfill(4),m)

        try:
            hdul = fitsio.FITS(spplate)
            head0 = hdul[0].read_header()
        except IOError:
            log.write("error reading {}\n".format(spplate))
            continue
        t0 = time.time()

        coeff0 = head0["COEFF0"]
        coeff1 = head0["COEFF1"]

        flux = hdul[0].read()
        ivar = hdul[1].read()*(hdul[2].read()==0)
        llam = coeff0 + coeff1*np.arange(flux.shape[1])

        ## now convert all those fluxes into forest objects
        for meta in platemjd[pm]:
            t = meta.thingid
            r = meta.ra
            d = meta.dec
            z = meta.z_qso
            f = meta.fiberid
            o = meta.order

            i = meta.fiberid-1
            d = Forest(llam,flux[i],ivar[i], t, r, d, z, p, m, f, o)
            if t in pix_data:
                pix_data[t] += d
            else:
                pix_data[t] = d
            if log is not None:
                log.write("{} read from file {} and mjd {}\n".format(t, spplate, m))
        nread = len(platemjd[pm])
        userprint("INFO: read {} from {} in {} per spec. Progress: {} of {} \n".format(nread, os.path.basename(spplate), (time.time()-t0)/(nread+1e-3), len(pix_data), len(thingid)))
        hdul.close()

    data = list(pix_data.values())
    return data

def read_from_desi(nside,in_dir,thingid,ra,dec,z_qso,plate,mjd,fiberid,order,pk1d=None):

    in_nside = int(in_dir.split('spectra-')[-1].replace('/',''))
    nest = True
    data = {}
    ndata = 0

    ztable = {t:z for t,z in zip(thingid,z_qso)}
    in_pixs = healpy.ang2pix(in_nside, sp.pi/2.-dec, ra,nest=nest)
    fi = np.unique(in_pixs)

    for i,f in enumerate(fi):
        path = in_dir+"/"+str(int(f/100))+"/"+str(f)+"/spectra-"+str(in_nside)+"-"+str(f)+".fits"

        userprint("\rread {} of {}. ndata: {}".format(i,len(fi),ndata))
        try:
            hdul = fitsio.FITS(path)
        except IOError:
            userprint("Error reading pix {}\n".format(f))
            continue

        ## get the quasars
        tid_qsos = thingid[(in_pixs==f)]
        plate_qsos = plate[(in_pixs==f)]
        mjd_qsos = mjd[(in_pixs==f)]
        fiberid_qsos = fiberid[(in_pixs==f)]
        if 'TARGET_RA' in hdul["FIBERMAP"].get_colnames():
            ra = hdul["FIBERMAP"]["TARGET_RA"][:]*sp.pi/180.
            de = hdul["FIBERMAP"]["TARGET_DEC"][:]*sp.pi/180.
        elif 'RA_TARGET' in hdul["FIBERMAP"].get_colnames():
            ## TODO: These lines are for backward compatibility
            ## Should be removed at some point
            ra = hdul["FIBERMAP"]["RA_TARGET"][:]*sp.pi/180.
            de = hdul["FIBERMAP"]["DEC_TARGET"][:]*sp.pi/180.
        pixs = healpy.ang2pix(nside, sp.pi / 2 - de, ra)
        #exp = h["FIBERMAP"]["EXPID"][:]
        #night = h["FIBERMAP"]["NIGHT"][:]
        #fib = h["FIBERMAP"]["FIBER"][:]
        in_tids = hdul["FIBERMAP"]["TARGETID"][:]

        specData = {}
        for spec in ['B','R','Z']:
            dic = {}
            try:
                dic['log_lambda'] = sp.log10(hdul['{}_WAVELENGTH'.format(spec)].read())
                dic['FL'] = hdul['{}_FLUX'.format(spec)].read()
                dic['IV'] = hdul['{}_IVAR'.format(spec)].read()*(hdul['{}_MASK'.format(spec)].read()==0)
                w = sp.isnan(dic['FL']) | sp.isnan(dic['IV'])
                for k in ['FL','IV']:
                    dic[k][w] = 0.
                dic['RESO'] = hdul['{}_RESOLUTION'.format(spec)].read()
                specData[spec] = dic
            except OSError:
                pass
        hdul.close()

        for t,p,m,f in zip(tid_qsos,plate_qsos,mjd_qsos,fiberid_qsos):
            wt = in_tids == t
            if wt.sum()==0:
                userprint("\nError reading thingid {}\n".format(t))
                continue

            d = None
            for tspecData in specData.values():
                ivar = tspecData['IV'][wt]
                flux = (ivar*tspecData['FL'][wt]).sum(axis=0)
                ivar = ivar.sum(axis=0)
                w = ivar>0.
                flux[w] /= ivar[w]
                if not pk1d is None:
                    reso_sum = tspecData['RESO'][wt].sum(axis=0)
                    reso_in_km_per_s = spectral_resolution_desi(reso_sum,tspecData['log_lambda'])
                    exposures_diff = np.zeros(tspecData['log_lambda'].shape)
                else:
                    reso_in_km_per_s = None
                    exposures_diff = None
                td = Forest(tspecData['log_lambda'],flux,ivar,t,ra[wt][0],de[wt][0],ztable[t],
                    p,m,f,order,exposures_diff,reso_in_km_per_s)
                if d is None:
                    d = copy.deepcopy(td)
                else:
                    d += td

            pix = pixs[wt][0]
            if pix not in data:
                data[pix]=[]
            data[pix].append(d)
            ndata+=1

    userprint("found {} quasars in input files\n".format(ndata))

    return data


def read_deltas(indir,nside,lambda_abs,alpha,zref,cosmo,nspec=None,no_project=False,from_image=None):
    '''
    reads deltas from indir
    fills the fields delta.z and multiplies the weights by (1+z)^(alpha-1)/(1+zref)^(alpha-1)
    returns data,zmin_pix
    '''

    fi = []
    indir = os.path.expandvars(indir)
    if from_image is None or len(from_image)==0:
        if len(indir)>8 and indir[-8:]=='.fits.gz':
            fi += glob.glob(indir)
        elif len(indir)>5 and indir[-5:]=='.fits':
            fi += glob.glob(indir)
        else:
            fi += glob.glob(indir+'/*.fits') + glob.glob(indir+'/*.fits.gz')
    else:
        for arg in from_image:
            if len(arg)>8 and arg[-8:]=='.fits.gz':
                fi += glob.glob(arg)
            elif len(arg)>5 and arg[-5:]=='.fits':
                fi += glob.glob(arg)
            else:
                fi += glob.glob(arg+'/*.fits') + glob.glob(arg+'/*.fits.gz')
    fi = sorted(fi)

    dels = []
    ndata = 0
    for i,f in enumerate(fi):
        userprint("\rread {} of {} {}".format(i,len(fi),ndata))
        if from_image is None:
            hdul = fitsio.FITS(f)
            dels += [Delta.from_fitsio(hdu) for hdu in hdul[1:]]
            hdul.close()
        else:
            dels += Delta.from_image(f)

        ndata = len(dels)
        if not nspec is None:
            if ndata>nspec:break

    ###
    if not nspec is None:
        dels = dels[:nspec]
        ndata = len(dels)

    userprint("\n")

    phi = [d.ra for d in dels]
    th = [sp.pi/2.-d.dec for d in dels]
    pix = healpy.ang2pix(nside,th,phi)
    if pix.size==0:
        raise AssertionError('ERROR: No data in {}'.format(indir))

    data = {}
    zmin = 10**dels[0].log_lambda[0]/lambda_abs-1.
    zmax = 0.
    for d,p in zip(dels,pix):
        if not p in data:
            data[p]=[]
        data[p].append(d)

        z = 10**d.log_lambda/lambda_abs-1.
        zmin = min(zmin,z.min())
        zmax = max(zmax,z.max())
        d.z = z
        if not cosmo is None:
            d.r_comov = cosmo.r_comoving(z)
            d.rdm_comov = cosmo.dm(z)
        d.weights *= ((1+z)/(1+zref))**(alpha-1)

        if not no_project:
            d.project()

    return data,ndata,zmin,zmax


def read_objects(drq,nside,zmin,zmax,alpha,zref,cosmo,keep_bal=True):
    objs = {}
    ra,dec,z_qso,thingid,plate,mjd,fiberid = read_drq(drq,zmin,zmax,keep_bal=True)
    phi = ra
    th = sp.pi/2.-dec
    pix = healpy.ang2pix(nside,th,phi)
    if pix.size==0:
        raise AssertionError()
    userprint("reading qsos")

    upix = np.unique(pix)
    for i,ipix in enumerate(upix):
        userprint("\r{} of {}".format(i,len(upix)))
        w=pix==ipix
        objs[ipix] = [QSO(t,r,d,z,p,m,f) for t,r,d,z,p,m,f in zip(thingid[w],ra[w],dec[w],z_qso[w],plate[w],mjd[w],fiberid[w])]
        for q in objs[ipix]:
            q.weights = ((1.+q.z_qso)/(1.+zref))**(alpha-1.)
            if not cosmo is None:
                q.r_comov = cosmo.r_comoving(q.z_qso)
                q.rdm_comov = cosmo.dm(q.z_qso)

    userprint("\n")

    return objs,z_qso.min()
