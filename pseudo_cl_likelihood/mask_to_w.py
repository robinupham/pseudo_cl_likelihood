"""
Code to calculate W0, W+/- objects (harmonic space window functions) from a mask.

The standard workflow to use this would be:
    1. Call mask_to_w to generate all the per-l W files (both W0 and W+/- in one call) and save to disk.
    2. Call combine_w_files once each for W0 and W+/- to combine per-l files.

Note pyshtools Wigners have the following limitations: (from https://shtools.oca.eu/shtools/public/pywigner3j.html)
"Returned values have a relative error less than ~1e-8 when j2 and j3 are less than about 100.
In practice, this routine is probably usable up to about 165."
To go to higher l, use e.g. https://github.com/ntessore/wigner (not used here).
"""

import os
import healpy as hp
import numpy as np
import pyshtools


def mask_to_wlm(mask_path, lmax, verbose=True):
    """
    Load in a mask FITS file and calculate its alms using healpix.
    """

    mask = hp.fitsfunc.read_map(mask_path, verbose=verbose)
    return hp.sphtfunc.map2alm(mask, lmax=lmax)


def wigner_prefactor(l, lp, lpp):
    """
    Calculate the prefactor sqrt((2l + 1)(2l' + 1)(2l'' + 1) / 4pi)
    """

    return np.sqrt((2 * l + 1.) * (2 * lp + 1.) * (2 * lpp + 1.) / (4 * np.pi))


def w0_element(l, lp, m, mp, wlms, lmax):
    """
    Calculate a single element of the W0 object,
    following Hamimeche & Lewis 2009 eqn B2.
    """

    # Calculate Wigner symbols for all valid l''
    mpp = m - mp
    wigners_0, lpp_min1, lpp_max1 = pyshtools.utils.Wigner3j(l, lp, 0, 0, 0)
    wigners_m, lpp_min2, lpp_max2 = pyshtools.utils.Wigner3j(l, lp, mpp, -m, mp)

    # Pad the two arrays so that they run over the same range, and trim to not go beyond lmax
    lpp_min = min(lpp_min1, lpp_min2)
    lpp_max = max(lpp_max1, lpp_max2)
    wigners_0 = np.pad(wigners_0, (lpp_min1 - lpp_min, lpp_max - lpp_max1), 'constant')[:(lmax - lpp_min + 1)]
    wigners_m = np.pad(wigners_m, (lpp_min2 - lpp_min, lpp_max - lpp_max2), 'constant')[:(lmax - lpp_min + 1)]

    res = 0
    for lpp in range(lpp_min, min(lpp_max, lmax) + 1):

        # Calculate the negative-m Wlm from the positive one if necessary
        if mpp < 0:
            wlm = (-1) ** -mpp * np.conj(wlms[hp.sphtfunc.Alm.getidx(lmax, lpp, -mpp)])
        else:
            wlm = wlms[hp.sphtfunc.Alm.getidx(lmax, lpp, mpp)]

        res += wlm * wigner_prefactor(l, lp, lpp) * wigners_0[lpp - lpp_min] * wigners_m[lpp - lpp_min]

    return (-1) ** m * res


def wpm_element(l, lp, m, mp, wlms, lmax):
    """
    Calculate a single element of the W+ and W- objects,
    following Hamimeche & Lewis 2009 eq B15 adapted to take the i inside the W- definition.
    """

    # Calculate Wigner symbols for all valid l''
    mpp = m - mp
    wigners_m, lpp_min1, lpp_max1 = pyshtools.utils.Wigner3j(l, lp, mpp, -m, mp)
    wigners_p2, lpp_min2, lpp_max2 = pyshtools.utils.Wigner3j(l, lp, 0, 2, -2)
    wigners_m2, lpp_min3, lpp_max3 = pyshtools.utils.Wigner3j(l, lp, 0, -2, 2)

    # Pad the three arrays so that they run over the same range and trim to not go beyond lmax
    lpp_min = min(lpp_min1, lpp_min2, lpp_min3)
    lpp_max = max(lpp_max1, lpp_max2, lpp_max3)
    wigners_m = np.pad(wigners_m, (lpp_min1 - lpp_min, lpp_max - lpp_max1), 'constant')[:(lmax - lpp_min + 1)]
    wigners_p2 = np.pad(wigners_p2, (lpp_min2 - lpp_min, lpp_max - lpp_max2), 'constant')[:(lmax - lpp_min + 1)]
    wigners_m2 = np.pad(wigners_m2, (lpp_min3 - lpp_min, lpp_max - lpp_max3), 'constant')[:(lmax - lpp_min + 1)]

    res_plus = 0
    res_minus = 0
    for lpp in range(lpp_min, min(lpp_max, lmax) + 1):

        # Calculate the negative-m Wlm from the positive one if necessary
        if mpp < 0:
            wlppmpp = (-1) ** -mpp * np.conj(wlms[hp.sphtfunc.Alm.getidx(lmax, lpp, -mpp)])
        else:
            wlppmpp = wlms[hp.sphtfunc.Alm.getidx(lmax, lpp, mpp)]

        multiplier = (wlppmpp * wigner_prefactor(l, lp, lpp) * wigners_m[lpp - lpp_min])
        res_plus += multiplier * (wigners_p2[lpp - lpp_min] + wigners_m2[lpp - lpp_min])
        res_minus += multiplier * 1j * (wigners_p2[lpp - lpp_min] - wigners_m2[lpp - lpp_min])

    pm_fac = 0.5 * (-1) ** m
    return pm_fac * res_plus, pm_fac * res_minus


def mask_to_w(mask_path, lmax, save_dir, do_w0=True, do_wpm=True, l_start=0, verbose=True):
    """
    For each l in turn, calculate all elements of the W+, W-, W0 objects from a path to a mask FITS file,
    following eqns B2 & B15 of Hamimeche & Lewis 2009.

    Saves one file per l for each of W0 and W+/-. These can be combined with the combine_w_files function.

    mask_path: path to mask FITS file
    lmax: maximum l to generate W object(s) to - implicitly any mixing to/from l > lmax is neglected
    save_dir: directory to save results to
    do_w0: calculate W0 object for spin-0 fields
    do_wpm: calculate W+/- objects for spin-2 fields
    l_start: minimum l to generate W object(s) from - mostly intended for batching

    Output is indexed [m + l, lpmp_idx] where lpmp_idx uses healpix indexing with lmax=l.
    """

    # Generate Wlms
    wlms = mask_to_wlm(mask_path, lmax, verbose=verbose)

    # Calculate the required  W objects for each l
    for l in range(l_start, lmax + 1):
        if verbose:
            print(f'Calculating W objects l = {l} / {lmax}')

        if do_w0:
            w0_l = np.full((2 * l + 1, hp.sphtfunc.Alm.getsize(l)), np.nan, dtype=complex)
        else:
            w0_l = None
        if do_wpm:
            wplus_l = np.full((2 * l + 1, hp.sphtfunc.Alm.getsize(l)), np.nan, dtype=complex)
            wminus_l = wplus_l.copy()
        else:
            wplus_l = wminus_l = None

        # Loop over m, l' and m'
        for m in range(-l, l + 1):
            m_idx = m + l
            for lp in range(l + 1):
                for mp in range(lp + 1):
                    lpmp_idx = hp.sphtfunc.Alm.getidx(l, lp, mp)

                    if do_w0:
                        w0_l[m_idx, lpmp_idx] = w0_element(l, lp, m, mp, wlms, lmax)

                    if do_wpm:
                        wplus_lmlpmp, wminus_lmlpmp = wpm_element(l, lp, m, mp, wlms, lmax)
                        wplus_l[m_idx, lpmp_idx] = wplus_lmlpmp
                        wminus_l[m_idx, lpmp_idx] = wminus_lmlpmp

        # Save the result
        if do_w0:
            filename = os.path.join(save_dir, f'w0_{l}.npz')
            np.savez_compressed(filename, w0=w0_l, l=l, lmax=lmax)
            if verbose:
                print('Saved ' + filename)
        if do_wpm:
            filename = os.path.join(save_dir, f'wpm_{l}.npz')
            np.savez_compressed(filename, wplus=wplus_l, wminus=wminus_l, l=l, lmax=lmax)
            if verbose:
                print('Saved ' + filename)


def combine_w_files(filemask, save_path, l_start, verbose=True):
    """
    Combine a bunch of per-l W files into a single file containing a list indexed as [l - l_start].

    filemask is the mask for input with the placeholder {l}, e.g. /path/to/w0_{l}.npz
    """

    # Loop over all consecutive files matching the mask and add them to a single list
    w0 = []
    wp = []
    wm = []
    l = l_start
    lmax = None
    while True:
        w_path = filemask.format(l=l)
        if not os.path.exists(w_path):
            break

        with np.load(w_path) as this_w:

            # Do some consistency checks
            assert this_w['l'] == l
            if lmax is None:
                lmax = this_w['lmax']
            else:
                assert this_w['lmax'] == lmax

            # Extract the relevant files
            if 'w0' in this_w.files:
                w0.append(this_w['w0'])
            elif 'wplus' in this_w.files and 'wminus' in this_w.files:
                wp.append(this_w['wplus'])
                wm.append(this_w['wminus'])
            else:
                err = f'Neither W0 nor W+/- present in {w_path}. Available files are: {this_w.files}.'
                raise ValueError(err)

        if verbose:
            print('Loaded ' + w_path)
        l += 1

    if w0:
        if verbose:
            print('Saving...')
        np.savez_compressed(save_path, w0=w0, lmax=lmax, l_start=l_start)
        if verbose:
            print('Saved ' + save_path)

    elif wp and wm:
        if verbose:
            print('Saving...')
        np.savez_compressed(save_path, wplus=wp, wminus=wm, lmax=lmax, l_start=l_start)
        if verbose:
            print('Saved ' + save_path)
