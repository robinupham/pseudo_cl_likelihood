"""
General function (and many helper functions) to calculate the pseudo-alm covariance matrix.

Implements eqs 19-21 from arXiv:1908.00795.

Main function is pseudo_alm_cov.
See examples function for some basic usage examples.

Some notes about the limitations/optimality of this:

    - The computation of Eqs 19-21 could probably be significantly sped up by rewriting as a matrix/tensor
      multiplication and then using numpy vectorisation. They are currently implemented using pure-Python 3D loops.

    - You can't have only E-modes or B-modes from a spin-2 field, you have to have both. So say you were only
      interested in TT, EE and TE, this would also unnecessarily give you BB, TB and EB. These would then be ignored in
      the pdf calculation - no bias will result from this at all, just unnecessary computing time.
"""

import enum
import multiprocessing as mup

import healpy as hp
import numpy as np


class DecompField(enum.Enum):
    """
    Types of decomposed fields.
    """
    SPIN_0 = enum.auto()
    E_MODE = enum.auto()
    B_MODE = enum.auto()


def decomp_field_type(field_type):
    """
    Convert a decomposed field type string '0', 'E' or 'B' into the corresponding enum type.
    """

    if field_type == '0':
        return DecompField.SPIN_0
    if field_type == 'E':
        return DecompField.E_MODE
    if field_type == 'B':
        return DecompField.B_MODE

    raise ValueError(f'Field type {field_type} not recognised')


def theory_cl(field_1_decomp_idx, field_2_decomp_idx, l, theory_spectra, lmin=0):
    """
    Return the theory cross-power for fields 1 and 2 for multipole l.
    """

    # Recover the theory spectrum from the appropriate location
    if field_1_decomp_idx >= field_2_decomp_idx:
        theory_spectrum = theory_spectra[field_1_decomp_idx][field_2_decomp_idx]
    else:
        theory_spectrum = theory_spectra[field_2_decomp_idx][field_1_decomp_idx]

    # If the underlying spectrum is not provided (because it's zero), return 0
    if np.size(theory_spectrum) == 1 and theory_spectrum == 0:
        return 0

    # Otherwise return the appropriate power
    return theory_spectrum[l - lmin]


def get_derivative(top_field, bottom_field, ws):
    """
    Return the W object corresponding to a particular derivative in the general covariance equations, with a flag for W-
    due to its different symmetry and a multiplier (+/- 1).
    """

    # Extract field index and type
    top_field_idx, top_field_type = top_field
    bottom_field_idx, _ = bottom_field

    # If the fields are the same then the derivative is either W0 for spin-0 or W+ for spin-2
    if top_field == bottom_field:
        return ws[top_field_idx][0], False, 1

    # If the field indices are the same (i.e. one is E-mode and other is corresponding B-mode), return W-
    # with a minus sign for dB/dE
    if top_field_idx == bottom_field_idx:
        if top_field_type == DecompField.E_MODE:
            return ws[top_field_idx][1], True, 1
        else:
            return ws[top_field_idx][1], True, -1

    # Any other derivative is zero
    return 0


def get_w_element(l, lp, m, mp, w, w_minus=False):
    """
    Return a single element of a W object corresponding to (l, l', m, m').

    This function looks after all the indexing and symmetries of the W objects.
    """

    if lp <= l:
        if mp >= 0:
            # just directly return the element for (l, l', m, m')
            return w[l][m + l, hp.sphtfunc.Alm.getidx(l, lp, mp)]
        else:
            # use time reversal and return element (l, l', -m, -m') with appropriate plus or minus
            return (-1) ** (-m - mp) * np.conj(w[l][-m + l, hp.sphtfunc.Alm.getidx(l, lp, -mp)])
    else:
        pm = -1 if w_minus else 1 # factor -1 for W- when swapping indices
        if m >= 0:
            # use swap symmetry and return element (l', l, m', m) with appropriate plus or minus
            return pm * np.conj(w[lp][mp + lp, hp.sphtfunc.Alm.getidx(lp, l, m)])
        else:
            # use swap symmetry and time reversal and return element (l', l, -m', -m) with both appropriate pms
            return pm * (-1) ** (-mp - m) * w[lp][-mp + lp, hp.sphtfunc.Alm.getidx(lp, l, -m)]


def get_deriv_elmt(top_field, bot_field, top_l, bot_l, top_m, bot_m, ws):
    """
    Evaluate a specific derivative in the general covariance equations.
    """

    # Get the object corresponding to this derivative, which will be a W object (with possible factor -1)
    deriv = get_derivative(top_field, bot_field, ws)

    # If the derivative is just 0, return 0
    if deriv == 0:
        return 0

    # Otherwise use the W object and W- flag to calculate W element, and multiply by +/- 1 as appropriate
    w, is_w_minus, pm_fac = deriv
    return pm_fac * get_w_element(top_l, bot_l, top_m, bot_m, w, is_w_minus)


def l_cov(alpha_idx, alpha, beta_idx, beta, l, alpha_start_idx, beta_start_idx, field_block_size, fields_decomp, ws,
          theory_spectra, lmax, lmin=0, lmax_out=None, lmin_out=None):
    """
    Calculate a block of the covariance matrix corresponding to a single (α, β, l) combination.
    """

    # Allow lmax_out and lmin_out to default to lmax and lmin
    if lmax_out is None:
        lmax_out = lmax
    if lmin_out is None:
        lmin_out = lmin

    cov = np.full((2 * l + 1, field_block_size), np.nan)

    # Determine lp_max to make use of symmetry
    if alpha_idx == beta_idx:
        lp_max = l
    else:
        lp_max = lmax_out
    for lp in range(lmin_out, lp_max + 1):
        lp_start_idx = (lp - lmin_out) * (lp + lmin_out)
        for m in range(l + 1):

            # Calculate real and imag m indices
            if m == 0:
                re_m_idx = 0
            else:
                re_m_idx = 2 * m - 1
                im_m_idx = re_m_idx + 1

            # Determine mp_max to make use of symmetry
            if alpha_idx == beta_idx and l == lp:
                mp_max = m
            else:
                mp_max = lp
            for mp in range(mp_max + 1):

                # Calculate real and imag m' indices
                if mp == 0:
                    re_mp_idx = lp_start_idx
                else:
                    re_mp_idx = lp_start_idx + 2 * mp - 1
                    im_mp_idx = re_mp_idx + 1

                # Calculate the three general covariances following eqs 19-21 of the paper, plus im-re:

                cov_re_re = 0
                cov_im_im = 0
                cov_re_im = 0
                cov_im_re = 0

                for gamma_idx, gamma in enumerate(fields_decomp):
                    for epsilon_idx, epsilon in enumerate(fields_decomp):
                        for lpp in range(lmax + 1):

                            # Extract the theory gamma-epsilon cross-power for l'' and skip if power = 0
                            power = theory_cl(gamma_idx, epsilon_idx, lpp, theory_spectra, lmin)
                            if power == 0:
                                continue

                            # Each 'd' is the argument of a Re() or Im() in eqs 19-21

                            # m'' = 0
                            d1 = get_deriv_elmt(alpha, gamma, l, lpp, m, 0, ws)
                            d2 = get_deriv_elmt(beta, epsilon, lp, lpp, mp, 0, ws)
                            this_re_re = np.real(d1) * np.real(d2)
                            if mp > 0:
                                this_re_im = np.real(d1) * np.imag(d2)
                                if m > 0:
                                    this_im_im = np.imag(d1) * np.imag(d2)
                            if m > 0:
                                this_im_re = np.imag(d1) * np.real(d2)

                            # m'' > 0
                            for mpp in range(1, lpp + 1):
                                d3 = (np.conj(get_deriv_elmt(alpha, gamma, l, lpp, m, mpp, ws))
                                      * get_deriv_elmt(beta, epsilon, lp, lpp, mp, mpp, ws))
                                d4 = (np.conj(get_deriv_elmt(alpha, gamma, l, lpp, m, -mpp, ws))
                                      * get_deriv_elmt(beta, epsilon, lp, lpp, mp, -mpp, ws))
                                d5 = (get_deriv_elmt(alpha, gamma, l, lpp, m, mpp, ws)
                                      * get_deriv_elmt(beta, epsilon, lp, lpp, mp, -mpp, ws))
                                d6 = (get_deriv_elmt(alpha, gamma, l, lpp, m, -mpp, ws)
                                      * get_deriv_elmt(beta, epsilon, lp, lpp, mp, mpp, ws))
                                pm_fac = (-1) ** mpp

                                this_re_re += 0.5 * (np.real(d3) + np.real(d4)
                                                     + pm_fac * (np.real(d5) + np.real(d6)))
                                if mp > 0:
                                    this_re_im += 0.5 * (np.imag(d3) + np.imag(d4)
                                                         + pm_fac * (np.imag(d5) + np.imag(d6)))
                                    if m > 0:
                                        this_im_im += 0.5 * (np.real(d3) + np.real(d4)
                                                             - pm_fac * (np.real(d5) + np.real(d6)))
                                if m > 0:
                                    this_im_re -= 0.5 * (np.imag(d3) + np.imag(d4)
                                                         - pm_fac * (np.imag(d5) + np.imag(d6)))

                            cov_re_re += power * this_re_re
                            if mp > 0:
                                cov_re_im += power * this_re_im
                                if m > 0:
                                    cov_im_im += power * this_im_im
                            if m > 0:
                                cov_im_re += power * this_im_re

                # Store the three covariances
                cov[re_m_idx, re_mp_idx] = cov_re_re
                if mp > 0:
                    cov[re_m_idx, im_mp_idx] = cov_re_im
                    if m > 0:
                        cov[im_m_idx, im_mp_idx] = cov_im_im
                if m > 0:
                    cov[im_m_idx, re_mp_idx] = cov_im_re

    # Return block with its offset indices
    l_start_idx = (l - lmin_out) * (l + lmin_out) + alpha_start_idx
    return cov, l_start_idx, beta_start_idx


def pseudo_alm_cov(spins, w_paths, theory_cls, lmax, lmin=0, lmax_out=None, lmin_out=None, verbose=True):
    """
    General function to calculate the pseudo-alm covariance matrix.

    Arguments:

        spins - a list of the spin of each field in the problem (0 or 2), e.g. [0, 2, 2, 0]

        w_paths - a list of paths to W objects corresponding to each field in the spins list, as numpy npz files

        theory_cls - a list of underlying power spectra, each of which is a tuple of (field 1 index, field 1 0/E/B,
                     field 2 index, field 2 E/0/B, spectrum) where the field index refers to index in the spins list and
                     the spectrum indexed as [l - lmin],
                     e.g. [(0, '0', 1, 'E', [{l=lmin power}, {l=lmin+1 power}, ...]), (...)]

        lmax, lmin - boundaries of the underlying power to consider mixing from (must match those used to calculate the
                     W objects)

        lmax_out, lmin_out - range of multipoles to calculate covariance for (default same as lmax, lmin)
    """

    # Load W objects into a list of tuples
    # Spin-0 fields will have (W0) in a singleton tuple
    # Spin-2 fields will have (W+, W-)
    n_fields = len(spins)
    assert n_fields == len(w_paths), 'spins and w_paths have different lengths'
    ws = [np.nan]*n_fields
    for field_idx, (w_path, spin) in enumerate(zip(w_paths, spins)):
        if verbose:
            print('Loading W objects %s / %s' % (field_idx + 1, n_fields))
        with np.load(w_path, allow_pickle=True) as data:

            # Do some consistency checks with parameters
            assert 'lmin' not in data.files or lmin == data['lmin'], 'lmin mismatch with file ' + w_path
            assert lmax == data['lmax'], 'lmax mismatch with file ' + w_path

            # Store the relevant W object(s)
            if spin == 0:
                ws[field_idx] = (data['w0'],)
            elif spin == 2:
                ws[field_idx] = (data['wplus'], data['wminus'])
            else:
                # After this point we can assume that everything is either spin-0 or spin-2
                raise ValueError('Spin %s is not 0 or 2' % spin)

    # Decompose fields into 0, E, B
    # Each entry in the decomposed fields list will be a tuple of (field index, {spin-0/E/B})
    if verbose:
        print('Decomposing fields')
    fields_decomp = []
    for field_idx, spin in enumerate(spins):
        if spin == 0:
            fields_decomp.append((field_idx, DecompField.SPIN_0))
        else:
            fields_decomp.append((field_idx, DecompField.E_MODE))
            fields_decomp.append((field_idx, DecompField.B_MODE))

    # Load theory Cls into a 2D list, each dimension indexed by the decomposed field index
    # e.g. index [i][j] would be the cross-spectrum between decomposed fields i and j
    # Because of symmetry spectra are only stored for j <= i
    # Non-provided spectra are set to zero
    n_fields_decomp = len(fields_decomp)
    theory_spectra = [[0]*(1 + i) for i in range(n_fields_decomp)]
    for counter, (field_1_idx, field_1_type, field_2_idx, field_2_type, spectrum) in enumerate(theory_cls):
        if verbose:
            print('Loading theory Cls %s / %s' % (counter + 1, len(theory_cls)))

        # Work out the decomposed field indices from reverse lookup in the list
        field_1_decomp_idx = fields_decomp.index((field_1_idx, decomp_field_type(field_1_type)))
        field_2_decomp_idx = fields_decomp.index((field_2_idx, decomp_field_type(field_2_type)))

        # Store the spectrum, making use of symmetry
        if field_1_decomp_idx >= field_2_decomp_idx:
            theory_spectra[field_1_decomp_idx][field_2_decomp_idx] = spectrum
        else:
            theory_spectra[field_2_decomp_idx][field_1_decomp_idx] = spectrum

    # Create covariance matrix filled with placeholder nans
    # indexed following eqn 25 in the paper
    field_block_size = (lmax_out - lmin_out + 1) * (lmax_out + lmin_out + 1)
    cov_size = field_block_size * n_fields_decomp
    cov = np.full((cov_size, cov_size), np.nan)

    # Function to store blocks into the covariance matrix
    if verbose:
        counter = [0]
        n_blocks = int(0.5 * n_fields_decomp * (n_fields_decomp + 1)) * (lmax_out - lmin_out + 1)
    def store_cov(result):
        block, l_start_idx, beta_start_idx = result
        l_stop_idx = l_start_idx + block.shape[0]
        beta_stop_idx = beta_start_idx + block.shape[1]
        cov[l_start_idx:l_stop_idx, beta_start_idx:beta_stop_idx] = block
        if verbose:
            counter[0] += 1
            print('Calculated covariance matrix block %s / %s' % (counter[0], n_blocks))
            print(f'Calculated covariance matrix block {counter[0]} / {n_blocks}')

    # Loop over the six indices in the general covariance equations: α, β, l, l', m, m'
    # (latter 3 are in the l_cov function which is run in parallel)
    n_proc = mup.cpu_count() - 1
    if verbose:
        print(f'Running in parallel with {n_proc} cores')
    pool = mup.Pool(processes=n_proc)
    for alpha_idx, alpha in enumerate(fields_decomp):
        alpha_start_idx = alpha_idx * field_block_size
        for beta_idx, beta in enumerate(fields_decomp[:alpha_idx + 1]):
            beta_start_idx = beta_idx * field_block_size
            for l in range(lmin_out, lmax_out + 1):
                pool.apply_async(l_cov, args=(alpha_idx, alpha, beta_idx, beta, l, alpha_start_idx, beta_start_idx,
                                              field_block_size, fields_decomp, ws, theory_spectra, lmax, lmin,
                                              lmax_out, lmin_out),
                                 callback=store_cov)
    pool.close()
    pool.join()

    # Reflect matrix to get remaining elements
    if verbose:
        print('Obtaining remaining elements by symmetry')
    cov = np.where(np.isnan(cov), cov.T, cov)

    # Run some basic checks
    assert np.all(np.isfinite(cov)), 'Covariance matrix not finite'
    if verbose:
        print('Covariance matrix all finite')
    assert np.allclose(cov, cov.T), 'Covariance matrix not symmetric'
    if verbose:
        print('Covariance matrix is symmetric')
    assert np.all(np.linalg.eigvals(cov) >= 0), 'Covariance matrix not positive semi-definite'
    if verbose:
        print('Covariance matrix is positive semi-definite')

    return cov


def load_theory_cls(path, theory_lmin, lmax, lmin=0):
    """
    Load and return an array of theory Cls from a single spectrum, indexed such that element 0 is lmin.

    Arguments:

        path - path to the input text file

        theory_lmin - l corresponding to the first numeric line in the input file

        lmax, lmin - parameters relating to the likelihood calculation, which may be different to the input file
    """

    # Read in the raw theory spectrum, such that the first element is theory_lmin and the last element is lmax
    raw_spectrum = np.loadtxt(path, max_rows=(lmax - theory_lmin + 1))

    # Then either prepend zeros or trim the start such that the first element is now lmin
    if lmin >= theory_lmin:
        return raw_spectrum[lmin - theory_lmin:]
    else:
        return np.concatenate((np.zeros(theory_lmin - lmin), raw_spectrum))


def examples():
    """Example usage"""

    # Set l range here: lmax and lmin are the limits of the sums,
    # whereas between _out are the ones you actually want the distributions for
    lmax = 32
    lmin = 0
    lmin_out = 2
    lmax_out = 2

    # Load theory Cls
    theory_cl_tt_path = 'tt.txt'
    theory_cl_ee_path = 'ee.txt'
    theory_cl_bb_path = 'bb.txt'
    theory_cl_te_path = 'te.txt'
    theory_cl_tt = load_theory_cls(theory_cl_tt_path, 2, lmax, lmin)
    theory_cl_ee = load_theory_cls(theory_cl_ee_path, 2, lmax, lmin)
    theory_cl_bb = load_theory_cls(theory_cl_bb_path, 2, lmax, lmin)
    theory_cl_te = load_theory_cls(theory_cl_te_path, 2, lmax, lmin)

    # Paths to W0/+/- files, e.g. output from mask_to_w.py
    w0_path = '/path/to/w0.npz'
    wpm_path = '/path/to/wpm.npz'

    # Setup for a single TT spectrum
    spins = [0]
    w_paths = [w0_path]
    theory_cls = [(0, '0', 0, '0', theory_cl_tt)]

    # Setup for EB
    # spins = [2]
    # w_paths = [wpm_path]
    # theory_cls = [(0, 'E', 0, 'E', theory_cl_ee), (0, 'B', 0, 'B', theory_cl_bb)]

    # Setup for TEB
    # spins = [0, 2]
    # w_paths = [w0_path, wpm_path]
    # theory_cls = [(0, '0', 0, '0', theory_cl_tt),
    #               (1, 'E', 1, 'E', theory_cl_ee),
    #               (1, 'B', 1, 'B', theory_cl_bb),
    #               (0, '0', 1, 'E', theory_cl_te)]

    cov = pseudo_alm_cov(spins, w_paths, theory_cls, lmax, lmin, lmax_out, lmin_out)
