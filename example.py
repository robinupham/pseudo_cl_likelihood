"""
Very simple example of a full workflow.
"""

import alm_cov
import cf
import cf_to_pdf
import mask_to_w

# import matplotlib.pyplot as plt # uncomment to enable plotting


def example():
    """
    Full workflow for the marginal likelihood of a spin-0 auto-Cl (l=2).
    """

    # 1. Calculate harmonic space window function
    mask_path = '/path/to/mask.fits'
    w_save_dir = '/xyz/'
    lmax = 10
    mask_to_w.mask_to_w(mask_path, lmax, w_save_dir, do_w0=True, do_wpm=False)

    # 2. Combine W files into a single file
    filemask = '/xyz/w0_{l}.npz'
    w_save_path = '/xyz/w0.npz'
    mask_to_w.combine_w_files(filemask, w_save_path, l_start=0)

    # 3. Calculate pseudo-alm covariance matrix
    theory_cl_path = 'path/to/theory_cls.txt'
    theory_lmin = 2
    theory_cl = alm_cov.load_theory_cls(theory_cl_path, theory_lmin, lmax)
    w0_path = '/xyz/w0.npz'
    spins = [0]
    w_paths = [w0_path]
    theory_cls = [(0, '0', 0, '0', theory_cl)]
    l = 2
    cov = alm_cov.pseudo_alm_cov(spins, w_paths, theory_cls, lmax, lmax_out=l, lmin_out=l)

    # 4. Calculate characteristic function
    t, cf_ = cf.cf_1d(l, cov)

    # 5. Calculate likelihood distribution
    t0 = t[0]
    dt = t[1] - t[0]
    x, pdf = cf_to_pdf.cf_to_pdf_1d(cf_, t0, dt)

    # # Plot distribution - uncomment line including matplotlib at top
    # plt.plot(x, pdf)
    # plt.show()


main()
