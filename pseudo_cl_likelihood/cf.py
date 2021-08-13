"""
Functions to calculate the pseudo-Cl characteristic function in 1 or 3 dimensions,
implementing eqn 28 of arXiv:1908.00795.
"""

import multiprocessing as mp
import numpy as np


def cf_1d(l, alm_cov, steps=100000):
    """
    Characteristic function for the marginal likelihood distribution of a single auto- pseudo-Cl.

    alm_cov is the pseudo-alm covariance matrix, for this Cl only.
    """

    # Form selection matrix and calculate eigenvalues of M @ cov
    m = np.diag(np.concatenate(([1], 2 * np.ones(2 * l)))) / (2 * l + 1.)
    evals = np.linalg.eigvals(m @ alm_cov)

    # Calculate FFT parameters - may need tweaking
    scale = np.sum(np.abs(evals))
    tmax = 400. / scale
    t = np.linspace(-tmax, tmax, steps - 1)

    # Calculate CF(t)
    cf = np.ones_like(t)
    for eigenval in evals:
        cf = np.multiply(cf, np.power(1 - 2j * eigenval * t, -0.5))

    return t, cf


def calculate_cf_row(i, t_grid_i, m_sigma):
    """
    Function to calculate a single row of the 3D characteristic function.
    """

    cf_row = np.full(t_grid_i.shape[0:2], 1, dtype=complex)

    t1_m1_sigma = t_grid_i[0, 0, 0] * m_sigma[0]
    for j in range(t_grid_i.shape[0]):
        t2_m2_sigma = t_grid_i[j, 0, 1] * m_sigma[1]
        for k in range(t_grid_i.shape[1]):
            t3_m3_sigma = t_grid_i[j, k, 2] * m_sigma[2]

            sum_ti_mi_sigma = t1_m1_sigma + t2_m2_sigma + t3_m3_sigma
            eigvals = np.linalg.eigvals(sum_ti_mi_sigma)

            for eigval in eigvals:
                cf_row[j, k] *= ((1 - 2j * eigval) ** -0.5)

    return (i, cf_row)


def cf_3d(t_grid, m, cov, verbose=True):
    """
    Calculate the 3D characteristic function, given:
        t_grid: 3D grid of t values, with last dimension being length-3 coordinate vector
        m: sequence of M matrices of the form of eqs. 26-27 of arXiv:1908.00795.
        cov: full pseudo-alm covariance matrix
    """

    m1_sigma = m[0] @ cov
    m2_sigma = m[1] @ cov
    m3_sigma = m[2] @ cov
    m_sigma = [m1_sigma, m2_sigma, m3_sigma]

    cf_grid = np.full(t_grid.shape[0:3], np.nan, dtype=complex)

    # Store the results
    def store_cf_row(result):
        i, row = result
        cf_grid[i] = row
        if verbose:
            print(f'Calculating CF {i + 1} / {t_grid.shape[0]}')

    pool = mp.Pool(processes=(mp.cpu_count() - 1))
    for i in range(t_grid.shape[0]):
        pool.apply_async(calculate_cf_row, args=(i, t_grid[i], m_sigma), callback=store_cf_row)
    pool.close()
    pool.join()

    return cf_grid


def examples_3d():
    """
    Examples of the M matrices and t grid for an EB analysis.
    """

    l = 2

    m_ee = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 2, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 2, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 2, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 2, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]) / (2 * l + 1.)

    m_bb = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 2, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 2, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 2, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 2]]) / (2 * l + 1.)

    m_eb = np.array([[0, 0, 0, 0, 0, .5, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                     [.5, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]]) / (2 * l + 1.)

    # Choose dt based on the desired range to cover, and t0 based on dt and the number of steps (FFT frequencies)
    max_ee = 0.1
    max_bb = 0.0125
    max_eb = 0.0125
    dt_ee = 0.45 * 2 * np.pi / max_ee
    dt_bb = 0.45 * 2 * np.pi / max_bb
    dt_eb = 0.45 * 2 * np.pi / max_eb

    steps_ee = 1000
    steps_bb = 1000
    steps_eb = 1000

    t0_ee = -0.5 * dt_ee * (steps_ee - 1)
    t0_bb = -0.5 * dt_bb * (steps_bb - 1)
    t0_eb = -0.5 * dt_eb * (steps_eb - 1)
    t_grid = np.stack(np.mgrid[t0_ee:-t0_ee:1j*steps_ee, t0_bb:-t0_bb:1j*steps_bb, t0_eb:-t0_eb:1j*steps_eb], axis=-1)
