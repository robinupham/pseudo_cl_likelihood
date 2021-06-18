"""
Contains functions to convert the characteristic function of a probability distribution to the probability density
function.

This is based on https://stackoverflow.com/a/24077914 (credit: thomasfermi), which describes how to use a numpy FFT to
evaluate a continuous Fourier transform.

The main two functions are cf_to_pdf_1d and cf_to_pdf_nd. The others contain usage examples.
"""

import numpy as np
# import matplotlib.pyplot as plt # uncomment for plotting
# import scipy.stats as stats # uncomment for additional examples


def cf_to_pdf_1d(cf, t0, dt):
    """
    Converts a characteristic function phi(t) to a probability density function f(x).

    Parameters:
    - cf: The characteristic function as a function of t
    - t0: The first value of t at which the characteristic function has been evaluated
    - dt: The increment in t used

    Returns tuple of (x, pdf).
    """

    # Main part of the pdf is given by a FFT
    pdf = np.fft.fft(cf)

    # x is given by the FFT frequencies multiplied by some normalisation factor 2pi/dt
    x = np.fft.fftfreq(cf.size) * 2 * np.pi / dt

    # Multiply pdf by factor to account for the differences between numpy's FFT and a true continuous FT
    pdf *= dt * np.exp(1j * x * t0) / (2 * np.pi)

    # Take the real part of the pdf, and sort both x and pdf by x
    # x, pdf = list(zip(*sorted(zip(x, pdf.real))))
    x, pdf = list(zip(*sorted(zip(x, np.abs(pdf))))) # I have found this tends to be more numerically stable

    return (np.array(x), np.array(pdf))


def cf_to_pdf_nd(cf_grid, t0, dt, verbose=True):
    """
    Converts an n-dimensional joint characteristic function to the corresponding joint probability density function.

    Arguments:

        cf_grid : n-dimensional grid of the characteristic function

        t0 : length-n vector of t0 values for each dimension

        dt : length-n vector of dt values for each dimension
    """

    # 1. Use an n-dimensional FFT to obtain the FFT grid
    if verbose:
        print('Performing FFT')
    fft_grid = np.fft.fftshift(np.fft.fftn(cf_grid))

    # 2. Obtain the range of each dimension of x and use this to form a grid of x
    if verbose:
        print('Forming x grid')

    # Form the x range in each dimension
    x_ranges = []
    n_dim = fft_grid.ndim
    grid_shape = fft_grid.shape
    for dim in range(n_dim):
        x_ranges.append(np.fft.fftshift(np.fft.fftfreq(grid_shape[dim])) * 2 * np.pi / dt[dim])

    # Form the grid
    x_grid = np.stack(np.meshgrid(*x_ranges, indexing='ij'), axis=-1)

    # 3. Use the grid of x to calculate the x-dependent phase factor and multiply this by the FFT grid
    if verbose:
        print('Applying phase factor')
    t0_dot_x = np.tensordot(t0, x_grid, axes=[0, n_dim])
    phase_factor_grid = np.exp(-1j * t0_dot_x)

    fft_grid_with_phase_factor = phase_factor_grid * fft_grid

    # 4. Uniformly multiply the entire grid by the normalisation factor and take the absolute value
    if verbose:
        print('Applying normalisation factor')
    norm_factor = np.prod(dt) / ((2 * np.pi) ** n_dim)
    pdf_grid = np.abs(fft_grid_with_phase_factor * norm_factor)

    return (x_grid, pdf_grid)


def univariate_example():
    """
    Example usage - uncomment the various CFs to try them.

    See 19-03-12_multi_cf_to_pdf for multivariate examples.
    """

    # Define a range of t over which to calculate the CF
    t_lower_lim = -100
    t_upper_lim = -t_lower_lim
    dt = 0.1
    t = np.arange(t_lower_lim, t_upper_lim, dt)

    print('Steps:' % ((-2 * t_lower_lim) / dt))

    # Calculate the CF over this range of t:

    # # Standard normal
    # cf = np.exp(- (t ** 2) / 2.)

    # # Central Gaussian
    # variance = 1e-1
    # cf = np.exp(- variance * (t ** 2) / 2.)

    # General Gaussian
    mean = 42
    variance = 1e-1
    cf = np.exp(1j * mean * t - variance * (t ** 2) / 2.)

    # # Gamma distribution
    # theta = 2.0
    # k = 5
    # cf = (1 - theta * 1j * t) ** -k

    # Use the cf_to_pdf function to convert the CF to a PDF
    x, pdf = cf_to_pdf_1d(cf, t_lower_lim, dt)

    # # Plot Result
    # plt.scatter(x, pdf, color="r", s=10)
    # plt.plot(x, pdf, color="r")

    # # For comparison, plot the analytical solution
    # plt.plot(x, np.exp(-np.abs(x)) * np.sqrt(np.pi / 2), color='g') - The example given on stack overflow
    # plt.plot(x, np.exp(- x ** 2 / 2) / np.sqrt(2 * np.pi), linestyle=':') # Standard normal distribution
    # plt.plot(x, np.exp(- x ** 2 / (2 * variance)) / np.sqrt(2 * np.pi * variance)) # Central Gaussian
    # plt.plot(x, np.exp(- (x - mean) ** 2 / (2 * variance)) / np.sqrt(2 * np.pi * variance), linestyle=':') # Non-central Gaussian
    # plt.plot(x, stats.gamma.pdf(x, a=k, scale=theta)) # Gamma

    # plt.gca().set_xlim(-5, 5)
    # plt.show()


def multivariate_examples():
    """
    Some multivariate examples.

    Works in 2D for the following cases:
    1. Independent standard normal, t_start = -1e1, steps = 1e3
    2. As 1 but rho = 0.5
    3. As 2 but variances = (0.8, 1.4)
    4. As 3 but mean = (-0.5, 1.2)

    Works in 3D for the following cases:
    1. Independent standard normal, t_start = -20, steps = 50
    2. As 1 but with rho = (0.4, -0.3, 0.2)
    3. As 2 but with var = (1.2, 0.9, 1.1)
    4. As 3 but mean = (-0.5, 1.2, 0.4) - minor numerical issues with some slightly negative values which should be ~0
    but these disappear with steps = 100.
    5. As 4 but with slightly different t0s (from each other)

    Works in 4D for the following cases:
    1. Independent standard normal, t_start = -20, steps = 50
    2. As 1 but with rho = (0.1, 0.2, 0.3, -0.4, -0.5, -0.3)
    3. As 2 but with var = (1.1, 1.2, 0.9, 1.3)
    4. As 3 but mean = (-0.5, 0.4, 0.2, -0.3) - as in the 3D case there are some minor numerical issues at 0. These
    disappear at steps = 70.
    5. As 4 but with slightly different t0s (from each other)

    Works for a symmetric Laplace distribution in 2D.
    """

    # 1. Create some n-dimensional grid of t
    print('1. Forming t grid')
    t1_start = -10 # -1e2
    t1_stop = -t1_start
    t2_start = t1_start
    t2_stop = t1_stop

    steps = 100

    dt1 = (t1_stop - t1_start) / (steps - 1.)
    dt2 = (t2_stop - t2_start) / (steps - 1.)

    # # Additional dimensions as needed
    # dt3 = (t3_stop - t3_start) / (steps - 1.)
    # dt4 = (t4_stop - t4_start) / (steps - 1.)

    t0_vec = np.array([t1_start, t2_start])
    dt_vec = np.array([dt1, dt2])

    t_grid = np.stack(np.mgrid[t1_start:t1_stop:1j*steps, t2_start:t2_stop:1j*steps], axis=-1)

    # 2. Calculate the CF at every grid point

    print('2. Calculating CF grid')


    ###############################################################################################
    # Multivariate Gaussian:
    ###############################################################################################

    mu = np.zeros(2) # 2D central Gaussian
    # mu = np.zeros(4) # 4D central Gaussian
    # mu = np.array([-0.5, 0.4, 0.2, -0.3]) # 4D non-Central Gaussian

    # Correlation coefficients
    rho = 0.1 # 0.5
    # rho_12 = 0.1 #0.4
    # rho_13 = 0.2 #-0.3
    # rho_14 = 0.3
    # rho_23 = -0.4 #0.2
    # rho_24 = -0.5
    # rho_34 = -0.3

    sigma = np.array([[1, rho],
                      [rho, 2]])
    # sigma = np.array([[1.2, rho_12, rho_13],
    #                   [rho_12, 0.9, rho_23],
    #                   [rho_13, rho_23, 1.1]])
    # sigma = np.array([[1.1, rho_12, rho_13, rho_14],
    #                   [rho_12, 1.2, rho_23, rho_24],
    #                   [rho_13, rho_23, 0.9, rho_34],
    #                   [rho_14, rho_24, rho_34, 1.3]])

    exp_arg_part1 = 1j * np.tensordot(mu, t_grid, axes=[0, 2])

    # Elementwise matmul; could be sped up with einsum
    exp_arg_part2 = np.full_like(exp_arg_part1, np.nan)
    for t1_idx, t1 in enumerate(t_grid):
        for t2_idx, t in enumerate(t1):
            exp_arg_part2[t1_idx, t2_idx] = -0.5 * t @ sigma @ t

    cf_grid = np.exp(exp_arg_part1 + exp_arg_part2)

    x_grid, pdf_grid = cf_to_pdf_nd(cf_grid, t0_vec, dt_vec)

    # # Compare each point with the corresponding known PDF
    # mvg = stats.multivariate_normal(mean=mu, cov=sigma)

    ###############################################################################################
    # Multivariate Laplace:
    ###############################################################################################

    def laplace_pdf(x, mu, sigma):
        """
        Symmetric multivariate Laplace distribution pdf, from wikipedia - wiki suggests this only holds for mu = 0.
        """

        k = len(mu)
        v = (2 - k) / 2.
        part1 = 2. / ((2 * np.pi) ** (k / 2.) * np.linalg.det(sigma) ** 0.5)
        inv_sigma = np.linalg.inv(sigma)
        x_invsigma_x = sum(x[..., idx] * np.tensordot(row, x, axes=[0, 1]) for idx, row in enumerate(inv_sigma))
        part2 = (x_invsigma_x / 2.) ** (v / 2.)
        bessel_arg = (2 * x_invsigma_x) ** 0.5

        # Uncomment these to use Laplace pdf - requires scipy.special
        # part3 = special.kv(v, bessel_arg)
        # res = part1 * part2 * part3
        # return res

    # n_dims = 3

    # mu = np.zeros(3)
    # rho_12 = 0.2
    # rho_13 = 0.5
    # rho_23 = -0.1
    # sigma = np.array([[1., rho_12, rho_13],
    #                   [rho_12, 1., rho_23],
    #                   [rho_13, rho_23, 1.]])

    # cf_top = np.exp(1j * np.tensordot(mu, t_grid, axes=[0, n_dims]))
    # t_sigma_t = sum(t_grid[..., idx] * np.tensordot(row, t_grid, axes=[0, n_dims]) for idx, row in enumerate(sigma))
    # cf_bottom = 0.5 * t_sigma_t + 1
    # cf_grid = cf_top / cf_bottom

    # x_grid, pdf_grid = cf_to_pdf.multi_cf_to_pdf(cf_grid, t0_vec, dt_vec)


    ###############################################################################################
    # General code for the comparison:
    ###############################################################################################

    print('Comparing: step 1/3')
    pdf = pdf_grid.flatten()
    print('Comparing: step 2/3')
    flat_x = x_grid.reshape(-1, x_grid.shape[-1])
    print('Comparing: step 3/3')
    # correct = mvg.pdf(flat_x) # Compare with known multivariate Gaussian pdf
    # correct = laplace_pdf(flat_x, mu, sigma) # Known Laplace pdf

    # Add scatter plot
    print('Making plot')
    # plt.scatter(correct, pdf, alpha=0.4, s=25, edgecolors='none')

    # Add y = x line
    # plt.plot([np.amin(pdf), np.amax(pdf)], [np.amin(pdf), np.amax(pdf)], color='grey')

    # plt.xlabel('Known PDF')
    # plt.ylabel('PDF from CF')
