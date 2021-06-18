# Pseudo-Cl likelihood #

Python code implementing the exact pseudo-Cl likelihood for correlated Gaussian fields described in [arXiv:1908.00795](https://arxiv.org/abs/1908.00795).

## Example workflow ##

A simple example workflow is given in [example.py](example.py), which runs through the process of going from a mask FITS file to a likelihood distribution.

There are some other examples in the other files listed below.

## Summary of most useful functions ##

### [mask_to_w.py](mask_to_w.py) ###
* *mask_to_w*: Calculate the spin-0 and/or spin-2 harmonic space window functions (W objects) from a mask FITS file, and save to disk on a per-l basis.
* *combine_w_files*: Combine the output from *mask_to_w* into a single file for each W object.

### [alm_cov.py](alm_cov.py) ###
* *pseudo_alm_cov*: Generate a general pseudo-alm covariance matrix from theory Cls and W objects.

### [cf.py](cf.py) ###
* *cf_1d*: 1D characteristic function for an auto-Cl.
* *cf_3d*: generic 3D characteristic function.

### [cf_to_pdf.py](cf_to_pdf.py) ###
* *cf_to_pdf_1d*: Calculate probability density function from characteristic function in 1 dimension.
* *cf_to_pdf_nd*: N-dimensional equivalent.

## Requirements ##
* Python 3
* numpy
* [healpy](https://healpy.readthedocs.io/en/latest/)
* [pyshtools](https://pypi.org/project/pyshtools/)
