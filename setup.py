from setuptools import setup

setup(name='pseudo_cl_likelihood',
      version='1.0',
      description='Pseudo-Cl likelihood',
      author='Robin Upham',
      url='https://github.com/robinupham/pseudo_cl_likelihood',
      packages=['pseudo_cl_likelihood'],
      package_dir = {'': '.'}
      install_requires=['numpy', 'healpy', 'pyshtools'],)
