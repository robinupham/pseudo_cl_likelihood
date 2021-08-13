from setuptools import setup, find_packages

setup(name='pseudo_cl_likelihood',
      version='1.0',
      description='Pseudo-Cl likelihood',
      author='Robin Upham',
      url='https://github.com/robinupham/pseudo_cl_likelihood',
      packages=find_packages(),
      install_requires=['numpy', 'healpy', 'pyshtools'],)
