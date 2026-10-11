#!/usr/bin/env python3

from setuptools import setup, find_packages

setup(
    name='gpid',
    version='0.1.0',
    description='Algorithms for computing Partial Information Decompositions on Gaussian distributions',
    author='Praveen Venkatesh',
    url='https://github.com/praveenv253/gpid',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    python_requires='>=3.5',
    install_requires=['numpy', 'scipy', 'pandas', 'cvxpy', 'matplotlib', 'jupyter'],
    #setup_requires=['pytest-runner', ],
    #tests_require=['pytest', ],
    license='MIT',
)
