# -*- encoding: utf-8 -*-

from setuptools import setup, find_packages

setup(
    name="htto",
    version=0.1,
    description="High Throughput Tomography framework",
    author="Jacob Williamson and Daniil Kazantsev",
    author_email="scientificsoftware@diamond.ac.uk",
    url="https://github.com/dkazanc/HTTO",
    license="BSD 3-clause",  
    packages = find_packages(),
    requires=['numpy'],
    long_description="""
    A tool for reading tomographic data in parallel using MPI protocols, analyse and 
    reconstruct it using already available packages like TomoPy and ASTRA. 
    """,
)
