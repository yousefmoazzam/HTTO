HTTO (High Throughput TOmography pipeline)
******************************************

* A Python tool for parallel read of h5 tomographic data using MPI protocols
* The data can be re-chunked, saved and re-loaded (e.g. projection or sinogram-wise)
* The data is then can be processed by any tomographic packages, e.g. TomoPy, ASTRA

Setup a Development Environment:
================================

* Clone the repository from GitHub using :code:`git clone git@github.com:dkazanc/HTTO.git`
* Install dependencies from the environment file :code:`conda env create htto --file conda/environment.yml`
* Activate the environment with :code:`conda activate htto`
* Install the enviroment in development mode with :code:`python setup.py develop`

Install as a Python module
==========================

* Ensure all necessary dependencies are present in the environment
* Install the module with :code:`python setup.py install`

Running the code:
=================

* Install the module as described in "Install as a Python module"
* Execute the python module with :code:`python -m htto <args>`
* For help with the command line interface, execute :code:`python -m htto --help`
