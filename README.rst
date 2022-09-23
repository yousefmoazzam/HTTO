HTTO (High Throughput TOmography pipeline)
******************************************

* A Python tool for parallel read of h5 tomographic data using MPI protocols
* The data can be re-chunked, saved and re-loaded (e.g. projection or sinogram-wise)
* The data is then can be processed by any tomographic packages, e.g. TomoPy, ASTRA

Setup a Development Environment:
================================

Using VScode Dev Containers
---------------------------

* Clone the repository from GitHub using :code:`git clone git@github.com:dkazanc/HTTO.git`
* Open the directory in ``VSCode`` and follow the prompts in the bottom right

Using Conda
-----------

* Clone the repository from GitHub using :code:`git clone git@github.com:dkazanc/HTTO.git`
* Install dependencies from the environment file :code:`conda env create htto --file conda/environment.yml`
* Activate the environment with :code:`conda activate htto`
* Install the enviroment in development mode with :code:`python setup.py develop`

Install as a Python module
==========================

* Ensure all nessacary dependencies are present in the environment (you may wish to refer to the Using Conda directions above)
* Install the module with :code:`python setup.py install`

Building the Container
======================

* Execute :code:`docker build . --tag htto`

Running the code:
=================

Using the python module
-----------------------

* Install the module as described in 
* Simply execute the python module with :code:`python -m htto <args>`
* For help with the command line interface simply execute :code:`python -m htto --help`

In a Container
--------------

* Build the container as described in Building the Container
* Run the container with :code:`docker run htto <args>`
* For help with the command line interface simply execute :code:`docker run htto python -m htto --help`
