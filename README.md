# HTTO stands for High Throughput Tomography pipeline
* A Python tool for parallel read of h5 tomographic data using MPI protocols
* The data can be re-chunked, saved and re-loaded (e.g. projection or sinogram-wise)
* The data is then can be processed by other tomographic packages, such as TomoPy, ASTRA and others

### Installation of software prerequisites:
* Due to a significant number of dependencies such as: CuPy, PyTorch, Cudatoolkit, etc. it is easier to conda install from the explicit [list](https://github.com/dkazanc/HTTO/blob/master/conda/specs_env_explicit.txt) as: 
```
conda create --name htto --file conda/specs_env_explicit.txt
```
* This will create the **htto** codna environment which you can activate with `conda activate htto`

