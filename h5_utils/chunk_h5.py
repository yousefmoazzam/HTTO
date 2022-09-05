"""
Parallel HDF5 save.

Methods to save parellalized data to file.

Author: Jacob Williamson
"""

from mpi4py import MPI
import h5py as h5
import argparse
import os.path

def __option_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("in_file", help="Input data file.")
    parser.add_argument("out_folder", help="Output folder.")
    parser.add_argument("-p", "--path", help="Data path", default="/entry1/tomo_entry/data/data")
    args = parser.parse_args()
    return args


def save_file_chunked(in_file, out_folder, chunks, path="/entry1/tomo_entry/data/data"):
    with h5.File(in_file, "r", driver="mpio", comm=MPI.COMM_WORLD) as my_file:
        dataset = my_file[path]
        shape = dataset.shape
        dtype = dataset.dtype
        new_filename = f"{in_file.split('/')[-1].split('.')[0]}_chunked.hdf"
        i = 0
        temp = new_filename
        new_filename = f"{out_folder}/{new_filename}"
        while os.path.isfile(temp):
            i += 1
            temp = new_filename.split(".")
            temp[0] += f"_{i}"
            temp = ".".join(temp)
        new_filename = temp
        new_file = h5.File(new_filename, "a")
        new_dataset = new_file.create_dataset(path, shape=shape, dtype=dtype, chunks=chunks, compression="gzip")
        new_dataset[...] = dataset[...]

    new_file.close()


def save_dataset(out_folder, file_name, data, slice_dim=1, chunks=(150, 150, 10), path="/data", comm=MPI.COMM_WORLD):
    """Save dataset in parallel.
    :param out_folder: Path to output folder.
    :param file_name: Name of file to save dataset in.
    :param data: Data to save to file.
    :param slice_dim: Where data has been parallelized (split into blocks, each of which is given to an MPI process),
        provide the dimension along which the data was sliced so it can be pieced together again.
    :param chunks: Specify how the data should be chunked when saved.
    :param path: Path to dataset within the file.
    :param comm: MPI communicator object.
    """
    shape = get_data_shape(data, slice_dim - 1, comm)
    dtype = data.dtype
    with h5.File(f"{out_folder}/{file_name}", "a", driver="mpio", comm=comm) as file:
        dataset = file.create_dataset(path, shape, dtype, chunks=chunks)
        save_data_parallel(dataset, data, slice_dim)


def save_data_parallel(dataset, data, slice_dim, comm=MPI.COMM_WORLD,):
    """Save data to dataset in parallel.
    :param dataset: Dataset to save data to.
    :param data: Data to save to dataset.
    :param slice_dim: Where data has been parallelized (split into blocks, each of which is given to an MPI process),
        provide the dimension along which the data was sliced so it can be pieced together again.
    :param comm: MPI communicator object.
    """
    rank = comm.rank
    nproc = comm.size
    length = dataset.shape[slice_dim - 1]
    i0 = round((length / nproc) * rank)
    i1 = round((length / nproc) * (rank + 1))
    if slice_dim == 1:
        dataset[i0:i1] = data[...]
    elif slice_dim == 2:
        dataset[:, i0:i1] = data[...]
    elif slice_dim == 3:
        dataset[:, :, i0:i1] = data[...]


def get_data_shape(data, dim, comm=MPI.COMM_WORLD):
    shape = list(data.shape)
    lengths = comm.gather(shape[dim], 0)
    lengths = comm.bcast(lengths, 0)
    shape[dim] = sum(lengths)
    shape = tuple(shape)
    return shape

