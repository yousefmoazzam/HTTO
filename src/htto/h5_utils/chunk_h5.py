import h5py as h5
from mpi4py import MPI


def save_dataset(
    out_folder,
    file_name,
    data,
    slice_dim=1,
    chunks=(150, 150, 10),
    path="/data",
    comm=MPI.COMM_WORLD,
):
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


def save_data_parallel(
    dataset,
    data,
    slice_dim,
    comm=MPI.COMM_WORLD,
):
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
