from larix.methods.misc import MEDIAN_FILT
from mpi4py.MPI import Comm
from numpy import ndarray


def filter_data(data: ndarray, flats: ndarray, darks: ndarray, comm: Comm):
    """Performs median filtration on the data using the larix library.

    Args:
        data: A numpy array containing the sample projections.
        flats: A numpy array containing the flatfield projections.
        darks: A numpy array containing the dark projections.
        comm: The MPI communicator to use.

    Returns:
        tuple[ndarray, ndarray, ndarray]: A tuple of numpy arrays containing the
            filtered projections, flatfields and darkfields.
    """
    kernel_size = 3  # full size kernel 3 x 3 x 3
    data = MEDIAN_FILT(data, kernel_size, comm.size)
    flats = MEDIAN_FILT(flats, kernel_size, comm.size)
    darks = MEDIAN_FILT(darks, kernel_size, comm.size)

    return data, flats, darks
