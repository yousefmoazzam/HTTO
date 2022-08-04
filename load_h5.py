"""
Parallel HDF5 load/read.

Splits data between processes in different ways.
Prints and/or collects time taken to read data in each of these ways.

Usage:
mpirun -np 4 python load_h5.py /path/to/datafile.h5 -p /entry/data -c /path/to/csvfile.csv -r 5 -s p
-p - path to dataset within data file. /entry1/tomo_entry/data/data by default.
-c - add results to a csv file (will create if file doesn't exist).
-r - repeat multiple times.
-s - slicing direction(s) to use. Projections by default

Author: Jacob Williamson

"""

from mpi4py import MPI
import h5py as h5
import argparse
import math
import pandas as pd
import os
from datetime import datetime


def __option_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("in_file", help="Input data file.")
    parser.add_argument("-p", "--path", default="/entry1/tomo_entry/data/data", help="Data path.")
    parser.add_argument("-c", "--csv", default=None, help="Write results to specified csv file.")
    parser.add_argument("-r", "--repeat", type=int, default=1, help="Number of repeats.")
    parser.add_argument("-s", "--slicing", type=list, default=["p"],
                        help="Which slicing options to use (c, p, s, t).")
    args = parser.parse_args()
    return args


def load_data(file, dim, path, comm=MPI.COMM_WORLD, preview=":,:,:"):
    if dim == 1:
        data = read_through_dim1(file, path, comm, preview)
    elif dim == 2:
        data = read_through_dim2(file, path, comm, preview)
    elif dim == 3:
        data = read_through_dim3(file, path, comm, preview)
    else:
        raise Exception("Invalid dimension. Choose 1, 2 or 3.")
    return data


def read_through_dim3(file, path, comm, preview=":,:,:"):
    rank = comm.rank
    nproc = comm.size
    slice_list = get_slice_list_from_preview(preview)
    with h5.File(file, "r", driver="mpio", comm=comm) as in_file:
        dataset = in_file[path]
        if slice_list[2] == slice(None):
            length = dataset.shape[2]
            offset = 0
            step = 1
        else:
            start = 0 if slice_list[2].start is None else slice_list[1].start
            stop = dataset.shape[2] if slice_list[2].stop is None else slice_list[2].stop
            step = 1 if slice_list[2].step is None else slice_list[2].step
            length = (stop - start) // step
            offset = start
        i0 = round((length / nproc) * rank) + offset
        i1 = round((length / nproc) * (rank + 1)) + offset
        proc_data = dataset[:, :, i0:i1:step]
        return proc_data


def read_through_dim2(file, path, comm, preview=":,:,:"):
    rank = comm.rank
    nproc = comm.size
    slice_list = get_slice_list_from_preview(preview)
    with h5.File(file, "r", driver="mpio", comm=comm) as in_file:
        dataset = in_file[path]
        if slice_list[1] == slice(None):
            length = dataset.shape[1]
            offset = 0
            step = 1
        else:
            start = 0 if slice_list[1].start is None else slice_list[1].start
            stop = dataset.shape[1] if slice_list[1].stop is None else slice_list[1].stop
            step = 1 if slice_list[1].step is None else slice_list[1].step
            length = (stop - start)//step
            offset = start
        i0 = round((length / nproc) * rank) + offset
        i1 = round((length / nproc) * (rank + 1)) + offset
        proc_data = dataset[:, i0:i1:step, :]
        return proc_data


def read_through_dim1(file, path, comm, preview=":,:,:"):
    rank = comm.rank
    nproc = comm.size
    slice_list = get_slice_list_from_preview(preview)
    with h5.File(file, "r", driver="mpio", comm=comm) as in_file:
        dataset = in_file[path]
        if slice_list[0] == slice(None):
            length = dataset.shape[0]
            offset = 0
            step = 1
        else:
            start = 0 if slice_list[0].start is None else slice_list[0].start
            stop = dataset.shape[0] if slice_list[0].stop is None else slice_list[0].stop
            step = 1 if slice_list[0].step is None else slice_list[0].step
            length = (stop - start)//step
            offset = start
        i0 = round((length / nproc) * rank) + offset
        i1 = round((length / nproc) * (rank + 1)) + offset
        proc_data = dataset[i0:i1:step, slice_list[1], slice_list[2]]
        return proc_data


def read_chunks(file, path, comm):
    rank = comm.rank
    nproc = comm.size
    with h5.File(file, "r", driver="mpio", comm=comm) as in_file:
        dataset = in_file[path]
        shape = dataset.shape
        chunks = dataset.chunks

    chunk_boundaries = [[None] * (math.ceil(shape[i] / chunks[i]) + 1) for i in range(len(shape))]

    # Creating a list of chunk boundaries in each dimension.
    for dim in range(len(shape)):
        boundary = 0
        for i in range(len(chunk_boundaries[dim])):
            if boundary > shape[dim]:
                boundary = shape[dim]
            chunk_boundaries[dim][i] = boundary
            boundary += chunks[dim]

    # Calculating number of chunks
    nchunks = 1
    for dim in range(len(chunk_boundaries)):
        nchunks *= (len(chunk_boundaries[dim]) - 1)
    chunk_slice_lists = [None for i in range(nchunks)]

    # Create a slice list for each chunk from the chunk boundaries.
    count = 0
    for i in range(1, len(chunk_boundaries[0])):
        for j in range(1, len(chunk_boundaries[1])):
            for k in range(1, len(chunk_boundaries[2])):
                chunk_slice_lists[count] = [slice(chunk_boundaries[0][i - 1], chunk_boundaries[0][i]),
                                            slice(chunk_boundaries[1][j - 1], chunk_boundaries[1][j]),
                                            slice(chunk_boundaries[2][k - 1], chunk_boundaries[2][k])]
                count += 1

    # Splitting chunks between each process.
    i0 = round((nchunks / nproc) * rank)
    i1 = round((nchunks / nproc) * (rank + 1))

    # Reading each chunk from the dataset using the slice lists.
    with h5.File(file, "r", driver="mpio", comm=MPI.COMM_WORLD) as in_file:
        dataset = in_file[path]
        for chunk_no in range(i0, i1):
            proc_data = dataset[chunk_slice_lists[chunk_no][0],  # slices 0th dimension
                                chunk_slice_lists[chunk_no][1],  # slices 1st dimension
                                chunk_slice_lists[chunk_no][2]]  # slices 2nd dimension
            # to produce a 3d chunk
    return nchunks


def get_angles(file, path="/entry1/tomo_entry/data/rotation_angle", comm=MPI.COMM_WORLD):
    with h5.File(file, "r", driver="mpio", comm=comm) as file:
        angles = file[path][...]
    return angles


def get_darks_flats(file, data_path="/entry1/tomo_entry/data/data",
                    image_key_path="/entry1/instrument/image_key/image_key", comm=MPI.COMM_WORLD, preview=":,:,:"):
    slice_list = get_slice_list_from_preview(preview)
    with h5.File(file, "r", driver="mpio", comm=comm) as file:
        darks_indices = []
        flats_indices = []
        for i, key in enumerate(file[image_key_path]):
            if int(key) == 1:
                flats_indices.append(i)
            elif int(key) == 2:
                darks_indices.append(i)
        darks = [file[data_path][x][slice_list[1]][slice_list[2]] for x in darks_indices]
        flats = [file[data_path][x][slice_list[1]][slice_list[2]] for x in flats_indices]
        return darks, flats


def get_data_indices(file, image_key_path="/entry1/instrument/image_key/image_key", comm=MPI.COMM_WORLD):
    with h5.File(file, "r", driver="mpio", comm=comm) as file:
        data_indices = []
        for i, key in enumerate(file[image_key_path]):
            if int(key) == 0:
                data_indices.append(i)
    return data_indices


def get_slice_list_from_preview(preview):
    slice_list = [None] * 3
    preview = preview.split(",")
    for dimension, value in enumerate(preview):
        values = value.split(":")
        new_values = [None if x.strip() == '' else int(x) for x in values]
        if len(values) == 1:
            slice_list[dimension] = slice(new_values[0])
        elif len(values) == 2:
            slice_list[dimension] = slice(new_values[0], new_values[1])
        elif len(values) == 3:
            slice_list[dimension] = slice(new_values[0], new_values[1], new_values[2])
    return slice_list


def main():
    args = __option_parser()
    rank = MPI.COMM_WORLD.rank
    nproc = MPI.COMM_WORLD.size

    filename = args.in_file.split("/")[-1]

    with h5.File(args.in_file, "r", driver="mpio", comm=MPI.COMM_WORLD) as in_file:
        size = in_file[args.path].size
        chunks = in_file[args.path].chunks
        shape = in_file[args.path].shape
        compression = in_file[args.path].compression
    if rank == 0:
        print()
        if args.repeat == 1:
            print(f"Reading {filename}")
        else:
            print(f"Reading {filename} {args.repeat} times")
        print(f"Number of processes: {nproc}")
        print(f"Dataset size = {size}")
        print(f"Dataset shape = {shape}")
        print(f"Chunk size = {chunks}")
        print(f"Compression = {compression}")
        print()

    times_dict = {"file name": [None] * args.repeat,
                  "data path": [None] * args.repeat,
                  "nproc": [None] * args.repeat,
                  "size": [None] * args.repeat,
                  "shape": [None] * args.repeat,
                  "chunks": [None] * args.repeat,
                  "chunks time (s)": [None] * args.repeat,
                  "projections time (s)": [None] * args.repeat,
                  "sinograms time (s)": [None] * args.repeat,
                  "tangentograms time (s)": [None] * args.repeat,
                  "date/time": [None] * args.repeat}

    for repeat in range(args.repeat):

        # Reading chunks
        MPI.COMM_WORLD.Barrier()
        if chunks is not None and "c" in args.include:
            tstart_c = MPI.Wtime()
            nchunks = read_chunks(args.in_file, args.path, MPI.COMM_WORLD)
            tstop_c = MPI.Wtime()
            time_c = tstop_c - tstart_c
            if rank == 0:
                print(f"{nchunks} chunks read in {time_c} seconds.")
        else:
            time_c = None

        # Reading along the 1st dimension (projections)
        MPI.COMM_WORLD.Barrier()
        if "p" in args.include:
            tstart_p = MPI.Wtime()
            read_through_dim1(args.in_file, args.path, MPI.COMM_WORLD)
            tstop_p = MPI.Wtime()
            time_p = tstop_p - tstart_p
            if rank == 0:
                print(f"{shape[0]} projections read in {time_p} seconds.")
        else:
            time_p = None

        # Reading along the 2nd dimension (sinograms)
        MPI.COMM_WORLD.Barrier()
        if "s" in args.include:
            tstart_s = MPI.Wtime()
            read_through_dim2(args.in_file, args.path, MPI.COMM_WORLD)
            tstop_s = MPI.Wtime()
            time_s = tstop_s - tstart_s
            if rank == 0:
                print(f"{shape[1]} sinograms read in {time_s} seconds.")
        else:
            time_s = None

        # Reading along the 3rd dimension (tangentograms)
        MPI.COMM_WORLD.Barrier()
        if "t" in args.include:
            tstart_t = MPI.Wtime()
            read_through_dim3(args.in_file, args.path, MPI.COMM_WORLD)
            tstop_t = MPI.Wtime()
            time_t = tstop_t - tstart_t
            if rank == 0:
                print(f"{shape[2]} tangentograms read in {time_t} seconds.")
        else:
            time_t = None

        # Recording results in a dictionary
        times_dict["file name"][repeat] = filename
        times_dict["data path"][repeat] = args.path
        times_dict["nproc"][repeat] = nproc
        times_dict["size"][repeat] = size
        times_dict["shape"][repeat] = str(shape)
        times_dict["chunks"][repeat] = str(chunks)
        times_dict["chunks time (s)"][repeat] = time_c
        times_dict["projections time (s)"][repeat] = time_p
        times_dict["sinograms time (s)"][repeat] = time_s
        times_dict["tangentograms time (s)"][repeat] = time_t
        times_dict["date/time"][repeat] = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

        if rank == 0:
            print()

    # Writing results to csv file if option used
    csv_path = args.csv
    if rank == 0:
        if csv_path is not None:
            times_df = pd.DataFrame(times_dict)
            times_df.to_csv(csv_path, mode="a", header=not os.path.exists(csv_path))
            print(f"Output written to {csv_path}")
            print()


if __name__ == '__main__':
    main()
