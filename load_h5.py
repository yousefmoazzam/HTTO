from mpi4py import MPI
import h5py as h5
import argparse
import math
import pandas as pd
import os
from pathlib import Path


def __option_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("in_file", help="Input data file.")
    #parser.add_argument("out_folder", help="Output folder.")
    parser.add_argument("-p", "--path", help="Data path", default="/entry1/tomo_entry/data/data")
    parser.add_argument("-c", "--csv", help="Write read information to specified csv file", default=None)
    args = parser.parse_args()
    return args


def read_projections(args, rank, nproc):
    with h5.File(args.in_file, "r", driver="mpio", comm=MPI.COMM_WORLD) as in_file:
        dataset = in_file[args.path]
        shape = dataset.shape
        i0 = round((shape[2] / nproc) * rank)
        i1 = round((shape[2] / nproc) * (rank + 1))
        proc_data = dataset[:, :, i0:i1]


def read_sinograms(args, rank, nproc):
    with h5.File(args.in_file, "r", driver="mpio", comm=MPI.COMM_WORLD) as in_file:
        dataset = in_file[args.path]
        shape = dataset.shape
        i0 = round((shape[1] / nproc) * rank)
        i1 = round((shape[1] / nproc) * (rank + 1))
        proc_data = dataset[:, i0:i1, :]


def read_tangentograms(args, rank, nproc):
    with h5.File(args.in_file, "r", driver="mpio", comm=MPI.COMM_WORLD) as in_file:
        dataset = in_file[args.path]
        shape = dataset.shape
        i0 = round((shape[0] / nproc) * rank)
        i1 = round((shape[0] / nproc) * (rank + 1))
        proc_data = dataset[i0:i1, :, :]


def read_chunks(args, rank, nproc):
    with h5.File(args.in_file, "r", driver="mpio", comm=MPI.COMM_WORLD) as in_file:
        dataset = in_file[args.path]
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
    with h5.File(args.in_file, "r", driver="mpio", comm=MPI.COMM_WORLD) as in_file:
        dataset = in_file[args.path]
        for chunk_no in range(i0, i1):
            proc_data = dataset[chunk_slice_lists[chunk_no][0],  # slices 0th dimension
                                chunk_slice_lists[chunk_no][1],  # slices 1st dimension
                                chunk_slice_lists[chunk_no][2]]  # slices 2nd dimension
                                                                 # to produce a 3d chunk
    return nchunks


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
        print(f"Number of processes: {nproc}")
        print(f"Dataset size = {size}")
        print(f"Dataset shape = {shape}")
        print(f"Chunk size = {chunks}")
        print(f"Compression = {compression}")
    
    MPI.COMM_WORLD.Barrier()
    if chunks is not None:
        tstart_c = MPI.Wtime()
        nchunks = read_chunks(args, rank, nproc)
        tstop_c = MPI.Wtime()
        time_c = tstop_c - tstart_c
        if rank == 0:
            print(f"{nchunks} chunks read in {tstop_c - tstart_c} seconds.")
    else:
        time_c = None

    MPI.COMM_WORLD.Barrier()
    tstart_p = MPI.Wtime()
    read_projections(args, rank, nproc)
    tstop_p = MPI.Wtime()
    if rank == 0:
        print(f"{shape[2]} projections read in {tstop_p-tstart_p} seconds.")

    MPI.COMM_WORLD.Barrier()
    tstart_s = MPI.Wtime()
    read_sinograms(args, rank, nproc)
    tstop_s = MPI.Wtime()
    if rank == 0:
        print(f"{shape[1]} sinograms read in {tstop_s - tstart_s} seconds.")

    MPI.COMM_WORLD.Barrier()
    tstart_t = MPI.Wtime()
    read_tangentograms(args, rank, nproc)
    tstop_t = MPI.Wtime()
    if rank == 0:
        print(f"{shape[0]} tangentograms read in {tstop_t - tstart_t} seconds.")

    times_dict = {"filename": filename,
                  "data_path": args.path,
                  "nproc": nproc,
                  "size": size,
                  "shape": str(shape),
                  "chunks": str(chunks),
                  "chunks_time(s)": time_c,
                  "projections_time(s)": tstop_p - tstart_p,
                  "sinograms_time(s)": tstop_s - tstart_s,
                  "tangentograms_time(s)": tstop_t - tstart_t}

    csv_path = args.csv
    if rank == 0:
        if csv_path is not None:
            times_df = pd.DataFrame(times_dict, index=[0])
            times_df.to_csv(csv_path, mode="a", header=not os.path.exists(csv_path))
            print(f"Output written to {csv_path}")



if __name__ == '__main__':
    main()
