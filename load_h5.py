from mpi4py import MPI
import h5py as h5
import argparse


def __option_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("in_file", help="Input data file.")
    #parser.add_argument("out_folder", help="Output folder.")
    parser.add_argument("-p", "--path", help="Data path", default="/entry1/tomo_entry/data/data")
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


def main():
    args = __option_parser()
    rank = MPI.COMM_WORLD.rank
    nproc = MPI.COMM_WORLD.size

    tstart_p = MPI.Wtime()
    read_projections(args, rank, nproc)
    tstop_p = MPI.Wtime()

    tstart_s = MPI.Wtime()
    read_sinograms(args, rank, nproc)
    tstop_s = MPI.Wtime()

    tstart_t = MPI.Wtime()
    read_tangentograms(args, rank, nproc)
    tstop_t = MPI.Wtime()

    with h5.File(args.in_file, "r", driver="mpio", comm=MPI.COMM_WORLD) as in_file:
        size = in_file[args.path].size
        shape = in_file[args.path].shape

    if rank == 0:
        print(f"Number of processes: {nproc}.")
        print(f"Dataset has size {size}")
        print(f"{shape[2]} projections read in {tstop_p-tstart_p} seconds.")
        print(f"{shape[1]} sinograms read in {tstop_s - tstart_s} seconds.")
        print(f"{shape[0]} tangentograms read in {tstop_t - tstart_t} seconds.")

if __name__ == '__main__':
    print("start")
    main()
