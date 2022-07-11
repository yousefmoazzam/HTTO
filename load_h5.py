from mpi4py import MPI
import h5py as h5
import argparse
import time

def __option_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("in_file", help="Input data file.")
    #parser.add_argument("out_folder", help="Output folder.")
    parser.add_argument("-p", "--path", help="Data path", default="/entry1/tomo_entry/data/data")
    args = parser.parse_args()
    return args

def main():
    args = __option_parser()
    rank = MPI.COMM_WORLD.rank
    nproc = MPI.COMM_WORLD.size
    tstart = MPI.Wtime()
    dtstart = time.time()

    with h5.File(args.in_file, "r", driver="mpio", comm=MPI.COMM_WORLD) as in_file:
        dataset = in_file[args.path]
        shape = dataset.shape
        i0 = round((shape[-1] / nproc) * rank)
        i1 = round((shape[-1] / nproc) * (rank + 1))
        proc_data = dataset[:, :, i0:i1]
        size = dataset.size

    tstop = MPI.Wtime()
    dtstop = time.time()

    if rank == 0:
        print(f"Number of processes: {nproc}.")
        print(f"Dataset of size {size} read in {tstop-tstart} seconds.")
        print(f"Time = {dtstop - dtstart} seconds.")

if __name__ == '__main__':
    print("start")
    main()
