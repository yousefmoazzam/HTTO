from mpi4py import MPI
import h5py as h5
import argparse
import tomopy
import load_h5

def __option_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("in_file", help="Input data file.")
    parser.add_argument("-p", "--path", default="/entry1/tomo_entry/data/data", help="Data path.")
    parser.add_argument("-c", "--csv", default=None, help="Write results to specified csv file.")
    parser.add_argument("-r", "--repeat", type=int, default=1, help="Number of repeats.")
    parser.add_argument("-d", "--dimension", type=int, choices=[1, 2, 3], default=1,
                        help="Which dimension to slice through (usually 1 = projections, 2 = sinograms).")
    parser.add_argument("-cr", "--crop", type=int, choices=range(1, 101), default=100,
                        help="Percentage of data to process. 10 will take the middle 10% of data in the slice dimension.")
    args = parser.parse_args()
    return args


def main():
    args = __option_parser()
    with h5.File(args.in_file, "r", driver="mpio", comm=MPI.COMM_WORLD) as in_file:
        dataset = in_file[args.path]
        shape = dataset.shape
    angles = load_h5.get_angles(args.in_file, comm=MPI.COMM_WORLD)
    data_indices = load_h5.get_data_indices(args.in_file, comm=MPI.COMM_WORLD)
    preview = [f"{data_indices[0]}: {data_indices[-1] + 1}", ":", ":"]
    if args.crop != 100:
        new_length = int(round(shape[0] * args.crop/100))
        offset = int((shape[0] - new_length) / 2)
        preview[args.dimension - 1] = f"{offset}: {offset + new_length}"
    preview = ", ".join(preview)
    data = load_h5.load_data(args.in_file, args.path, args.dimension, comm=MPI.COMM_WORLD, preview=preview)
    darks, flats = load_h5.get_darks_flats(args.in_file, comm=MPI.COMM_WORLD, preview=preview)
    data = tomopy.normalize(data, flats, darks)
    data = tomopy.minus_log(data)
    data


if __name__ == '__main__':
    main()
