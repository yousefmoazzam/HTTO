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


def main():
    args = __option_parser()
    chunks = (150, 150, 10)
    save_file_chunked(args.in_file, args.out_folder, chunks, path=args.path)


if __name__ == '__main__':
    print("start")
    main()
