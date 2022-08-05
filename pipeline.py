import numpy as np
from mpi4py import MPI
import h5py as h5
import argparse
import tomopy
import load_h5
import chunk_h5
from datetime import datetime
import os


def __option_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("in_file", help="Input data file.")
    parser.add_argument("out_folder", help="Output folder.")
    parser.add_argument("-p", "--path", default="/entry1/tomo_entry/data/data", help="Data path.")
    parser.add_argument("-c", "--csv", default=None, help="Write results to specified csv file.")
    parser.add_argument("-r", "--repeat", type=int, default=1, help="Number of repeats.")
    parser.add_argument("-d", "--dimension", type=int, choices=[1, 2, 3], default=1,
                        help="Which dimension to slice through (usually 1 = projections, 2 = sinograms).")
    parser.add_argument("-cr", "--crop", type=int, choices=range(1, 101), default=100,
                        help="Percentage of data to process. 10 will take the middle 10% of data in the second.")
    args = parser.parse_args()
    return args


def main():
    args = __option_parser()
    total_time0 = MPI.Wtime()
    with h5.File(args.in_file, "r", driver="mpio", comm=MPI.COMM_WORLD) as in_file:
        dataset = in_file[args.path]
        shape = dataset.shape
    angles = load_h5.get_angles(args.in_file, comm=MPI.COMM_WORLD)
    #angles = np.deg2rad(angles)
    data_indices = load_h5.get_data_indices(args.in_file,
                                            image_key_path="/entry1/tomo_entry/instrument/detector/image_key",
                                            comm=MPI.COMM_WORLD)
    preview = [f"{data_indices[0]}: {data_indices[-1] + 1}", ":", ":"]
    if args.crop != 100:
        new_length = int(round(shape[1] * args.crop/100))
        offset = int((shape[1] - new_length) / 2)
        preview[1] = f"{offset}: {offset + new_length}"
    preview = ", ".join(preview)

    load_time0 = MPI.Wtime()
    data = load_h5.load_data(args.in_file, args.dimension, args.path, comm=MPI.COMM_WORLD, preview=preview)
    load_time1 = MPI.Wtime()
    load_time = load_time1 - load_time0
    print(f"Data loaded in {load_time} seconds")

    darks, flats = load_h5.get_darks_flats(args.in_file, args.path,
                                           image_key_path="/entry1/tomo_entry/instrument/detector/image_key",
                                           comm=MPI.COMM_WORLD, preview=preview)

    norm_time0 = MPI.Wtime()
    data = tomopy.normalize(data, flats, darks)
    norm_time1 = MPI.Wtime()
    norm_time = norm_time1 - norm_time0
    print(f"Data normalised in {norm_time} seconds")

    min_log_time0 = MPI.Wtime()
    data = tomopy.minus_log(data)
    min_log_time1 = MPI.Wtime()
    min_log_time = min_log_time1 - min_log_time0
    print(f"Minus log process executed in {min_log_time} seconds")

    abs_out_folder = os.path.abspath(args.out_folder)
    out_folder = f"{abs_out_folder}/{datetime.now().strftime('%d-%m-%Y_%H:%M:%S')}_recon"
    if MPI.COMM_WORLD.rank == 0:
        print("Making directory")
        os.mkdir(out_folder)
        print("Directory made")

    save_time0 = MPI.Wtime()
    chunk_h5.save_dataset(out_folder, "intermediate.h5", data, args.dimension, comm=MPI.COMM_WORLD)
    save_time1 = MPI.Wtime()
    save_time = save_time1 - save_time0
    print(f"Intermediate data saved in {save_time} seconds")

    reload_time0 = MPI.Wtime()
    data = load_h5.load_data(f"{out_folder}/intermediate.h5", 2, "/data", comm=MPI.COMM_WORLD)
    reload_time1 = MPI.Wtime()
    reload_time = reload_time1 - reload_time0
    print(f"Data reloaded in {reload_time} seconds")

    center_time0 = MPI.Wtime()
    rot_center = tomopy.find_center(data, angles)
    center_time1 = MPI.Wtime()
    center_time = center_time1 - center_time0
    print(f"COR found in {center_time} seconds")

    recon_time0 = MPI.Wtime()
    recon = tomopy.recon(data, angles, center=rot_center, algorithm='gridrec', sinogram_order=False)
    recon_time1 = MPI.Wtime()
    recon_time = recon_time1 - recon_time0
    print(f"Data reconstructed in {recon_time} seconds")

    save_recon_time0 = MPI.Wtime()
    chunk_h5.save_dataset(out_folder, "reconstruction.h5", recon, 1, comm=MPI.COMM_WORLD)
    save_recon_time1 = MPI.Wtime()
    save_recon_time = save_recon_time1 - save_recon_time0
    print(f"Reconstruction saved in {save_recon_time} seconds")

    total_time1 = MPI.Wtime()
    total_time = total_time1 - total_time0
    print(f"Total time = {total_time} seconds.")


if __name__ == '__main__':
    main()
