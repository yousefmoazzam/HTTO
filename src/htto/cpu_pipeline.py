import os
import sys
from datetime import datetime
from pathlib import Path

import h5py as h5
import numpy as np
import tomopy
from larix.methods.misc import MEDIAN_FILT
from mpi4py import MPI
from nvtx import annotate

from htto.common import PipelineStages
from htto.utils import print_once, print_rank

from .h5_utils import chunk_h5, load_h5


def cpu_pipeline(
    in_file: Path,
    out_dir: Path,
    data_key: str,
    dimension: int,
    crop: int = 100,
    pad: int = 0,
    stop_after: PipelineStages = PipelineStages.RECONSTRUCT,
):
    """Run the CPU pipline to reconstruct the data.

    Args:
        in_file: The file to read data from.
        out_dir: The directory to write data to.
        data_key: The input file dataset key to read.
        dimension: The dimension to slice in.
        crop: The percentage of data to use. Defaults to 100.
        pad: The padding size to use. Defaults to 0.
        stop_after: The stage after which the pipeline should stop. Defaults to
            PipelineStages.RECONSTRUCT.
    """
    comm = MPI.COMM_WORLD

    total_time0 = MPI.Wtime()
    with h5.File(in_file, "r", driver="mpio", comm=comm) as file:
        dataset = file[data_key]
        shape = dataset.shape
    print_once(f"Dataset shape is {shape}", comm)
    ###################################################################################
    #                                 Loading the data
    with annotate(PipelineStages.LOAD.name):
        angles_degrees = load_h5.get_angles(in_file, comm=comm)
        data_indices = load_h5.get_data_indices(
            in_file,
            image_key_path="/entry1/tomo_entry/instrument/detector/image_key",
            comm=comm,
        )
        angles_radians = np.deg2rad(angles_degrees[data_indices])

        # preview to prepare to crop the data from the middle when --crop is used to
        # avoid loading the whole volume and crop out darks and flats when loading data.
        preview = [f"{data_indices[0]}: {data_indices[-1] + 1}", ":", ":"]
        if crop != 100:
            new_length = int(round(shape[1] * crop / 100))
            offset = int((shape[1] - new_length) / 2)
            preview[1] = f"{offset}: {offset + new_length}"
            cropped_shape = (
                data_indices[-1] + 1 - data_indices[0],
                new_length,
                shape[2],
            )
        else:
            cropped_shape = (data_indices[-1] + 1 - data_indices[0], shape[1], shape[2])
        preview = ", ".join(preview)

        print_once(f"Cropped data shape is {cropped_shape}", comm)

        load_time0 = MPI.Wtime()
        dim = dimension
        pad_values = load_h5.get_pad_values(
            pad,
            dim,
            shape[dim - 1],
            data_indices=data_indices,
            preview=preview,
            comm=comm,
        )
        print_rank(f"Pad values are {pad_values}.", comm)
        data = load_h5.load_data(
            in_file, dim, data_key, preview=preview, pad=pad_values, comm=comm
        )
        load_time1 = MPI.Wtime()
        load_time = load_time1 - load_time0
        print_once(f"Raw projection data loaded in {load_time} seconds", comm)

        darks, flats = load_h5.get_darks_flats(
            in_file,
            data_key,
            image_key_path="/entry1/tomo_entry/instrument/detector/image_key",
            comm=comm,
            preview=preview,
            dim=dimension,
        )

        (angles_total, detector_y, detector_x) = np.shape(data)
        print_rank(
            f"Data shape is {(angles_total, detector_y, detector_x)}"
            + f" of type {data.dtype}",
            comm,
        )
    if stop_after == PipelineStages.LOAD:
        # you might want to write the resulting volume here for testing/comparison with
        # CPU?
        sys.exit()
    ###################################################################################
    #                3D Median filter to apply to raw data/flats/darks
    with annotate(PipelineStages.FILTER.name):
        median_time0 = MPI.Wtime()
        kernel_size = 3  # full size kernel 3 x 3 x 3
        data = MEDIAN_FILT(data, kernel_size, comm.size)
        flats = MEDIAN_FILT(np.asarray(flats), kernel_size, comm.size)
        darks = MEDIAN_FILT(np.asarray(darks), kernel_size, comm.size)
        median_time1 = MPI.Wtime()
        median_time = median_time1 - median_time0
        print_once(f"Median filtering took {median_time} seconds", comm)
    if stop_after == PipelineStages.FILTER:
        # you might want to write the resulting volume here for testing/comparison with
        # GPU?
        sys.exit()
    ###################################################################################
    #                 Normalising the data and taking the negative log
    with annotate(PipelineStages.NORMALIZE.name):
        norm_time0 = MPI.Wtime()
        data = tomopy.normalize(data, flats, darks, ncore=comm.size, cutoff=10)
        data[data == 0.0] = 1e-09
        data = tomopy.minus_log(data, ncore=comm.size)
        # data[data > 0.0] = -np.log(data[data > 0.0])
        norm_time1 = MPI.Wtime()
        norm_time = norm_time1 - norm_time0
        print_once(
            f"Normalising the data and negative log took {norm_time} seconds", comm
        )
    if stop_after == PipelineStages.NORMALIZE:
        # you might want to write the resulting volume here for testing/comparison with
        # GPU?
        sys.exit()
    ###################################################################################
    #                                 Removing stripes
    with annotate(PipelineStages.STRIPES.name):
        stripes_time0 = MPI.Wtime()
        data = tomopy.prep.stripe.remove_stripe_ti(
            data, nblock=0, alpha=1.5, ncore=comm.size
        )
        stripes_time1 = MPI.Wtime()
        stripes_time = stripes_time1 - stripes_time0
        print_once(f"Data unstriped in {stripes_time} seconds", comm)
    if stop_after == PipelineStages.STRIPES:
        # you might want to write the resulting volume here for testing/comparison with
        # GPU?
        sys.exit()
    ###################################################################################
    #                        Calculating the center of rotation
    with annotate(PipelineStages.CENTER.name):
        center_time0 = MPI.Wtime()
        rot_center = 0
        mid_rank = int(round(comm.size / 2) + 0.1)
        if comm.rank == mid_rank:
            mid_slice = int(np.size(data, 1) / 2)
            rot_center = tomopy.find_center_vo(
                data[:, mid_slice, :], step=0.5, ncore=comm.size
            )
        rot_center = comm.bcast(rot_center, root=mid_rank)
        center_time1 = MPI.Wtime()
        center_time = center_time1 - center_time0
        print_once(f"COR {rot_center} found in {center_time} seconds", comm)
    if stop_after == PipelineStages.CENTER:
        # you might want to write the resulting volume here for testing/comparison with
        # GPU?
        sys.exit()
    ###################################################################################
    #                    Saving/reloading the intermediate dataset
    with annotate(PipelineStages.RESLICE.name):
        run_out_dir = f"{out_dir}/{datetime.now().strftime('%d-%m-%Y_%H_%M_%S')}_recon"
        if comm.rank == 0:
            print("Making directory")
            os.mkdir(run_out_dir)
            print("Directory made")

        # calculate the chunk size for the projection data
        slices_no_in_chunks = 4
        if dimension == 1:
            chunks_data = (slices_no_in_chunks, detector_y, detector_x)
        elif dimension == 2:
            chunks_data = (angles_total, slices_no_in_chunks, detector_x)
        else:
            chunks_data = (angles_total, detector_y, slices_no_in_chunks)

        if dimension == 1:
            save_time0 = MPI.Wtime()
            chunk_h5.save_dataset(
                run_out_dir,
                "intermediate.h5",
                data,
                dimension,
                chunks_data,
                comm=comm,
            )
            save_time1 = MPI.Wtime()
            save_time = save_time1 - save_time0
            print_once(f"Intermediate data saved in {save_time} seconds", comm)

            slicing_dim = 2  # assuming sinogram slicing here to get it loaded
            reload_time0 = MPI.Wtime()
            data = load_h5.load_data(
                f"{run_out_dir}/intermediate.h5", slicing_dim, "/data", comm=comm
            )
            dim = slicing_dim
            reload_time1 = MPI.Wtime()
            reload_time = reload_time1 - reload_time0
            print_once(f"Data reloaded in {reload_time} seconds", comm)
    if stop_after == PipelineStages.RESLICE:
        # you might want to write the resulting volume here for testing/comparison with
        # GPU?
        sys.exit()
    ###################################################################################
    #                           Reconstruction with gridrec
    with annotate(PipelineStages.RECONSTRUCT.name):
        recon_time0 = MPI.Wtime()
        print_once(f"Using CoR {rot_center}", comm)
        recon = tomopy.recon(
            np.swapaxes(data, 0, 1),
            angles_radians,
            center=rot_center,
            algorithm="gridrec",
            sinogram_order=True,
            ncore=comm.size,
        )
        recon_time1 = MPI.Wtime()
        recon_time = recon_time1 - recon_time0
        print_once(f"Data reconstructed in {recon_time} seconds", comm)
    if stop_after == PipelineStages.RECONSTRUCT:
        # you might want to write the resulting volume here for testing/comparison with
        # GPU?
        sys.exit()
    ###################################################################################
    #                     Saving the result of the reconstruction
    with annotate(PipelineStages.SAVE.name):
        (vert_slices, recon_x, recon_y) = np.shape(recon)
        chunks_recon = (1, recon_x, recon_y)

        save_recon_time0 = MPI.Wtime()
        chunk_h5.save_dataset(
            run_out_dir, "reconstruction.h5", recon, dim, chunks_recon, comm=comm
        )
        save_recon_time1 = MPI.Wtime()
        save_recon_time = save_recon_time1 - save_recon_time0
        print_once(f"Reconstruction saved in {save_recon_time} seconds", comm)
    if stop_after == PipelineStages.SAVE:
        # you might want to write the resulting volume here for testing/comparison with
        # GPU?
        sys.exit()
    ####################################################################################
    total_time1 = MPI.Wtime()
    total_time = total_time1 - total_time0
    print_once(f"Total time = {total_time} seconds.", comm)
