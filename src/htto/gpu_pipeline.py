import math
import os
import sys
from datetime import datetime
from enum import IntEnum, unique
from pathlib import Path

import cupy as cp
import h5py as h5
import numpy as np
import tomopy
from larix.methods.misc_gpu import MEDIAN_FILT_GPU
from mpi4py import MPI
from nvtx import annotate
from tomobar.methodsDIR import RecToolsDIR

from htto.common import PipelineStages
from htto.utils import print_once, print_rank

from .h5_utils import chunk_h5, load_h5


@unique
class Reconstors(IntEnum):
    """An enumeration of available reconstruction methods."""

    TOMOPY = 0
    TOMOBAR = 1


def gpu_pipeline(
    in_file: Path,
    out_dir: Path,
    data_key: str,
    dimension: int,
    crop: int = 100,
    pad: int = 0,
    stop_after: PipelineStages = PipelineStages.RECONSTRUCT,
    reconstruction: Reconstors = Reconstors.TOMOPY,
):
    """Run the GPU pipline to reconstruct the data.

    Args:
        in_file: The file to read data from.
        out_dir: The directory to write data to.
        data_key: The input file dataset key to read.
        dimension: The dimension to slice in.
        crop: The percentage of data to use. Defaults to 100.
        pad: The padding size to use. Defaults to 0.
        stop_after: The stage after which the pipeline should stop. Defaults to
            PipelineStages.RECONSTRUCT.
        reconstruction: The reconstruction method (library) to be used. Defaults to
            Reconstors.TOMOPY.
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
        data = MEDIAN_FILT_GPU(data, kernel_size)
        flats = MEDIAN_FILT_GPU(np.asarray(flats), kernel_size)
        darks = MEDIAN_FILT_GPU(np.asarray(darks), kernel_size)
        median_time1 = MPI.Wtime()
        median_time = median_time1 - median_time0
        print_once(f"Median filtering took {median_time} seconds", comm)
    if stop_after == PipelineStages.FILTER:
        # you might want to write the resulting volume here for testing/comparison with
        # CPU?
        sys.exit()
    ###################################################################################
    #                 Normalising the data and taking the negative log
    with annotate(PipelineStages.NORMALIZE.name):
        norm_time0 = MPI.Wtime()
        data_gpu = cp.asarray(data)
        dark0 = cp.mean(cp.asarray(darks), axis=0)
        flat0 = cp.mean(cp.asarray(flats), axis=0)
        data_gpu = (data_gpu - dark0) / (flat0 - dark0 + 1e-3)
        data_gpu[data_gpu <= 0] = 1
        data_gpu = -cp.log(data_gpu)
        data_gpu[cp.isnan(data_gpu)] = 6.0
        data_gpu[cp.isinf(data_gpu)] = 0
        norm_time1 = MPI.Wtime()
        norm_time = norm_time1 - norm_time0
        print_once(
            f"Normalising the data and negative log took {norm_time} seconds", comm
        )
    if stop_after == PipelineStages.NORMALIZE:
        # you might want to write the resulting volume here for testing/comparison with
        # CPU?
        sys.exit()
    ###################################################################################
    #                                 Removing stripes
    with annotate(PipelineStages.STRIPES.name):
        stripes_time0 = MPI.Wtime()
        #  Remove stripes with a new method by V. Titarenko  (TomoCuPy)
        beta = 0.1  # lowering the value increases the filter strength
        gamma = beta * ((1 - beta) / (1 + beta)) ** cp.abs(
            cp.fft.fftfreq(data_gpu.shape[-1]) * data_gpu.shape[-1]
        )
        gamma[0] -= 1
        v = cp.mean(data_gpu, axis=0)
        v = v - v[:, 0:1]
        v = cp.fft.irfft(cp.fft.rfft(v) * cp.fft.rfft(gamma))
        data_gpu[:] += v
        data = data_gpu.get()
        stripes_time1 = MPI.Wtime()
        stripes_time = stripes_time1 - stripes_time0
        print_once(f"Data unstriped in {stripes_time} seconds", comm)
    if stop_after == PipelineStages.STRIPES:
        # you might want to write the resulting volume here for testing/comparison with
        # CPU?
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
    #        Reconstruction with either Tomopy-ASTRA (2D) or ToMoBAR-ASTRA (3D)
    with annotate(PipelineStages.RECONSTRUCT.name):
        nNodes = 1  # change this when nNodes > 1
        GPU_devicesNo = cp.cuda.runtime.getDeviceCount()
        GPU_index_wr_to_rank = __calculate_GPU_index(nNodes, comm.rank, GPU_devicesNo)

        recon_time0 = MPI.Wtime()
        print_once(f"Using CoR {rot_center}", comm)
        print_once(f"Number of GPUs = {GPU_devicesNo}", comm)

        if GPU_devicesNo is not None or GPU_devicesNo > 0:
            data = concat_for_gpu(data, 2, GPU_devicesNo, comm)
            if data is not None:
                if reconstruction == Reconstors.TOMOPY:
                    # use Tomopy-ASTRA toolbox for reconstruction on a GPU
                    print_rank("GPU reconstruction.", comm)
                    opts = {}
                    opts["method"] = "FBP_CUDA"
                    opts["proj_type"] = "cuda"
                    opts["gpu_list"] = [GPU_index_wr_to_rank]
                    recon = tomopy.recon(
                        data,
                        angles_radians,
                        center=rot_center,
                        algorithm=tomopy.astra,
                        options=opts,
                        ncore=comm.size,
                    )
                elif reconstruction == Reconstors.TOMOBAR:
                    # using ToMoBAR software (wraps ASTRA but uses 3D BP and CuPy for
                    # filtering)
                    RectoolsDIR = RecToolsDIR(
                        DetectorsDimH=detector_x,  # Horizontal detector dimension
                        DetectorsDimV=np.size(
                            data, 1
                        ),  # Vertical detector dimension (3D case)
                        CenterRotOffset=detector_x * 0.5
                        - rot_center,  # Center of Rotation scalar or a vector
                        AnglesVec=angles_radians,  # A vector of projection angles
                        ObjSize=detector_x,  # Reconstructed object dimensions (scalar)
                        device_projector=GPU_index_wr_to_rank,
                    )
                    recon = RectoolsDIR.FBP3D_cupy(
                        np.swapaxes(data, 0, 1)
                    )  # perform FBP as 3D BP with Astra and then filtering
            else:
                print_rank("Waiting for GPU processes.", comm)
                recon = data
            recon = scatter_after_gpu(recon, 2, GPU_devicesNo, comm)
        else:
            raise Exception("There are no GPUs available for reconstruction")
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
    ###################################################################################

    total_time1 = MPI.Wtime()
    total_time = total_time1 - total_time0
    print_once(f"Total time = {total_time} seconds.", comm)


def __calculate_GPU_index(nNodes, rank, GPU_devicesNo):
    return int(rank / nNodes) % GPU_devicesNo


def concat_for_gpu(data, dim, nGPUs, comm=MPI.COMM_WORLD):
    """Give all the data to GPU processes by concatenating it into larger arrays.

    Args:
        data: Data to be concatenated.
        dim: Dimension along which to concatenate (1, 2, 3).
        nGPUs: Number of GPUs, which corresponds to number of processes the data will
            be split between.
        comm: MPI communicator object.
    """
    my_gpu_proc = (
        comm.rank % nGPUs
    )  # GPU process that will receive data from this process.
    if comm.rank == my_gpu_proc:  # This is a GPU process
        data = [data]
        # Creating a list of data blocks.
        for source in range(my_gpu_proc + nGPUs, comm.size, nGPUs):
            data.append(__recv_big(source, comm))
        axis = dim - 1  # Dim 1 = axis 0
        data = np.concatenate(data, axis=axis)
    else:  # This is not a GPU process
        __send_big(data, my_gpu_proc, comm)
        data = None
    return data


def scatter_after_gpu(data, dim, nGPUs, comm=MPI.COMM_WORLD):
    """Split data back up between all processes after data has been concatenated.

    Args:
        data: Data to be scattered.
        dim: Dimension along which data has been concatenated, so data can be split
            along the same dimension.
        nGPUs: Number of GPUs. This is the number of processes who currently have all
            the data.
        comm: MPI communicator object.
    """
    my_gpu_proc = comm.rank % nGPUs  # GPU process that will send this process data.
    if comm.rank == my_gpu_proc:  # This is a GPU process
        group_size = 0
        for i in range(comm.rank, comm.size, nGPUs):
            group_size += 1  # Number of processes this gpu process has to send data to.
        axis = dim - 1
        # Splitting data into blocks
        data = np.array_split(data, group_size, axis=axis)
        for i, dest in enumerate(range(comm.rank + nGPUs, comm.size, nGPUs)):
            __send_big(data[i + 1], dest, comm)  # Keeping
        data = data[0]
    else:  # This is not a GPU process
        data = __recv_big(my_gpu_proc, comm)
    return data


def __send_big(data, dest, comm=MPI.COMM_WORLD):
    """Send data between MPI processes that exceeds the 2GB MPI buffer limit.

    Args:
        data: Data to be sent.
        dest: Rank of the process receiving the data.
        comm: MPI communicator object
    """
    n_bytes = data.size * data.itemsize
    # Every block must be < 2GB (MPI buffer limit)
    n_blocks = math.ceil(n_bytes / 2000000000)
    print_rank(f"Sending {n_bytes} bytes in {n_blocks} blocks to rank {dest}.", comm)
    # Sending number of blocks so destination process knows how many to expect.
    comm.send(n_blocks, dest, tag=123456)
    data_blocks = np.array_split(data, n_blocks, axis=0)
    for i, data_block in enumerate(data_blocks):
        comm.send(
            data_block, dest, tag=i
        )  # Tag to ensure messages arrive in correct order - may not be needed


def __recv_big(source, comm=MPI.COMM_WORLD):
    """Receive data that has been sent using the __send_big() function.

    Args:
        source: Rank of the process sending the data.
        comm: MPI communicator object.
    """
    # Sender tells receiver number of blocks to expect.
    n_blocks = comm.recv(source=source, tag=123456)
    print_rank(f"Recieving {n_blocks} blocks from rank {source}.", comm)
    data_blocks = [None] * n_blocks
    for i in range(n_blocks):
        data_blocks[i] = comm.recv(
            source=source, tag=i
        )  # Tag to ensure messages arrive in correct order
    data = np.concatenate(data_blocks, axis=0)
    return data
