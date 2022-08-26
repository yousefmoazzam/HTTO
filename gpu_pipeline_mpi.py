import numpy as np
from mpi4py import MPI
import h5py as h5
import argparse
import tomopy
import load_h5
import chunk_h5
from datetime import datetime
import os

import GPUtil
from tomobar.methodsDIR import RecToolsDIR

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
                        help="Percentage of data to process. 10 will take the middle 10% of data in the second dimension.")    
    parser.add_argument("-rec", "--reconstruction", default="tomopy", help="The reconstruction toolbox to use for FBP CUDA: tomopy or tomobar.")
    parser.add_argument("-rings", "--stripe", default=None, help="The stripes removal method to apply.")
    parser.add_argument("-nc", "--ncore", type=int, default=1, help="The number of cores.")
    args = parser.parse_args()
    return args


def main():
    args = __option_parser()
    comm = MPI.COMM_WORLD
    comm_size = comm.Get_size()
    rank = comm.Get_rank()     
    for i in range(args.repeat):
        print(f"Run number: {i}")
        total_time0 = MPI.Wtime()
        with h5.File(args.in_file, "r", driver="mpio", comm=comm) as in_file:
            dataset = in_file[args.path]
            shape = dataset.shape
        print_once(f"Dataset shape is {shape}")
        angles_degrees = load_h5.get_angles(args.in_file, comm=comm)    
        data_indices = load_h5.get_data_indices(args.in_file,
                                                image_key_path="/entry1/tomo_entry/instrument/detector/image_key",
                                                comm=comm)
        angles_radians = np.deg2rad(angles_degrees[data_indices])
        
        # preview to prepeare to crop the data from the middle when --crop is used to avoid loading the whole volume
        preview = [f"{data_indices[0]}: {data_indices[-1] + 1}", ":", ":"]
        if args.crop != 100:
            new_length = int(round(shape[1] * args.crop/100))
            offset = int((shape[1] - new_length) / 2)
            preview[1] = f"{offset}: {offset + new_length}"
            cropped_shape = (data_indices[-1] + 1 - data_indices[0], new_length, shape[2])
        else:
            cropped_shape = (data_indices[-1] + 1 - data_indices[0], shape[1], shape[2])
        preview = ", ".join(preview)

        print_once(f"Cropped data shape is {cropped_shape}")

        load_time0 = MPI.Wtime()
        data = load_h5.load_data(args.in_file, args.dimension, args.path, comm=comm, preview=preview)
        load_time1 = MPI.Wtime()
        load_time = load_time1 - load_time0
        print_once(f"Raw projection data loaded in {load_time} seconds")

        darks, flats = load_h5.get_darks_flats(args.in_file, args.path,
                                            image_key_path="/entry1/tomo_entry/instrument/detector/image_key",
                                            comm=comm, preview=preview, dim=args.dimension)

        (angles_total, detector_y, detector_x) = np.shape(data)
        print(f"Process {rank}'s data shape is {(angles_total, detector_y, detector_x)}")
        
        norm_time0 = MPI.Wtime()
        data = tomopy.normalize(data, flats, darks, ncore=args.ncore, cutoff=10)
        norm_time1 = MPI.Wtime()
        norm_time = norm_time1 - norm_time0
        print_once(f"Data normalised in {norm_time} seconds")
        
        min_log_time0 = MPI.Wtime()
        data[data == 0.0] = 1e-09
        data = tomopy.minus_log(data, ncore=args.ncore)
        #data[data > 0.0] = -np.log(data[data > 0.0])        
        min_log_time1 = MPI.Wtime()
        min_log_time = min_log_time1 - min_log_time0
        print_once(f"Minus log process executed in {min_log_time} seconds")
        
        abs_out_folder = os.path.abspath(args.out_folder)
        out_folder = f"{abs_out_folder}/{datetime.now().strftime('%d-%m-%Y_%H_%M_%S')}_recon"
        if rank == 0:
            print("Making directory")
            os.mkdir(out_folder)
            print("Directory made")

        nNodes = 1 # change this when nNodes > 1
        GPUs_list = GPUtil.getAvailable(order='memory', limit=4) # will return a list of availble GPUs
        GPU_index_wr_to_rank = __calculate_GPU_index(nNodes, rank, GPUs_list)
        #print(len(GPUs_list))
        #print(GPU_index_wr_to_rank)
        
        # calculate the chunk size for the projection data
        slices_no_in_chunks = 4
        if (args.dimension == 1):
            chunks_data = (slices_no_in_chunks, detector_y, detector_x)
        elif (args.dimension == 2):
            chunks_data = (angles_total, slices_no_in_chunks, detector_x)
        else:
            chunks_data = (angles_total, detector_y, slices_no_in_chunks)

        if args.dimension == 1:
            save_time0 = MPI.Wtime()
            chunk_h5.save_dataset(out_folder, "intermediate.h5", data, args.dimension, chunks_data, comm=comm)
            save_time1 = MPI.Wtime()
            save_time = save_time1 - save_time0
            print_once(f"Intermediate data saved in {save_time} seconds")

            slicing_dim = 2 # assuming sinogram slicing here to get it loaded
            reload_time0 = MPI.Wtime()
            data = load_h5.load_data(f"{out_folder}/intermediate.h5", slicing_dim, "/data", comm=comm)
            reload_time1 = MPI.Wtime()
            reload_time = reload_time1 - reload_time0
            print_once(f"Data reloaded in {reload_time} seconds")
        

        # calculating the center of rotation 
        center_time0 = MPI.Wtime()
        rot_center = 0
        mid_rank = int(round(comm_size/2)+0.1)        
        if rank == mid_rank:
            mid_slice = int(np.size(data,1)/2)
            #print(f"Slice for calculation CoR {mid_slice}")
            rot_center = tomopy.find_center_vo(data[:, mid_slice, :], step=0.5, ncore=args.ncore)
            #print(f"The calculated CoR is {rot_center}")
        rot_center = comm.bcast(rot_center, root=mid_rank)
        center_time1 = MPI.Wtime()
        center_time = center_time1 - center_time0
        print_once(f"COR found in {center_time} seconds")
        
        if args.stripe is not None:
            # removing stripes
            stripes_time0 = MPI.Wtime()
            if args.stripe == 'remove_stripe_fw':
                data = tomopy.prep.stripe.remove_stripe_fw(data, wname='db5', sigma=5, ncore=args.ncore)
            if args.stripe == 'remove_all_stripe':
                data = tomopy.prep.stripe.remove_all_stripe(data, ncore=args.ncore)
            if args.stripe == 'remove_stripe_based_sorting':
                data = tomopy.prep.stripe.remove_stripe_based_sorting(data, ncore=args.ncore)
            stripes_time1 = MPI.Wtime()
            stripes_time = stripes_time1 - stripes_time0
            print_once(f"Data unstriped in {stripes_time} seconds")

        recon_time0 = MPI.Wtime()
        print_once(f"Using CoR {rot_center}")    
            
        if args.reconstruction == 'tomopy':
            # use ASTRA toolbox for reconstruction on a GPU
            if GPUs_list is not None:
                opts = {}
                opts['method']='FBP_CUDA'
                opts['proj_type']='cuda'
                opts['gpu_list']=[GPU_index_wr_to_rank]
                recon = tomopy.recon(data,
                        angles_radians,
                        center=rot_center,
                        algorithm=tomopy.astra,
                        options=opts,
                        ncore=args.ncore)
            else:
                raise Exception("There are no GPUs available for reconstruction")
        else:
            # use tomobar software (aslo wraps ASTRA similarly to tomopy)
            RectoolsDIR = RecToolsDIR(
                    DetectorsDimH = detector_x,                 # Horizontal detector dimension
                    DetectorsDimV = detector_y,                 # Vertical detector dimension (3D case)
                    CenterRotOffset = detector_x*0.5 - rot_center,       # Center of Rotation scalar or a vector
                    AnglesVec = angles_radians,         # A vector of projection angles in radians
                    ObjSize = detector_x,               # Reconstructed object dimensions (scalar)
                    device_projector=GPU_index_wr_to_rank)
            recon = RectoolsDIR.FBP(np.swapaxes(data, 0, 1)) # perform FBP as 3D BP with Astra and then filtering
        recon_time1 = MPI.Wtime()
        recon_time = recon_time1 - recon_time0
        print_once(f"Data reconstructed in {recon_time} seconds") 
        
        (vert_slices, recon_x, recon_y) = np.shape(recon)
        chunks_recon = (1, recon_x, recon_y) 
        
        save_recon_time0 = MPI.Wtime()
        chunk_h5.save_dataset(out_folder, "reconstruction.h5", recon, 1, chunks_recon, comm=comm)
        save_recon_time1 = MPI.Wtime()
        save_recon_time = save_recon_time1 - save_recon_time0
        print_once(f"Reconstruction saved in {save_recon_time} seconds")

        total_time1 = MPI.Wtime()
        total_time = total_time1 - total_time0
        print_once(f"Total time = {total_time} seconds.")

def print_once(output):
    if MPI.COMM_WORLD.rank == 0:
        print(output)
        
def __calculate_GPU_index(nNodes, rank, GPUs_list):
    nGPUs = len(GPUs_list)
    return int(rank / nNodes) % nGPUs

def concat_for_gpu(data, dim, nGPUs, comm=MPI.COMM_WORLD):
    """Concatonate data into larger arrays for processes that will be active during gpu methods."""
    root = comm.rank % nGPUs
    if comm.rank == root:
        active = True
        data = [data]
        for i in range(root + nGPUs, comm.size, nGPUs):
            data.append(comm.recv(source=i, tag=0))
    else:
        active = False
        comm.send(data, root, tag=0)
    if active:
        axis = dim - 1
        data = np.concatenate(data, axis=axis)
    else:
        data = None
    return data


def scatter_after_gpu(data, dim, nGPUs, comm=MPI.COMM_WORLD):
    """After a GPU plugin where data has been concatonated, split data back up between all processes."""
    root = comm.rank % nGPUs
    if comm.rank == root:
        group_size = 0
        for i in range(comm.rank, comm.size, nGPUs):
            group_size += 1
        axis = dim - 1
        data = np.array_split(data, group_size, axis=axis)
        for i, process_rank in enumerate(range(comm.rank + nGPUs, comm.size, nGPUs)):
            comm.send(data[i], process_rank, tag=0)
        data = data[0]
    else:
        data = comm.recv(source=root, tag=0)
    return data

if __name__ == '__main__':
    main()
