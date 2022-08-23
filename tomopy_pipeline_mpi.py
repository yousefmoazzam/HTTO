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
    parser.add_argument("-m", "--methods_no", type=int, default=1, help="The number of different methods to apply to data.")
    parser.add_argument("-nc", "--ncore", type=int, default=1, help="The number of cores.")
    args = parser.parse_args()
    return args


def main():
    args = __option_parser()
    for i in range(args.repeat):
        print(f"Run number: {i}")
        total_time0 = MPI.Wtime()
        with h5.File(args.in_file, "r", driver="mpio", comm=MPI.COMM_WORLD) as in_file:
            dataset = in_file[args.path]
            shape = dataset.shape
        angles_degrees = load_h5.get_angles(args.in_file, comm=MPI.COMM_WORLD)    
        data_indices = load_h5.get_data_indices(args.in_file,
                                                image_key_path="/entry1/tomo_entry/instrument/detector/image_key",
                                                comm=MPI.COMM_WORLD)
        angles_radians = np.deg2rad(angles_degrees[data_indices])
        
        # preview to prepeare to crop the data from the middle when --crop is used to avoid loading the whole volume
        preview = [f"{data_indices[0]}: {data_indices[-1] + 1}", ":", ":"]
        if args.crop != 100:
            new_length = int(round(shape[1] * args.crop/100))
            offset = int((shape[1] - new_length) / 2)
            preview[1] = f"{offset}: {offset + new_length}"
        preview = ", ".join(preview)

        load_time0 = MPI.Wtime()
        data = load_h5.load_data(args.in_file, 2, args.path, comm=MPI.COMM_WORLD, preview=preview)
        load_time1 = MPI.Wtime()
        load_time = load_time1 - load_time0
        print(f"Raw projection data loaded in {load_time} seconds")

        darks, flats = load_h5.get_darks_flats(args.in_file, args.path,
                                            image_key_path="/entry1/tomo_entry/instrument/detector/image_key",
                                            comm=MPI.COMM_WORLD, preview=preview, dim=2)

        (angles_total, detector_y, detector_x) = np.shape(data)
        print(f"Data shape is {(angles_total, detector_y, detector_x)}")
        
        norm_time0 = MPI.Wtime()
        data = tomopy.normalize(data, flats, darks, ncore=args.ncore, cutoff=10)
        norm_time1 = MPI.Wtime()
        norm_time = norm_time1 - norm_time0
        print(f"Data normalised in {norm_time} seconds")
        
        min_log_time0 = MPI.Wtime()
        data[data == 0.0] = 1e-09
        data = tomopy.minus_log(data, ncore=args.ncore)
        #data[data > 0.0] = -np.log(data[data > 0.0])        
        min_log_time1 = MPI.Wtime()
        min_log_time = min_log_time1 - min_log_time0
        print(f"Minus log process executed in {min_log_time} seconds")
        
        abs_out_folder = os.path.abspath(args.out_folder)
        out_folder = f"{abs_out_folder}/{datetime.now().strftime('%d-%m-%Y_%H_%M_%S')}_recon"
        if MPI.COMM_WORLD.rank == 0:
            print("Making directory")
            os.mkdir(out_folder)
            print("Directory made")

        # calculate the chunk size for the projection data
        slices_no_in_chunks = 1
        if (args.dimension == 1):
            chunks_data = (slices_no_in_chunks, detector_y, detector_x)
        elif (args.dimension == 2):
            chunks_data = (angles_total, slices_no_in_chunks, detector_x)
        else:
            chunks_data = (angles_total, detector_y, slices_no_in_chunks)

        #save_time0 = MPI.Wtime()
        #chunk_h5.save_dataset(out_folder, "intermediate.h5", data, args.dimension, chunks_data, comm=MPI.COMM_WORLD)
        #save_time1 = MPI.Wtime()
        #save_time = save_time1 - save_time0
        #print(f"Intermediate data saved in {save_time} seconds")
        
        #slicing_dim = 2 # assuming sinogram slicing here to get it loaded
        #reload_time0 = MPI.Wtime()
        #data = load_h5.load_data(f"{out_folder}/intermediate.h5", slicing_dim, "/data", comm=MPI.COMM_WORLD)
        #reload_time1 = MPI.Wtime()
        #reload_time = reload_time1 - reload_time0
        #print(f"Data reloaded in {reload_time} seconds")
        
        # calculating the center of rotation 
        center_time0 = MPI.Wtime()
        rot_center = 0
        data = np.swapaxes(data, 0, 1)
        mid_rank = int(round(MPI.COMM_WORLD.size/2)+0.1)
        print(data.shape)
        if MPI.COMM_WORLD.rank == mid_rank:
            print(np.size(data, 0))
            mid_slice = int(np.size(data,0)/2)
            #print(f"Slice for calculation CoR {mid_slice}")
            rot_center = tomopy.find_center_vo(data[mid_slice, :, :], step=0.5, ncore=args.ncore)
            #print(f"The calculated CoR is {rot_center}")
        rot_center = MPI.COMM_WORLD.bcast(rot_center, root=mid_rank)
        center_time1 = MPI.Wtime()
        center_time = center_time1 - center_time0
        print(f"COR found in {center_time} seconds")               

        if (args.methods_no == 2):
            # removing stripes
            stripes_time0 = MPI.Wtime()
            data = tomopy.prep.stripe.remove_stripe_based_sorting(data)    
            stripes_time1 = MPI.Wtime()
            stripes_time = stripes_time1 - stripes_time0
            print(f"Data unstriped in {stripes_time} seconds")
        if (args.methods_no == 3):
            # removing stripes
            stripes_time0 = MPI.Wtime()
            data = tomopy.prep.stripe.remove_stripe_based_sorting(data)    
            stripes_time1 = MPI.Wtime()
            stripes_time = stripes_time1 - stripes_time0
            print(f"Data unstriped in {stripes_time} seconds")               
            # applying median filter
            filter_time0 = MPI.Wtime()
            data = tomopy.misc.corr.median_filter(data)  
            filter_time1 = MPI.Wtime()
            filter_time = filter_time1 - filter_time0
            print(f"Data median filtered in {filter_time} seconds")      
        if (args.methods_no == 4):
            # removing stripes
            stripes_time0 = MPI.Wtime()
            data = tomopy.prep.stripe.remove_stripe_based_sorting(data)    
            stripes_time1 = MPI.Wtime()
            stripes_time = stripes_time1 - stripes_time0
            print(f"Data unstriped in {stripes_time} seconds")               
            # applying median filter
            filter_time0 = MPI.Wtime()
            data = tomopy.misc.corr.median_filter(data)  
            filter_time1 = MPI.Wtime()
            filter_time = filter_time1 - filter_time0
            print(f"Data median filtered in {filter_time} seconds")            
            # applying Paganin filter
            filter_time0 = MPI.Wtime()
            data = tomopy.prep.phase.retrieve_phase(data)  
            filter_time1 = MPI.Wtime()
            filter_time = filter_time1 - filter_time0
            print(f"Applying Paganin filter in {filter_time} seconds")              
            
        recon_time0 = MPI.Wtime()
        print(f"The actual used CoR is {rot_center}")
        recon = tomopy.recon(data, angles_radians, center=rot_center, algorithm='gridrec', sinogram_order=True, ncore=args.ncore)
        recon_time1 = MPI.Wtime()
        recon_time = recon_time1 - recon_time0
        print(f"Data reconstructed in {recon_time} seconds")
        
        (vert_slices, recon_x, recon_y) = np.shape(recon)
        chunks_recon = (slices_no_in_chunks, recon_x, recon_y)    
        
        save_recon_time0 = MPI.Wtime()
        chunk_h5.save_dataset(out_folder, "reconstruction.h5", recon, 1, chunks_recon, comm=MPI.COMM_WORLD)
        save_recon_time1 = MPI.Wtime()
        save_recon_time = save_recon_time1 - save_recon_time0
        print(f"Reconstruction saved in {save_recon_time} seconds")

        total_time1 = MPI.Wtime()
        total_time = total_time1 - total_time0
        print(f"Total time = {total_time} seconds.")       

if __name__ == '__main__':
    main()
