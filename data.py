from mpi4py import MPI
import h5py as h5
import argparse
import math
import pandas as pd
import numpy as np
import os
from datetime import datetime
import load_h5


class Data:

    def __init__(self,
                 file,
                 path="/entry1/tomo_entry/data/data",
                 image_key_path="/entry1/tomo_entry/instrument/detector/image_key",
                 comm=MPI.COMM_WORLD):

        self.file = file
        self.data_path = path
        self.image_key_path = image_key_path
        with h5.File(file, "r", driver="mpio", comm=MPI.COMM_WORLD) as in_file:
            self.dataset = in_file[path]
            self.dataset_shape = self.dataset.shape
        self.data = None
        self.darks = None
        self.flats = None
        self._comm = comm
        self.data_indices = load_h5.get_data_indices(file, image_key_path=image_key_path, comm=comm)
        self.angles_degrees = load_h5.get_angles(file, comm)
        self.angles_radians = np.deg2rad(self.angles_degrees[self.data_indices])

    def load_data(self, dim, preview=None, crop=100, pad=0):
        if preview is None:
            preview = [f"{self.data_indices[0]}: {self.data_indices[-1] + 1}", ":", ":"]
        else:
            preview = preview.split(",")
        if crop != 100:
            new_length = int(round(self.dataset_shape[1] * crop / 100))
            offset = int((self.dataset_shape[1] - new_length) / 2)
            preview[1] = f"{offset}: {offset + new_length}"
            cropped_shape = (self.data_indices[-1] + 1 - self.data_indices[0], new_length, self.dataset_shape[2])
        preview = ", ".join(preview)
        self.data = load_h5.load_data(self.file, dim, self.data_path, comm=self._comm, preview=preview)
        self.darks, self.flats = load_h5.get_darks_flats(self.file, data_path=self.data_path,
                                                         image_key_path=self.image_key_path, comm=self._comm,
                                                         preview=preview, dim=dim)
