from osgeo import gdal
import numpy as np
import matplotlib.pyplot as plt


class MultispectralDataset:
    # ToDo: Implement preprocessing
    def __init__(self, path, num_of_layers):
        self.path = path
        self.num_of_layers = num_of_layers

        # Load channel array
        self.all_channels = self.load_data()
        # Slice channel array and drop patches, which don't contain relevant data
        self.slice_and_filter(self.all_channels, (1000, 500))

    def load_data(self):
        # ToDo: Norm the matrix (all layers by themselves or the whole matrix?)
        """
        Loads the tif file into a numpy array
        :return: all_channels_raw: Array that contains all channel data, layer by layer
        """
        file = gdal.Open(self.path)

        all_channels_raw = np.array(file.GetRasterBand(1).ReadAsArray())
        all_channels_raw = np.expand_dims(all_channels_raw, 2)

        # Add layer by layer to the numpy array 'all_channels_raw'
        for i in range(1, self.num_of_layers):
            channel = np.expand_dims(np.array(file.GetRasterBand(i + 1).ReadAsArray()), 2)
            all_channels_raw = np.append(all_channels_raw, channel, 2)

        return all_channels_raw

    def slice_and_filter(self, all_channels_raw, slice_size):
        # ToDo: Don't forget to process the label layer (e.g. layer 11)
        # ToDo: Export parts of this function to new functions to shrink the size of this function
        # ToDo: Enumerate every slice with a number to track its relative position
        # ToDo: Output is a 4D tensor
        # ToDo: Also a suitable idea: define centroids for each small patch and take patches of different size

        all_channels_sliceable = self.prepare_slicing(all_channels_raw, slice_size)
        shape = all_channels_sliceable.shape

        num_slices = (int(shape[0]/slice_size[0]), int(shape[1]/slice_size[1]))

        for rows in range(num_slices[0]):
            for cols in range(num_slices[1]):
                return # Vorsicht beim Slicing mit den Indizes! D[0:3] 0 ist inkl. 3 ist exkl.

    def prepare_slicing(self, all_channels_raw, slice_size):
        """
        Prepares the channel array for slicing and therefore compensates the slice size surplus by adding more
        "non-data" values
        :param all_channels_raw: Uncompensated channel array
        :param slice_size: Defines the size of the slicing kernel
        :return: all_channels_sliceable: Channel array ready for slicing
        """
        shape = all_channels_raw.shape

        # Calculate the "slicing surplus" for the given slice size towards axis 0 and 1
        surp_0 = slice_size[0] - (shape[0] % slice_size[0])
        surp_1 = slice_size[1] - (shape[1] % slice_size[1])

        # Calculate compensation matrices
        comp_0 = np.zeros((surp_0, shape[1], shape[2]))
        # comp_0[comp_0 == 0] = 65535.00000 # ToDo: Delete after implementing a normalization
        comp_1 = np.zeros((shape[0] + surp_0, surp_1, shape[2]))
        # comp_1[comp_1 == 0] = 65535.00000 # ToDo: Delete after implementing a normalization

        # Compensate the surplus on both axes
        all_channels_raw = np.append(all_channels_raw, comp_0, 0)
        all_channels_slizeable = np.append(all_channels_raw, comp_1, 1)

        return all_channels_slizeable


if __name__ == "__main__":
    ds = MultispectralDataset("C:/Users/Mikey/Documents/MEGAsync/Master/Forschungsarbeit/Daten/Weizen_mit_Rost_Ortenau_04-07-21_EPSG_25831.tif", 10)
