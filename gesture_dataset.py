import os
import tqdm

import torch
from torch.utils.data import Dataset

import numpy as np
from utils import aedat2torch
import MinkowskiEngine as ME

class SparseDvsGestureDataset(Dataset):
    def __init__(self, args, mode="train"):
        self.mode = mode
        self.T = args.T
        self.sample_size = args.sample_size
        self.quantization_size = [args.sample_size//args.T,1,1]
        self.h, self.w = args.image_shape
        self.quantized_w = self.w // self.quantization_size[1]
        self.quantized_h = self.h // self.quantization_size[2]

        save_file_name = f"dvsgesture_{mode}_{self.sample_size//1000}_{self.quantization_size[0]/1000}ms.pt"
        save_file = os.path.join(args.path, save_file_name)
        
        if os.path.isfile(save_file):
            self.samples = torch.load(save_file)
            print("File loaded.")
        else:
            mode_files = f"trials_to_{mode}.txt"
            self.samples = self.build_dataset(args.path, mode_files, save_file)
            torch.save(self.samples, save_file)
            print(f"Done! File saved as {save_file}")

    def build_dataset(self, path, mode_files, save_file):
        print("Building the Dataset")
        samples = []

        with open(os.path.join(path, mode_files), 'r') as files:
            pbar = tqdm.tqdm(unit='File', unit_scale=True)
            for f in files:
                f = f.rstrip("\n")
                print("Processing {} file".format(f))
                file = os.path.join(path, f)
                data = aedat2torch(file).t() # x,y,ts,p

                labels_file = file.replace(".aedat","_labels.csv")
                labels_csv = np.genfromtxt(labels_file, delimiter=',')[1:,:] # skip header

                # add labels and indices
                for i in range(labels_csv.shape[0]):
                    classLabel, start_range, end_range = labels_csv[i,:]
                    indices_sample = torch.where((data[:,2] >= start_range) & (data[:,2] < start_range+self.sample_size))[0]
                    sample = data[indices_sample,:]
                    label = torch.LongTensor([int(classLabel-1)])
                    
                    locations_list = []
                    features_list = []

                    first_ts = sample[0,2]

                    for (x,y,ts,p) in sample:
                        ts_rescaled = max(0, min(int(ts-first_ts), self.sample_size-1))
                        locations_list.append([ts_rescaled, y, x])
                        if p==0: features_list.append([-1])
                        else: features_list.append([1])

                    locations = torch.FloatTensor(locations_list) # [[ts, y, x, batchIdx]]
                    features = torch.FloatTensor(features_list) # [[p]]

                    # Quantize the input
                    discrete_coords, unique_feats = ME.utils.sparse_quantize(
                        coordinates=locations,
                        features=features,
                        quantization_size=self.quantization_size)
                    
                    samples.append((discrete_coords, unique_feats,label))
                    
                pbar.update(1)
                
        pbar.close()
        return samples
       
    def __getitem__(self, index):
        return self.samples[index] # discrete_coords, unique_feats, label

    def __len__(self):
        return len(self.samples)