import torch
import numpy as np
from torchvision import transforms
import cv2
import skimage.io as io
from torch.utils.data import Dataset


class DiodeDataLoader(Dataset):
    """Some Information about DiodeDataLoader"""

    def __init__(self, data_frame, max_depth, img_dim=(192, 256), depth_dim=(192, 256)):
        super(DiodeDataLoader, self).__init__()
        self.data_frame = data_frame
        self.min_depth = 0.1
        self.max_depth = max_depth
        self.img_dim = img_dim
        self.depth_dim = depth_dim
        
        self.transform = transforms.Compose([
                            transforms.ToTensor(), 
                            transforms.Resize(self.img_dim)
                        ])

    def __getitem__(self, index):
        # start = t.time()
        image_path, depth_path, mask_path = self.data_frame.iloc[index]
        
        # Todo: read images
        img = io.imread(image_path)
        img = self.transform(img)

        # Todo: depth processing
        depth_map = np.load(depth_path).squeeze()
        mask = np.load(mask_path)
        mask = mask > 0

        max_depth = min(self.max_depth, np.percentile(depth_map, 99))
        depth_map = np.clip(depth_map, self.min_depth, max_depth)

        depth_map /= self.max_depth      
        depth_map = np.ma.masked_where(~mask, depth_map)

        # * resize depth map yang telah di proses
        depth_map = cv2.resize(depth_map, (self.depth_dim[1], self.depth_dim[0]))
        depth_map = np.expand_dims(depth_map, axis=0)
        depth_map = torch.tensor(depth_map, dtype=torch.float32)
        
        return img, depth_map

    def __len__(self):
        return len(self.data_frame)
