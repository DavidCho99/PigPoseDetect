import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

'''
https://docs.pytorch.org/tutorials/beginner/data_loading_tutorial.html
Dataset class

torch.utils.data.Dataset is an abstract class representing a dataset.
Your custom dataset should inherit Dataset and override the following methods:

Sample of our dataset will be a dict {'image': image, 'landmarks': landmarks}.
Our dataset will take an optional argument transform so that any required processing can be applied on the sample.
We will see the usefulness of transform in the next section.
'''


class PigPostureDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            img_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir  # for images
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_id = self.annotations.iloc[idx, 0]
        img_path = os.path.join(self.img_dir, img_id) + ".jpg"

        image = Image.open(img_path).convert("RGB")

        target = torch.tensor(self.annotations.iloc[idx, 5], dtype=torch.long)

        if self.transform:
            image = self.transform(image)

        return image, target
