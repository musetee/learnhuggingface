from torch.utils.data import DataLoader, Dataset, random_split
import torch.utils.data as data
import os.path
import random
from torchvision import transforms
from PIL import Image
import torch
from PIL import ImageFile
#from utils.MattingLaplacian import compute_laplacian

import nibabel as nib
import numpy as np
import os
import csv
import pandas as pd
from transformers import CLIPTokenizer

Image.MAX_IMAGE_PIXELS = None  # Disable DecompressionBombError
ImageFile.LOAD_TRUNCATED_IMAGES = True  # Disable OSError: image file is truncated



IMG_EXTENSIONS = [
    #'.jpg', '.JPG', '.jpeg', '.JPEG',
    #'.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', 
    '.nrrd', '.nii.gz'
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset_synthrad(dir, modality='ct'):
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    # for image data in the following structure:
    # root/
    #     patient_folder/
    #         ct_image.nii.gz
    #         mr_image.nii.gz
    #         ...
    #     patient_folder2/
    #         ct_image.nii.gz
    #         mr_image.nii.gz
    #         ...
    images = []
    for patient_folder, _, fnames in sorted(os.walk(dir)): # means that it will go through all the files in the directory
        #print(patient_folder)
        if patient_folder != dir:
            #print('patient folder:',patient_folder)
            for root2, _, fnames2 in sorted(os.walk(patient_folder)):
                #print('files:',fnames2)
                for fname2 in fnames2:
                    if is_image_file(fname2) and (modality in fname2):
                        #print('passed file:',fname2)
                        path = os.path.join(root2, fname2)
                        images.append(path)
    return images

# load dataset according to the folder structure and the modality information
class ImageDataset(Dataset):
    def __init__(self, root, modality='ct', tissue='pelvis', transform=None, load_patient_number=1):
        self.imgs_paths = sorted(make_dataset_synthrad(root, modality))
        self.transform = transform
        self.tissue = tissue
        self.modality = modality
        self.to_tensor = transforms.ToTensor()  # Might need adjustment for 3D

        if len(self.imgs_paths) == 0:
            raise RuntimeError(f"Found 0 images in: {root}")
        # form the images to be in the form of [D, H, W]
        all_slices = None
        for img_path in self.imgs_paths[:load_patient_number]:
            volume = nib.load(img_path)
            volume_data = volume.get_fdata() # load as [H, W, D]
            # 
            # Convert numpy array to PyTorch tensor
            # Note: You might need to add channel dimension or perform other adjustments
            volume_tensor = torch.tensor(volume_data, dtype=torch.float32)
            volume_tensor = volume_tensor.permute(2, 1, 0) # [D, H, W]
            volume_tensor = volume_tensor.unsqueeze(1)  # Add channel dimension [D, H, W] -> [D, 1, H, W]
            if self.transform is not None:
                volume_tensor = self.transform(volume_tensor)

            print('volume tensor:',volume_tensor.shape)
            if all_slices is None:
                all_slices = volume_tensor
            else:
                all_slices = torch.cat((all_slices, volume_tensor), 0)
        print('slices:',all_slices.shape)
        self.all_slices = all_slices

    def __getitem__(self, index):
        img_capture = f"This is the {index}th slice of a {self.modality} volume of the {self.tissue}"
        return {'img': self.all_slices[index], 'text': img_capture}

    def __len__(self):
        return self.all_slices.shape[0]

def write_synthrad_csv(med_info_pairs={"root": "path/to/data", "modality": "ct", "tissue": "pelvis"}):
    with open('data.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['path_to_image', 'modality', 'tissue'])  # Writing the headers

        for med_info in med_info_pairs:
            root = med_info["root"]
            modality = med_info["modality"]
            tissue = med_info["tissue"]
            images = make_dataset_synthrad(root, modality)

            for image_path in images:
                writer.writerow([image_path, modality, tissue])

# load dataset according to the csv file
class MedImageDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        """
        csv_file: Path to the CSV file containing image paths and labels.
        """
        self.data_info = pd.read_csv(csv_file)
        self.transform = transform
        self.to_tensor = transforms.ToTensor()
        self.images = []
        self.labels = []

        for _, row in self.data_info.iterrows():
            path, modality, tissue = row['path_to_image'], row['modality'], row['tissue']
            volume = nib.load(path)
            volume_data = volume.get_fdata()
            volume_tensor = torch.tensor(volume_data, dtype=torch.float32)
            volume_tensor = volume_tensor.permute(2, 0, 1) # [D, H, W]
            volume_tensor = volume_tensor.unsqueeze(3)  # Add channel dimension [N, H, W] -> [N, H, W, 1]
            # pasting grayscale information to all three channels.
            volume_tensor = volume_tensor.repeat(1, 1, 1, 3) # expand it to three channels

            if self.transform is not None:
                volume_tensor = self.transform(volume_tensor)

            for i in range(volume_tensor.shape[0]):
                self.images.append(volume_tensor[i])
                title = f"This is the {i+1}th slice of a {modality} volume of the {tissue}"
                self.labels.append(title)

    def __getitem__(self, index):
        return {'img': self.images[index], 'text': self.labels[index]}

    def __len__(self):
        return len(self.images)
    
from monai.transforms import (
    ResizeWithPadOrCrop,
    Compose,
    ThresholdIntensity,
    ScaleIntensity,
)

def get_data_loader_folder(csv_file, batch_size, height=256, width=256, drop_last=False, num_workers=None, ifsplit=False):
    WINDOW_LEVEL,WINDOW_WIDTH = 50, 800
    min, max=WINDOW_LEVEL-(WINDOW_WIDTH/2), WINDOW_LEVEL+(WINDOW_WIDTH/2)
    transform_list = []
    transform_list = [ResizeWithPadOrCrop(spatial_size=[height,width,-1],mode="minimum")] + transform_list
    transform_list = [ThresholdIntensity(threshold=min, above=True, cval=min)] + transform_list
    transform_list = [ThresholdIntensity(threshold=max, above=False, cval=max)] + transform_list
    transform_list = [ScaleIntensity(minv=0, maxv=1)] + transform_list
    #transform_list = [transforms.Resize(new_size)] + transform_list
    transform = Compose(transform_list)

    dataset = MedImageDataset(csv_file, transform=transform)
    if num_workers is None:
            num_workers = 0
    if not ifsplit:
        train_loader = DataLoader(dataset=dataset, 
                        batch_size=batch_size, 
                        drop_last=drop_last, 
                        num_workers=num_workers, 
                        #sampler=InfiniteSamplerWrapper(dataset), 
                        #collate_fn=collate_fn
                        )
        val_loader = None
    else:
        train_size = int(0.9 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        train_loader = DataLoader(dataset=train_dataset, 
                            batch_size=batch_size, 
                            drop_last=drop_last, 
                            num_workers=num_workers, 
                            #sampler=InfiniteSamplerWrapper(dataset), 
                            #collate_fn=collate_fn
                            )
        val_loader = DataLoader(dataset=val_dataset,
                            batch_size=batch_size,
                            drop_last=drop_last,
                            num_workers=num_workers,
                            #sampler=InfiniteSamplerWrapper(dataset),
                            #collate_fn=collate_fn
                            )
    return train_loader, val_loader

# save the images in the dataset to a folder in the following form:
# folder/train/metadata.jsonl
# folder/train/0001.png
# folder/train/0002.png

#{"file_name": "0001.png", "additional_feature": "This is a first value of a text feature you added to your images"}
#{"file_name": "0002.png", "additional_feature": "This is a second value of a text feature you added to your images"}

from monai.transforms import SaveImage
import json
from tqdm import tqdm
def ConvertDatasetToRGB(output_dir= 'datafolder/train'):
    os.makedirs(output_dir, exist_ok=True)
    loader, _ = get_data_loader_folder("test.csv", 1, 512, 512, True, 0, False)
    # The metadata file
    metadata_file = os.path.join(output_dir, 'metadata.jsonl')
    loader_length = len(loader)
    idx = 0
    # Open the metadata file
    with open(metadata_file, 'w') as f:
        for batch in tqdm(loader):
            img = batch['img']
            text = batch['text']
            saved_name = f"{idx:04d}.png"

            fig = plt.figure() #, figsize=(5, 4))
            plt.gca().set_axis_off()
            plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
                        hspace = 0, wspace = 0)
            plt.margins(0,0)
            plt.imshow(img[0,:,:,0].permute(1,0), cmap='gray')
            plt.savefig(os.path.join(output_dir,saved_name), bbox_inches='tight', pad_inches=0, dpi=100)
            plt.close(fig)
            # save the text to a json file
            metadata_entry = {
            "file_name": saved_name,
            "additional_feature": text
            }
            f.write(json.dumps(metadata_entry) + '\n')  # Write as JSON Lines format
            idx += 1
            
from datasets import load_dataset, Image
def test_load_dataset(path="datafolder/train"):
    dataset = load_dataset("imagefolder", data_dir=path, split="train")
    print(dataset[0]["additional_feature"])

import matplotlib.pyplot as plt
def main():
    # Example usage
    
    batch_size = 2
    new_size = 512
    height = 512
    width = 512
    num_workers = None
    load_patient_number = 1
    

    # Example usage
    med_info_pairs = [
        {"root": r'D:\Projects\data\Task1\pelvis', "modality": "ct", "tissue": "pelvis"},
        {"root": r'D:\Projects\data\Task1\brain', "modality": "mr", "tissue": "brain"}
    ]
    #write_synthrad_csv(med_info_pairs)

    loader, _ = get_data_loader_folder("test.csv", batch_size, height, width, True, num_workers, False)
    for i, batch in enumerate(loader):
        print(f'Batch {i}')
        #print(len(batch['img']))
        print(batch['img'].shape)
        print(batch['text'])
        # plot all the images in this batch in grid 
        plt.figure(figsize=(16, 16))
        for i in range(8):
            plt.subplot(4, 4, i+1)
            plt.imshow(batch['img'][i], cmap='gray')
            plt.title(batch['text'][i])
            plt.axis('off')
        plt.show()
    print('Done')

if __name__=='__main__':
    #root = r'C:\Users\56991\Projects\Datasets\Task1\pelvis'
    #root = r'D:\Projects\data\Task1\pelvis'
    test_load_dataset()
    