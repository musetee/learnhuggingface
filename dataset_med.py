from torch.utils.data import DataLoader, Dataset
import torch.utils.data as data
import os.path
import random
from torchvision import transforms
from PIL import Image
import torch
from PIL import ImageFile

import nibabel as nib
import numpy as np

Image.MAX_IMAGE_PIXELS = None  # Disable DecompressionBombError
ImageFile.LOAD_TRUNCATED_IMAGES = True  # Disable OSError: image file is truncated



IMG_EXTENSIONS = [
    #'.jpg', '.JPG', '.jpeg', '.JPEG',
    #'.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', 
    '.nrrd', '.nii.gz'
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset_modality(dir, modality='ct'):
    images = []
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
    for patient_folder, _, fnames in sorted(os.walk(dir)): # means that it will go through all the files in the directory
        #print(patient_folder)
        if patient_folder != dir:
            #print('patient folder:',patient_folder)
            for root2, _, fnames2 in sorted(os.walk(patient_folder)):
                #print('files:',fnames2)
                for fname2 in fnames2:
                    if is_image_file(fname2) and modality in fname2:
                        #print('passed file:',fname2)
                        path = os.path.join(root2, fname2)
                        images.append(path)
    return images


class CTImageDataset(Dataset):
    def __init__(self, root, modality='ct', transform=None, 
                 load_patient_number=1):
        self.imgs_paths = sorted(make_dataset_modality(root, modality))
        self.transform = transform
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
            volume_tensor = volume_tensor.permute(2, 1, 0) # [N, H, W]
            volume_tensor = volume_tensor.unsqueeze(3)  # Add channel dimension [N, H, W] -> [N, H, W, 1]
            # pasting grayscale information to all three channels.
            volume_tensor = volume_tensor.repeat(1, 1, 1, 3)
            #print('Debug, volume tensor:',volume_tensor.shape)
            if self.transform is not None:
                volume_tensor = self.transform(volume_tensor)
            if all_slices is None:
                all_slices = volume_tensor
            else:
                all_slices = torch.cat((all_slices, volume_tensor), 0)
        print(f'slices of {modality} dataset:',all_slices.shape)
        self.all_slices = all_slices

    def __getitem__(self, index):
        img = self.all_slices[index]
        # permute img from [H, W, C] to [C, H, W]
        img = img.permute(2, 0, 1)
        return {'img': img}

    def __len__(self):
        return self.all_slices.shape[0]

from monai.transforms import (
    ResizeWithPadOrCrop,
    ScaleIntensity,
    Compose,
)

def get_data_loader_folder(input_folder, modality, 
                           batch_size, new_size=288, 
                           height=256, width=256, 
                           num_workers=None, load_patient_number=1):
    transform_list = []
    transform_list = [ResizeWithPadOrCrop(spatial_size=[height,width, -1],mode="minimum")] + transform_list
    transform_list = [ScaleIntensity(minv=0, maxv=1.0)]+ transform_list
    #transform_list = [ScaleIntensity(factor=-0.9)]+ transform_list
    #transform_list = [transforms.Resize(new_size)] + transform_list
    transform = Compose(transform_list)

    dataset = CTImageDataset(input_folder, modality=modality, transform=transform, load_patient_number=load_patient_number)
    
    if num_workers is None:
        num_workers = 0
    loader = DataLoader(dataset=dataset, 
                        batch_size=batch_size, 
                        drop_last=True, 
                        num_workers=num_workers, 
                        sampler=InfiniteSamplerWrapper(dataset), 
                        collate_fn=collate_fn
                        )
    return loader

def main(root = r'C:\Users\56991\Projects\Datasets\Task1\pelvis',modality='ct'):
    # Example usage
    
    batch_size = 8
    new_size = 512
    height = 512
    width = 512
    num_workers = None
    load_patient_number = 1
    loader = get_data_loader_folder(root,modality, batch_size, new_size, height, width, num_workers, load_patient_number)
    #print length of loader
    print('Length of loader:',len(loader))
    for i, batch in enumerate(loader):
        print(f'Batch {i}:',batch['img'].shape)
    print('Done')

if __name__=='__main__':
    main()
    

def InfiniteSampler(n):
    # i = 0
    i = n - 1
    order = np.random.permutation(n)
    while True:
        yield order[i]
        i += 1
        if i >= n:
            np.random.seed()
            order = np.random.permutation(n)
            i = 0


class InfiniteSamplerWrapper(data.sampler.Sampler):
    def __init__(self, data_source):
        self.num_samples = len(data_source)

    def __iter__(self):
        return iter(InfiniteSampler(self.num_samples))

    def __len__(self):
        return 2 ** 31


def collate_fn(batch):
    img = [b['img'] for b in batch]
    img = torch.stack(img, dim=0)

    laplacian_m = [b['laplacian_m'] for b in batch]

    return {'img': img, 'laplacian_m': laplacian_m}

