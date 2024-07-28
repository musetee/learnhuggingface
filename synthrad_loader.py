import SimpleITK as sitk
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from monai.transforms import ResizeWithPadOrCrop

VERBOSE = False

import json
import os

IMG_EXTENSIONS = [
    #'.jpg', '.JPG', '.jpeg', '.JPEG',
    #'.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', 
    '.nrrd', '.nii.gz'
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset_modality(dir, accepted_modalities = ["ct"], saved_name=None):
    # it works for root path of any layer:
    # data_path/Task1 or Task2/pelvis or brain
            # |-patient1
            #   |-ct.nii.gz
            #   |-mr.nii.gz
            # |-patient2
            #   |-ct.nii.gz
            #   |-mr.nii.gz
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    for roots, _, files in sorted(os.walk(dir)): # os.walk digs all folders and subfolders in all layers of dir
        for file in files:
            if is_image_file(file) and file.split('.')[0] in accepted_modalities:
                path = os.path.join(roots, file)
                images.append(path)
    print(f'Found {len(images)} {accepted_modalities} files in {dir} \n')
    if saved_name is not None:
        with open(saved_name,"w") as file:
            for image in images:
                file.write(f'{image} \n')
    return images

import json
from collections import defaultdict
def create_metadata_jsonl(file_paths, output_file= "dataset.json"):
    # Organize files by patient ID
    files_by_patient = defaultdict(dict)
    for file_path in file_paths:
        parts = file_path.split("\\")
        patient_id = parts[-2]
        modality = parts[-1].split('.')[0]  # e.g., 'ct', 'mr', 'cbct', 'mask'
        files_by_patient[patient_id][modality] = file_path

    # Generate JSON structure
    dataset = []
    for patient_id, files in files_by_patient.items():
        mask_path = files.get('mask')
        parts2 = mask_path.split("\\")
        patient_dir = "\\".join(parts2[:-1])
        if mask_path:
            for modality in ['ct', 'mr', 'cbct']:
                image_path = files.get(modality)
                if image_path:
                    entry = {
                        "patient_name": patient_dir,
                        "original_image": mask_path,
                        "edited_image": image_path,
                        "edit_prompt": f"a brain {modality} image"
                    }
                    dataset.append(entry)

    # Save to JSON file
    with open(output_file, 'w') as f:
        json.dump(dataset, f, indent=4)

    print(f"Dataset saved to {output_file}")
   
def read_metadata_jsonl(file_path):
    with open(file_path, 'r') as f:
        dataset = json.load(f)
    return dataset

def print_json_info(data_info):
        for entry in tqdm(data_info, desc="Calculating slice info"):
            print(entry['patient_name'])

def main():

    dataset_dir=r"D:\Projects\data\synthrad\train"
    accepted_modalities = ["ct", "mr", "mask", "cbct"]
    create_new_json = False
    if create_new_json:
        images_list = make_dataset_modality(dataset_dir, accepted_modalities, None)
        create_metadata_jsonl(images_list, "./logs/dataset.json")
    else:
        json_file = "./logs/dataset_test.json"
        dataset = read_metadata_jsonl(json_file)
        print(f"Dataset length: {len(dataset)}")
        print(dataset[0])
        dataset = SynthradDataset(json_file, mode='train', slice_axis=2)
        dataloader=DataLoader(dataset, batch_size=4, shuffle=True)

        print("Length of dataset:", len(dataset))
        for batch in dataloader:
            data = batch["original_image"]
            label = batch["edited_image"]
            print(data.shape)
            print(label.shape)
            print(batch["edit_prompt"])
            break
            #print_json_info(dataset)


class synthradDataset_old(Dataset):
    def __init__(self, file_ids, mode='train', transform_list=None, slice_axis=2):
        """
        Args:
            file_ids (list): List of file ids to load data from.
            mode (str): 'train' or 'test'. Determines if augmentation is applied.
            transform_list (list of callable, optional): List of transforms to be applied on a sample.
            slice_axis (int): The axis along which to slice the 3D volumes (0, 1, or 2).
        """
        self.file_ids = file_ids
        self.mode = mode
        self.transform_list = transform_list
        self.slice_axis = slice_axis
        self.data_slices, self.label_slices = self._load_and_slice_all_files()

    def __len__(self):
        return len(self.data_slices)

    def __getitem__(self, idx):
        data_slice = self.data_slices[idx]
        label_slice = self.label_slices[idx]

        # Apply additional transforms if specified
        if self.transform_list:
            for transform in self.transform_list:
                data_slice, label_slice = transform(data_slice, label_slice)

        # Expand dimensions to include channel dimension
        data_slice = np.expand_dims(data_slice, axis=0)
        label_slice = np.expand_dims(label_slice, axis=0)
        
        # Convert to torch tensors
        data_slice = torch.from_numpy(data_slice).float()
        label_slice = torch.from_numpy(label_slice).float()
        
        batch = {"data": data_slice, 
                 "label": label_slice}
        return batch

    def _load_and_slice_all_files(self):
        data_slices = []
        label_slices = []
        for file_id in self.file_ids:
            data_img, label_img = self._load_file(file_id)
            slices_data, slices_label = self._slice_volume(data_img, label_img)
            data_slices.extend(slices_data)
            label_slices.extend(slices_label)
        return data_slices, label_slices

    def _load_file(self, file_id):
        _, file_number = os.path.split(file_id)

        data_img = sitk.ReadImage(os.path.join(file_id + '_sino_Metal.nrrd'))
        label_img = sitk.ReadImage(os.path.join(file_id + '_img_GT_noNoise.nrrd'))

        data_img = sitk.GetArrayFromImage(data_img)
        label_img = sitk.GetArrayFromImage(label_img)

        data_img = np.moveaxis(data_img, 0, -1)
        label_img = np.moveaxis(label_img, 0, -1)

        data_img, label_img = self.adapt_to_task(data_img, label_img)
        self._check_images(data_img, label_img)
        return data_img, label_img

    def _slice_volume(self, data_img, label_img):
        if self.slice_axis == 0:
            data_slices = [data_img[i, :, :] for i in range(data_img.shape[0])]
            label_slices = [label_img[i, :, :] for i in range(label_img.shape[0])]
        elif self.slice_axis == 1:
            data_slices = [data_img[:, i, :] for i in range(data_img.shape[1])]
            label_slices = [label_img[:, i, :] for i in range(label_img.shape[1])]
        elif self.slice_axis == 2:
            data_slices = [data_img[:, :, i] for i in range(data_img.shape[2])]
            label_slices = [label_img[:, :, i] for i in range(label_img.shape[2])]
        else:
            raise ValueError(f"Invalid axis: {self.slice_axis}. Axis must be 0, 1, or 2.")

        return data_slices, label_slices
    
    def _preprocess(self, data):
        if self.mode == 'train':
            for sample in range(data.shape[0]):
                interval = 10
                variation = np.random.randint(-interval, interval)
                data[sample, :, :, 0] = data[sample, :, :, 0] + variation
                interval = 2
                variation = np.random.randint(-interval, interval)
                data[sample, :, :, 1] = data[sample, :, :, 1] + variation

        data = self.normalize(data)
        if data.ndim < 4:
            data = np.expand_dims(data, axis=-1)
        return data

    @staticmethod
    def adapt_to_task(data_img, label_img):
        return data_img, label_img

    def _check_images(self, data, lbl):
        print('            Data :     ', data.shape, np.max(data), np.min(data))
        print('            Label:       ', lbl.shape, np.max(lbl), np.min(lbl))
        print('-------------------------------------------')
        pass

class SynthradDataset(Dataset):
    def __init__(self, json_path, mode='train', slice_axis=2):
        """
        Args:
            json_path (str): Path to the JSON file containing the dataset information.
            mode (str): 'train' or 'test'. Determines if augmentation is applied.
            transform_list (list of callable, optional): List of transforms to be applied on a sample.
            slice_axis (int): The axis along which to slice the 3D volumes (0, 1, or 2).

        JSON file structure:    
            entry = {
                    "patient_name": patient_dir,
                    "original_image": mask_path,
                    "edited_image": image_path,
                    "edit_prompt": f"a brain {modality} image"
                }

        """
        self.json_path = json_path
        self.mode = mode
        self.slice_axis = slice_axis
        
        self.data_info = self._load_json()
        self.slice_info = self._calculate_slice_info()

    def _load_json(self):
        with open(self.json_path, 'r') as file:
            data_info = json.load(file)
        return data_info

    def _calculate_slice_info(self):
        slice_info = []
        for entry in tqdm(self.data_info, desc="Calculating slice info"):
            data_img = self._load_file(entry['original_image'])
            num_slices = data_img.shape[self.slice_axis]
            for i in range(num_slices):
                slice_info.append((entry, i))
        print (f"slice info: {slice_info}")
        return slice_info

    def __len__(self):
        return len(self.slice_info)
    
    def __getitem__(self, idx):
        entry, slice_idx = self.slice_info[idx]
        original_image = self._load_file(entry['original_image'])
        original_slice = self._slice_volume(original_image, slice_idx)

        edited_image = self._load_file(entry['edited_image'])
        edited_slice = self._slice_volume(edited_image, slice_idx)

        edit_prompt = entry['edit_prompt']

         # Apply additional transforms if specified

        # Normalize
        scale_factor=3000
        edited_slice = (edited_slice - np.min(edited_slice)) / scale_factor

        # Expand dimensions to include channel dimension
        original_slice = np.expand_dims(original_slice, axis=0)
        edited_slice = np.expand_dims(edited_slice, axis=0)
        
        # Convert to torch tensors
        original_slice = torch.from_numpy(original_slice).float()
        edited_slice = torch.from_numpy(edited_slice).float()
        
        # Resize
        resize = ResizeWithPadOrCrop(spatial_size=(256, 256), mode="minimum")
        original_slice = resize(original_slice)
        edited_slice = resize(edited_slice)


        batch = {"original_image": original_slice, 
                 "edited_image": edited_slice,
                 "edit_prompt": edit_prompt}
        return batch

    def _load_file(self, file_id):
        data_img = sitk.ReadImage(file_id)

        data_img = sitk.GetArrayFromImage(data_img)

        data_img = np.moveaxis(data_img, 0, -1)

        if VERBOSE:
            self._check_images(data_img)
        return data_img

    def _slice_volume(self, data_img, slice_idx):
        if self.slice_axis == 0:
            data_slice = data_img[slice_idx, :, :]
        elif self.slice_axis == 1:
            data_slice = data_img[:, slice_idx, :]
        elif self.slice_axis == 2:
            data_slice = data_img[:, :, slice_idx]
        else:
            raise ValueError(f"Invalid axis: {self.slice_axis}. Axis must be 0, 1, or 2.")

        return data_slice

    def _preprocess(self, data):
        if self.mode == 'train':
            for sample in range(data.shape[0]):
                interval = 10
                variation = np.random.randint(-interval, interval)
                data[sample, :, :, 0] = data[sample, :, :, 0] + variation
                interval = 2
                variation = np.random.randint(-interval, interval)
                data[sample, :, :, 1] = data[sample, :, :, 1] + variation

        data = self.normalize(data)
        if data.ndim < 4:
            data = np.expand_dims(data, axis=-1)
        return data

    @staticmethod
    def adapt_to_task(data_img, label_img):
        return data_img, label_img

    def _check_images(self, data):
        print('            Data :     ', data.shape, np.max(data), np.min(data))
        print('-------------------------------------------')
        pass

class SliceTransform:
    def __init__(self, axis=2):
        """
        Args:
            axis (int): The axis along which to slice the 3D volume (0, 1, or 2).
        """
        self.axis = axis

    def __call__(self, data_img, label_img):
        """
        Slices the 3D volumes into 2D slices along the specified axis.

        Args:
            data_img (numpy.ndarray): 3D data volume.
            label_img (numpy.ndarray): 3D label volume.

        Returns:
            data_slices (list of numpy.ndarray): List of 2D data slices.
            label_slices (list of numpy.ndarray): List of 2D label slices.
        """
        if self.axis == 0:
            data_slices = [data_img[i, :, :] for i in range(data_img.shape[0])]
            label_slices = [label_img[i, :, :] for i in range(label_img.shape[0])]
        elif self.axis == 1:
            data_slices = [data_img[:, i, :] for i in range(data_img.shape[1])]
            label_slices = [label_img[:, i, :] for i in range(label_img.shape[1])]
        elif self.axis == 2:
            data_slices = [data_img[:, :, i] for i in range(data_img.shape[2])]
            label_slices = [label_img[:, :, i] for i in range(label_img.shape[2])]
        else:
            raise ValueError(f"Invalid axis: {self.axis}. Axis must be 0, 1, or 2.")

        return data_slices, label_slices

class Normalize:
    def __init__(self, type):
        self.type = type
    
    def __call__(self, label_img):
        if self.type == "window":
            #print("windowing")
            label_img = self.windowing(label_img)
        elif self.type == "scale":
            #print("scaling")
            label_img = self.scaling(label_img)
        return label_img
    def windowing(self, img):
        img_min=-100
        img_max=500
        img = np.clip(img, img_min, img_max)
        img = (img - img_min) / (img_max-img_min)
        return img
    def scaling(self, img):
        scale_factor=3000
        img = (img - np.min(img)) / scale_factor
        return img
    

if __name__ == "__main__":
    '''    
    test_csv = r"./data_table/test.csv"
    test_data = pd.read_csv(test_csv, dtype=object)
    test_filelist = test_data.iloc[:, -1].tolist()
    print(test_filelist)
    transform_list = []
    transform_list.append(Normalize(type="scale"))
    
    dataset = synthradDataset(file_ids=test_filelist, mode='train', transform_list=transform_list, slice_axis=2)
    dataloader=DataLoader(dataset, batch_size=4, shuffle=True)
    
    print("Length of dataset:", len(dataset))
    for batch in dataloader:
        data = batch["data"]
        label = batch["label"]
        print(data.shape)
        print(label.shape)
        break'''
    main()