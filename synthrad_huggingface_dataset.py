import SimpleITK as sitk
import numpy as np
import torch
from datasets import Dataset as huggingfaceDataset
from tqdm import tqdm
from monai.transforms import ResizeWithPadOrCrop
import json
from typing import (
    TYPE_CHECKING,
    Any,
    BinaryIO,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Tuple,
    Union,
    overload,
)

VERBOSE = False
class SynthradDataset(huggingfaceDataset):
    def __init__(self, json_path, mode='train', slice_axis=2, tokenizer=None):
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
        self.tokenizer = tokenizer
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
        with open(r".\logs\slice_info.json", 'w') as f:
            json.dump(slice_info, f, indent=4)
        return slice_info

    def __len__(self):
        return len(self.slice_info)

    def __getitem__(self, idx: Union[int, List[int]]) -> Union[Dict, List[Dict]]:
        '''
        for huggingface dataset, batch should be a dictionary:
        batch = {
            "original_image": [img1, img2, img3],
            "edited_image": [edited_img1, edited_img2, edited_img3],
            "edit_prompt": ["prompt1", "prompt2", "prompt3"]
        }
        '''
        if isinstance(idx, int):
            return self.__get_single_item__(idx)
        elif isinstance(idx, list):
            return self.__get_batch_items__(idx)
        else:
            raise TypeError(f"Invalid index type: {type(idx)}. Expected int or list of int.")
    
        
    def __get_single_item__(self, idx: int) -> Dict:
        entry, slice_idx = self.slice_info[idx]
        original_image = self._load_file(entry['original_image'])
        original_slice = self._slice_volume(original_image, slice_idx)

        edited_image = self._load_file(entry['edited_image'])
        edited_slice = self._slice_volume(edited_image, slice_idx)
        
        un_tokenized_prompt = entry['edit_prompt']

        tokenized_prompt = self.tokenizer(
            un_tokenized_prompt, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        edit_prompt = tokenized_prompt.input_ids

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
        resize = ResizeWithPadOrCrop(spatial_size=(512, 512), mode="minimum")
        original_slice = resize(original_slice)
        edited_slice = resize(edited_slice)

        # Duplicate the single channel to create a three-channel image
        original_slice = original_slice.repeat(3, 1, 1)
        edited_slice = edited_slice.repeat(3, 1, 1)

        single_item = {"original_image": original_slice, 
                 "edited_image": edited_slice,
                 "edit_prompt": edit_prompt}
        return single_item

    def __get_batch_items__(self, indices: List[int]) -> Dict[str, List]:
        batch = {"original_image": [], "edited_image": [], "edit_prompt": []}
        for idx in indices:
            item = self.__get_single_item__(idx)
            for key in batch.keys():
                batch[key].append(item[key])
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

def collate_fn(examples):
    original_pixel_values = torch.stack([example["original_image"] for example in examples])
    original_pixel_values = original_pixel_values.to(memory_format=torch.contiguous_format).float()
    edited_pixel_values = torch.stack([example["edited_image"] for example in examples])
    edited_pixel_values = edited_pixel_values.to(memory_format=torch.contiguous_format).float()
    input_ids = torch.stack([example["edit_prompt"] for example in examples])
    return {
        "original_pixel_values": original_pixel_values,
        "edited_pixel_values": edited_pixel_values,
        "input_ids": input_ids,
    }

if __name__ == "__main__":
    json_file = "./logs/dataset_test.json"
    pretrained_model_name_or_path = "runwayml/stable-diffusion-v1-5"
    from transformers import CLIPTextModel, CLIPTokenizer
    tokenizer = CLIPTokenizer.from_pretrained(
        pretrained_model_name_or_path, subfolder="tokenizer", revision=None)

    dataset = SynthradDataset(json_file, mode='train', slice_axis=2, tokenizer=tokenizer)
    
    train_dataloader=torch.utils.data.DataLoader(
        dataset, 
        collate_fn=collate_fn,
        batch_size=4, 
        shuffle=False
        
        )
    
    print("Length of dataset:", len(dataset))
    for step, batch in enumerate(train_dataloader):
        data = batch["original_pixel_values"]
        label = batch["edited_pixel_values"]
        input_ids = batch["input_ids"]
        print(data.shape)
        print(label.shape)
        print(input_ids.shape)
        break
        