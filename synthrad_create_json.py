from tqdm import tqdm
import json
from collections import defaultdict

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

if __name__ == "__main__":
    dataset_dir=r"E:\Projects\yang_proj\data\synthrad"
    accepted_modalities = ["ct", "mr", "mask", "cbct"]
    create_new_json = False
    json_file = "./datasets/dataset.json"
    slice_info_file = "./datasets/slice_info.json"
    if create_new_json:
        images_list = make_dataset_modality(dataset_dir, accepted_modalities, None)
        create_metadata_jsonl(images_list, json_file)
    else:
        
        dataset = read_metadata_jsonl(json_file)
        print(f"Dataset length: {len(dataset)}")
        print(f"first element in the dataset: \n {dataset[0]}")
        from synthrad_huggingface_dataset import SynthradDataset
        from torch.utils.data import DataLoader
        dataset = SynthradDataset(json_file, 
                                  mode='train', slice_axis=2, 
                                  resolution=256,
                                  slice_info_file=slice_info_file,
                                use_saved_slice_info=True)
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