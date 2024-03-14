import monai
import os
import numpy as np
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    SqueezeDimd,
    CenterSpatialCropd,
    Rotate90d,
    ScaleIntensityd,
    ResizeWithPadOrCropd,
    DivisiblePadd,
    
    ThresholdIntensityd,
    NormalizeIntensityd,
    ShiftIntensityd,
    Identityd,
    ScaleIntensityRanged,
    Spacingd,
)
from torch.utils.data import DataLoader
import torch

class monai_loader:
    def __init__(self,configs,paths): 
        self.configs=configs
        self.paths=paths
        self.get_loader()
        self.finalcheck(ifsave=True,ifcheck=False,iftest_volumes_pixdim=False)

    def get_loader(self):
        # volume-level transforms for both image and label
        train_transforms = self.get_transforms(self.configs,mode='train')
        val_transforms = self.get_transforms(self.configs,mode='val')
        indicator_A=self.configs.dataset.indicator_A	
        indicator_B=self.configs.dataset.indicator_B
        self.indicator_A=indicator_A
        self.indicator_B=indicator_B
        train_number=self.configs.dataset.train_number
        val_number=self.configs.dataset.val_number
        train_batch_size=self.configs.dataset.batch_size
        val_batch_size=self.configs.dataset.val_batch_size
        load_masks=self.configs.dataset.load_masks

        # Conditional dictionary keys based on whether masks are loaded
        keys = [indicator_A, indicator_B, "mask"] if load_masks else [indicator_A, indicator_B]


        #list all files in the folder
        file_list=[i for i in os.listdir(self.configs.dataset.data_dir) if 'overview' not in i]
        file_list_path=[os.path.join(self.configs.dataset.data_dir,i) for i in file_list]
        #list all ct and mr files in folder
        
        source_file_list=[os.path.join(j,f'{self.configs.dataset.source_name}.nii.gz') for j in file_list_path] # "ct" for example
        target_file_list=[os.path.join(j,f'{self.configs.dataset.target_name}.nii.gz') for j in file_list_path] # "mr" for example
        mask_file_list=[os.path.join(j,f'{self.configs.dataset.mask_name}.nii.gz') for j in file_list_path]
        
        if load_masks:  
            train_ds = [{indicator_A: i, indicator_B: j, 'mask': k, 'A_paths': i, 'B_paths': j, 'mask_path': k} 
                        for i, j, k in zip(source_file_list[0:train_number], target_file_list[0:train_number], mask_file_list[0:train_number])]
            val_ds = [{indicator_A: i, indicator_B: j, 'mask': k, 'A_paths': i, 'B_paths': j, 'mask_path': k} 
                    for i, j, k in zip(source_file_list[-val_number:], target_file_list[-val_number:], mask_file_list[-val_number:])]
        else:
            train_ds = [{indicator_A: i, indicator_B: j} 
                        for i, j in zip(source_file_list[0:train_number], target_file_list[0:train_number])]
            val_ds = [{indicator_A: i, indicator_B: j} 
                    for i, j in zip(source_file_list[-val_number:], target_file_list[-val_number:])]

        print('all files in dataset:',len(file_list))

        # load volumes and center crop
        center_crop = self.configs.dataset.center_crop
        transformations_crop = [
            LoadImaged(keys=keys),
            EnsureChannelFirstd(keys=keys),
        ]
        if center_crop>0:
            transformations_crop.append(CenterSpatialCropd(keys=keys, roi_size=(-1,-1,center_crop)))
        transformations_crop=Compose(transformations_crop)
        train_crop_ds = monai.data.Dataset(data=train_ds, transform=transformations_crop)
        val_crop_ds = monai.data.Dataset(data=val_ds, transform=transformations_crop)

        # load volumes
        train_volume_ds = monai.data.Dataset(data=train_crop_ds, transform=train_transforms) 
        val_volume_ds = monai.data.Dataset(data=val_crop_ds, transform=val_transforms)

        # batch-level slicer for both image and label
        window_width=1
        patch_func = monai.data.PatchIterd(
            keys=keys,
            patch_size=(None, None, window_width),  # dynamic first two dimensions
            start_pos=(0, 0, 0)
        )
        if window_width==1:
            patch_transform = Compose(
                [
                    SqueezeDimd(keys=keys, dim=-1),  # squeeze the last dim
                ]
            )
        else:
            patch_transform = None
            
        # for training
        train_patch_ds = monai.data.GridPatchDataset(
            data=train_volume_ds, patch_iter=patch_func, transform=patch_transform, with_coordinates=False)
        train_loader = DataLoader(
            train_patch_ds,
            batch_size=train_batch_size,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )

        # for validation
        if self.configs.model_name=='ddpm':
            val_patch_ds = monai.data.GridPatchDataset(
            data=val_volume_ds, patch_iter=patch_func, transform=patch_transform, with_coordinates=False)
            val_loader = DataLoader(
                val_patch_ds, #val_volume_ds, 
                num_workers=0, 
                batch_size=val_batch_size,
                pin_memory=torch.cuda.is_available())
        else:
            val_loader = DataLoader(
                val_volume_ds, 
                num_workers=0, 
                batch_size=val_batch_size,
                pin_memory=torch.cuda.is_available())
        
        self.saved_name_train=self.paths["saved_name_train"]
        self.saved_name_val=self.paths["saved_name_val"]

        self.train_ds=train_ds
        self.val_ds=val_ds
        self.train_volume_ds=train_volume_ds
        self.val_volume_ds=val_volume_ds
        self.train_patch_ds=train_patch_ds
        
        self.train_batch_size=train_batch_size
        self.val_batch_size=val_batch_size

        self.train_crop_ds=train_crop_ds
        self.val_crop_ds=val_crop_ds
        self.train_loader=train_loader
        self.val_loader=val_loader
        self.train_transforms=train_transforms
        self.val_transforms=val_transforms

    def get_transforms(self, configs, mode='train'):
        normalize=configs.dataset.normalize
        pad=configs.dataset.pad
        resized_size=configs.dataset.resized_size
        WINDOW_WIDTH=configs.dataset.WINDOW_WIDTH
        WINDOW_LEVEL=configs.dataset.WINDOW_LEVEL
        prob=configs.dataset.augmentationProb
        background=configs.dataset.background
        indicator_A=configs.dataset.indicator_A
        indicator_B=configs.dataset.indicator_B
        load_masks=self.configs.dataset.load_masks
        transform_list=[]
        min, max=WINDOW_LEVEL-(WINDOW_WIDTH/2), WINDOW_LEVEL+(WINDOW_WIDTH/2)
        #transform_list.append(ThresholdIntensityd(keys=[indicator_B], threshold=min, above=True, cval=background))
        #transform_list.append(ThresholdIntensityd(keys=[indicator_B], threshold=max, above=False, cval=-1000))
        # filter the source images
        # transform_list.append(ThresholdIntensityd(keys=[indicator_A], threshold=configs.dataset.MRImax, above=False, cval=0))
        if normalize=='zscore':
            transform_list.append(NormalizeIntensityd(keys=[indicator_A, indicator_B], nonzero=False, channel_wise=True))
            print('zscore normalization')
        elif normalize=='minmax':
            transform_list.append(ScaleIntensityd(keys=[indicator_A, indicator_B], minv=-1.0, maxv=1.0))
            print('minmax normalization')

        elif normalize=='scale4000':
            transform_list.append(ScaleIntensityd(keys=[indicator_A], minv=-1, maxv=1))
            transform_list.append(ScaleIntensityd(keys=[indicator_B], minv=0))
            transform_list.append(ScaleIntensityd(keys=[indicator_B], factor=-0.99975)) # x=x(1+factor)
            print('scale1000 normalization')

        elif normalize=='scale1000':
            transform_list.append(ScaleIntensityd(keys=[indicator_A], minv=0, maxv=1))
            transform_list.append(ScaleIntensityd(keys=[indicator_B], minv=0))
            transform_list.append(ScaleIntensityd(keys=[indicator_B], factor=-0.999)) 
            print('scale1000 normalization')

        elif normalize=='inputonlyzscore':
            transform_list.append(NormalizeIntensityd(keys=[indicator_A], nonzero=False, channel_wise=True))
            print('only normalize input MRI images')

        elif normalize=='inputonlyminmax':
            transform_list.append(ScaleIntensityd(keys=[indicator_A], minv=configs.dataset.normmin, maxv=configs.dataset.normmax))
            print('only normalize input MRI images')
        
        elif normalize=='none' or normalize=='nonorm':
            print('no normalization')

        spaceXY=0
        if spaceXY>0:
            transform_list.append(Spacingd(keys=[indicator_A], pixdim=(spaceXY, spaceXY, 2.5), mode="bilinear")) # 
            transform_list.append(Spacingd(keys=[indicator_B, "mask"] if load_masks else [indicator_B], 
                                           pixdim=(spaceXY, spaceXY , 2.5), mode="bilinear"))
        
        transform_list.append(ResizeWithPadOrCropd(keys=[indicator_A, indicator_B,"mask"] if load_masks else [indicator_A, indicator_B], 
                                                   spatial_size=resized_size,mode=pad))
                
        if configs.dataset.rotate:
            transform_list.append(Rotate90d(keys=[indicator_A, indicator_B, "mask"] if load_masks else [indicator_A, indicator_B], k=3))

        if mode == 'train':
            from monai.transforms import (
                # data augmentation
                RandRotated,
                RandZoomd,
                RandBiasFieldd,
                RandAffined,
                RandGridDistortiond,
                RandGridPatchd,
                RandShiftIntensityd,
                RandGibbsNoised,
                RandAdjustContrastd,
                RandGaussianSmoothd,
                RandGaussianSharpend,
                RandGaussianNoised,
            )
            Aug=True
            if Aug:
                transform_list.append(RandRotated(keys=[indicator_A, indicator_B, "mask"] if load_masks else [indicator_A, indicator_B],
                                                  range_x = 0.1, range_y = 0.1, range_z = 0.1, 
                                                  prob=prob, padding_mode="border", keep_size=True))
                transform_list.append(RandZoomd(keys=[indicator_A, indicator_B, "mask"] if load_masks else [indicator_A, indicator_B], 
                                                prob=prob, min_zoom=0.9, max_zoom=1.3,padding_mode= "minimum" ,keep_size=True))
                transform_list.append(RandAffined(keys=[indicator_A, indicator_B], padding_mode="border" , prob=prob))
                #transform_list.append(Rand3DElasticd(keys=[indicator_A, indicator_B], prob=prob, sigma_range=(5, 8), magnitude_range=(100, 200), spatial_size=None, mode='bilinear'))
            intensityAug=False
            if intensityAug:
                print('intensity data augmentation is used')
                transform_list.append(RandBiasFieldd(keys=[indicator_A], degree=3, coeff_range=(0.0, 0.1), prob=prob)) # only apply to MRI images
                transform_list.append(RandGaussianNoised(keys=[indicator_A], prob=prob, mean=0.0, std=0.01))
                transform_list.append(RandAdjustContrastd(keys=[indicator_A], prob=prob, gamma=(0.5, 1.5)))
                transform_list.append(RandShiftIntensityd(keys=[indicator_A], prob=prob, offsets=20))
                transform_list.append(RandGaussianSharpend(keys=[indicator_A], alpha=(0.2, 0.8), prob=prob))
            
        #transform_list.append(Rotate90d(keys=[indicator_A, indicator_B], k=3))
        #transform_list.append(DivisiblePadd(keys=[indicator_A, indicator_B], k=div_size, mode="minimum"))
        #transform_list.append(Identityd(keys=[indicator_A, indicator_B]))  # do nothing for the no norm case
        train_transforms = Compose(transform_list)
        return train_transforms

    def finalcheck(self,ifsave=False,ifcheck=False,iftest_volumes_pixdim=False):
        if ifsave:
            self.save_volumes(self.train_ds, self.val_ds, self.saved_name_train, self.saved_name_val)
        if iftest_volumes_pixdim:
            self.test_volumes_pixdim(self.train_volume_ds)
        if ifcheck:
            self.check_volumes(self.train_ds, self.train_volume_ds, self.val_volume_ds, self.val_ds)
            self.check_batch_data(self.train_loader,self.val_loader,
                                  self.train_patch_ds,self.val_volume_ds,
                                  self.train_batch_size,self.val_batch_size)
    
    def test_volumes_pixdim(self, train_volume_ds):
        train_loader = DataLoader(train_volume_ds, batch_size=1)
        for step, data in enumerate(train_loader):
            mr_data=data[self.indicator_A]
            ct_data=data[self.indicator_B]
            
            print(f"source image shape: {mr_data.shape}")
            print(f"source image affine:\n{mr_data.meta['affine']}")
            print(f"source image pixdim:\n{mr_data.pixdim}")

            # target image information
            print(f"target image shape: {ct_data.shape}")
            print(f"target image affine:\n{ct_data.meta['affine']}")
            print(f"target image pixdim:\n{ct_data.pixdim}")

    def check_volumes(self, train_ds, train_volume_ds, val_volume_ds, val_ds):
        # use batch_size=1 to check the volumes because the input volumes have different shapes
        train_loader = DataLoader(train_volume_ds, batch_size=1)
        val_loader = DataLoader(val_volume_ds, batch_size=1)
        train_iterator = iter(train_loader)
        val_iterator = iter(val_loader)
        print('check training data:')
        idx=0
        for idx in range(len(train_loader)):
            try:
                train_check_data = next(train_iterator)
                ds_idx = idx * 1
                current_item = train_ds[ds_idx]
                current_name = os.path.basename(os.path.dirname(current_item['image']))
                print(idx, current_name, 'image:', train_check_data['image'].shape, 'label:', train_check_data['label'].shape)
            except:
                ds_idx = idx * 1
                current_item = train_ds[ds_idx]
                current_name = os.path.basename(os.path.dirname(current_item['image']))
                print('check data error! Check the input data:',current_name)
        print("checked all training data.")

        print('check validation data:')
        idx=0
        for idx in range(len(val_loader)):
            try:
                val_check_data = next(val_iterator)
                ds_idx = idx * 1
                current_item = val_ds[ds_idx]
                current_name = os.path.basename(os.path.dirname(current_item['image']))
                print(idx, current_name, 'image:', val_check_data['image'].shape, 'label:', val_check_data['label'].shape)
            except:
                ds_idx = idx * 1
                current_item = val_ds[ds_idx]
                current_name = os.path.basename(os.path.dirname(current_item['image']))
                print('check data error! Check the input data:',current_name)
        print("checked all validation data.")

    def save_volumes(self, train_ds, val_ds, saved_name_train, saved_name_val):
        shape_list_train=[]
        shape_list_val=[]
        # use the function of saving information before
        for sample in train_ds:
            name = os.path.basename(os.path.dirname(sample[self.indicator_A]))
            shape_list_train.append({'patient': name})
        for sample in val_ds:
            name = os.path.basename(os.path.dirname(sample[self.indicator_A]))
            shape_list_val.append({'patient': name})
        np.savetxt(saved_name_train,shape_list_train,delimiter=',',fmt='%s',newline='\n') # f means format, r means raw string
        np.savetxt(saved_name_val,shape_list_val,delimiter=',',fmt='%s',newline='\n') # f means format, r means raw string

    def check_batch_data(self, train_loader,val_loader,train_patch_ds,val_volume_ds,train_batch_size,val_batch_size):
        for idx, train_check_data in enumerate(train_loader):
            ds_idx = idx * train_batch_size
            current_item = train_patch_ds[ds_idx]
            print('check train data:')
            print(current_item, 'image:', train_check_data['image'].shape, 'label:', train_check_data['label'].shape)
        
        for idx, val_check_data in enumerate(val_loader):
            ds_idx = idx * val_batch_size
            current_item = val_volume_ds[ds_idx]
            print('check val data:')
            print(current_item, 'image:', val_check_data['image'].shape, 'label:', val_check_data['label'].shape)

    def len_patchloader(self, train_volume_ds,train_batch_size):
        slice_number=sum(train_volume_ds[i][self.indicator_A].shape[-1] for i in range(len(train_volume_ds)))
        print('total slices in training set:',slice_number)

        import math
        batch_number=sum(math.ceil(train_volume_ds[i][self.indicator_A].shape[-1]/train_batch_size) for i in range(len(train_volume_ds)))
        print('total batches in training set:',batch_number)
        return slice_number,batch_number
    
    def get_length(self, dataset, patch_batch_size):
        loader=DataLoader(dataset, batch_size=1)
        iterator = iter(loader)
        sum_nslices=0
        for idx in range(len(loader)):
            check_data = next(iterator)
            nslices=check_data[self.indicator_A].shape[-1]
            sum_nslices+=nslices
        if sum_nslices%patch_batch_size==0:
            return sum_nslices//patch_batch_size
        else:
            return sum_nslices//patch_batch_size+1
