from transformers import CLIPProcessor, CLIPModel

import torch
from PIL import Image
from dataset_med import get_data_loader_folder,write_synthrad_csv

import torch.nn as nn


def main():
    # Load the model and processor
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    med_info_pairs = [
            {"root": r'D:\Projects\data\Task1\pelvis', "modality": "ct", "tissue": "pelvis"},
            {"root": r'D:\Projects\data\Task1\brain', "modality": "mr", "tissue": "brain"}
        ]
    create_new_csv = False
    if create_new_csv:
        write_synthrad_csv(med_info_pairs)

    csv_file = "test.csv"
    batch_size = 8
    new_size = 512
    height = 512
    width = 512
    num_workers = None
    load_patient_number = 1
    train_loader, val_loader = get_data_loader_folder("test.csv", batch_size, height, width, False, num_workers, load_patient_number)

    import torch

    # Assume 'dataloader' is your DataLoader instance loaded with your custom dataset
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.train()
    loss_img = nn.CrossEntropyLoss()
    loss_txt = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5,betas=(0.9,0.98),eps=1e-6,weight_decay=0.2)

    num_epochs = 1
    total_loss = 0
    for epoch in range(num_epochs):  # num_epochs is the number of epochs you want to train for
        for batch_idx, batch in enumerate(train_loader):
            # Get images and texts from the batch
            images = batch['img'].to(device)
            captions = batch['text']

            # Use the processor to prepare the inputs
            inputs = processor(text=captions, images=images, return_tensors="pt", padding=True, truncation=True).to(device)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Forward pass
            outputs = model(**inputs)
            loss = outputs.loss
            
            print(loss)
            
            # Compute loss
            '''
            logits_per_image = outputs.logits_per_image
            logits_per_text = outputs.logits_per_text
            ground_truth = torch.arange(len(images),dtype=torch.long,device=device) # torch.arange(8) = [0,1,2,3,4,5,6,7]
            total_loss = (loss_img(logits_per_image,ground_truth) + loss_txt(logits_per_text,ground_truth))/2
            '''

            # Backward pass
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            print(f'Epoch {epoch}, Batch {batch_idx}, Loss {loss.item()}')
        
        # Save the model after each epoch
        torch.save(model.state_dict(), f"./log/model_{epoch}.pt")

    # validate
    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            # Get images and texts from the batch
            images = batch['img'].to(device)
            captions = batch['text']

            # Use the processor to prepare the inputs
            inputs = processor(text=captions, images=images, return_tensors="pt", padding=True, truncation=True).to(device)

            # Forward pass
            ClipOutputs= model(**inputs, return_loss=True)
            logits_per_image = ClipOutputs.logits_per_image
            logits_per_text = ClipOutputs.logits_per_text
            # Compute loss

            ground_truth = torch.arange(len(images),dtype=torch.long,device=device) # torch.arange(8) = [0,1,2,3,4,5,6,7]
            total_loss = (loss_img(logits_per_image,ground_truth) + loss_txt(logits_per_text,ground_truth))/2
            print(f'Validation Batch {batch_idx}, Loss {total_loss.item()}')

if __name__ == '__main__':
    main()