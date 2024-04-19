# PowerShell script equivalent to the provided Unix shell script

# Set environment variables
$Env:MODEL_NAME="CompVis/stable-diffusion-v1-4"
$Env:TRAIN_DIR="datafolder"
$Env:OUTPUT_DIR="results"

# Execute the training command
python train_text_to_image.py `
  --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" `
  --train_data_dir="datafolder" `
  --use_ema `
  --resolution=512 --center_crop --random_flip `
  --train_batch_size=1 `
  --gradient_accumulation_steps=4 `
  --gradient_checkpointing `
  --mixed_precision="fp16" `
  --max_train_steps=15000 ` 
  --learning_rate=1e-05 `
  --max_grad_norm=1 `
  --lr_scheduler="constant" --lr_warmup_steps=0 `
  --output_dir="results" `
  --caption_column="additional_feature"