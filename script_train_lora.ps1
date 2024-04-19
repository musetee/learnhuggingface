# PowerShell script equivalent to the provided Unix shell script

# Set environment variables
$Env:MODEL_NAME="runwayml/stable-diffusion-v1-5"
$Env:TRAIN_DIR="/datafolder/train"
$Env:OUTPUT_DIR="/results/ckpt"

# Execute the training command
train_dreambooth_lora.py `
  --pretrained_model_name_or_path=$Env:MODEL_NAME `
  --train_data_dir=$Env:TRAIN_DIR `
  -output_dir=$OUTPUT_DIR `
  --instance_prompt="a photo of sks dog" `
  --resolution=512 `
  --train_batch_size=1 `
  --gradient_accumulation_steps=1 `
  --checkpointing_steps=100 `
  --learning_rate=1e-4 `
  --report_to="wandb" `
  --lr_scheduler="constant" `
  --lr_warmup_steps=0 `
  --max_train_steps=500 `
  --validation_prompt="A photo of sks dog in a bucket" `
  --validation_epochs=50 `
  --seed="0" `
  --push_to_hub