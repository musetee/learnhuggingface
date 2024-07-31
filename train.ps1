accelerate launch --mixed_precision="fp16" train_synthrad_pix2pix.py --dataset_name="synthrad" --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5"   --output_dir="./logs/synthrad-pix2pix" --validation_epochs=1 --validation_prompt "a pelvis CT image" --num_validation_images=4 --enable_xformers_memory_efficient_attention  --resolution=256 --random_flip --train_batch_size=4 --gradient_accumulation_steps=4 --gradient_checkpointing --num_train_epochs=100 --checkpointing_steps=500 --checkpoints_total_limit=5 --learning_rate=5e-05 --max_grad_norm=1 --lr_warmup_steps=0 --conditioning_dropout_prob=0.05 --mixed_precision=fp16 --seed=42
