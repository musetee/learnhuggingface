from diffusers import DiffusionPipeline
from diffusers import DPMSolverMultistepScheduler
from diffusers import AutoencoderKL
import torch
def get_inputs(prompt, batch_size=1):
    generator = [torch.Generator("cuda").manual_seed(i) for i in range(batch_size)]
    prompts = batch_size * [prompt]
    num_inference_steps = 20

    return {"prompt": prompts, "generator": generator, "num_inference_steps": num_inference_steps}

from PIL import Image

def image_grid(imgs, rows=2, cols=2):
    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid

if __name__ == "__main__":
    model_id = "runwayml/stable-diffusion-v1-5"
    pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to("cuda")
    use_better_vae, attention_slicing_enabled = False, False
    if use_better_vae:
        vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=torch.float16).to("cuda")
        pipe.vae = vae
    if attention_slicing_enabled:
        pipe.enable_attention_slicing() # enable attention slicing for saving memory
    
    prompt = "photo of a sexy girl, wearing pantyhose."
    prompt += ", make up, serious eyes"
    prompt += " 50mm portrait photography, hard rim lighting photography--beta --ar 2:3  --beta --upbeta"
    batch_size=6
    images = pipe(**get_inputs(prompt=prompt,batch_size=batch_size)).images
    grid=image_grid(images, rows=2, cols=int(batch_size/2))
    grid.save(f"./results/ch_sexy_girl_DPMSolver_float16_batch{batch_size}.png")
    
    
    """ single mode
    generator = torch.Generator("cuda").manual_seed(0)
    num_inference_steps = 50
    image = pipe(prompt, generator=generator, num_inference_steps=num_inference_steps).images[0]  
    image.save(f"./results/old_warrior_chief_DPMSolver_step{num_inference_steps}.png")
    """