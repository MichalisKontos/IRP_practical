from diffusers import StableDiffusionPipeline
import torch
import os

def callback(iter, t, latents):
    # convert latents to image
    with torch.no_grad():
        latents = 1 / 0.18215 * latents
        image = pipe.vae.decode(latents).sample

        image = (image / 2 + 0.5).clamp(0, 1)

        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()

        # convert to PIL Images
        image = pipe.numpy_to_pil(image)

        # do something with the Images
        for i, img in enumerate(image):
            img.save(os.path.join(output_dir, f"{iter}.png"))

prompt = "a photo of an astronaut riding a horse on mars"
output_dir_1_4 = "IRP_practical/1_4_iter"
output_dir_1_5 = "IRP_practical/1_5_iter"
output_dir_2 = "IRP_practical/2_iter"
output_dirs = [output_dir_1_4, output_dir_1_5, output_dir_2]

# -------------- Stable diffusion v1.4 -------------- #
model_id_1_4 = "CompVis/stable-diffusion-v1-4"
pipe_1_4 = StableDiffusionPipeline.from_pretrained(model_id_1_4, torch_dtype=torch.float16)
pipe_1_4 = pipe_1_4.to("mps")

# -------------- Stable diffusion v1.5 -------------- #
model_id_1_5 = "runwayml/stable-diffusion-v1-5"
pipe_1_5 = StableDiffusionPipeline.from_pretrained(model_id_1_5, torch_dtype=torch.float16)
pipe_1_5 = pipe_1_5.to("mps")

# -------------- Stable diffusion v2 -------------- #
model_id_2 = "stabilityai/stable-diffusion-2"
pipe_2 = StableDiffusionPipeline.from_pretrained(model_id_2, torch_dtype=torch.float16)
pipe_2 = pipe_2.to("mps")

pipes = [pipe_1_4, pipe_1_5, pipe_2]
model_versions = ["1_4", "1_5", "2"]

# generate image (note the `callback` and `callback_steps` argument)
# image_1_4 = pipe_1_4(prompt, callback=callback, callback_steps=5)
# for step in range(1, 51):  # 50 steps
for i in range(len(model_versions)):
    pipe = pipes[i]
    output_dir = output_dirs[i]
    model_version = model_versions[i]
    image = pipe(prompt, callback=callback, callback_steps=5)