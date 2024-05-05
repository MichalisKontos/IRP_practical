from diffusers import StableDiffusionPipeline
import torch
import os
import time

prompt = "a photo of an astronaut riding a horse on mars"
output_dir_1_4 = "/Users/mk7n23/Desktop/University of Southampton/COMP6228 - IRP/Practical/1_4"
output_dir_1_5 = "/Users/mk7n23/Desktop/University of Southampton/COMP6228 - IRP/Practical/1_5"
output_dir_2 = "/Users/mk7n23/Desktop/University of Southampton/COMP6228 - IRP/Practical/2"
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
model_version = ["1_4", "1_5", "2"]


def generate_and_measure_time(prompt, output_dir, pipe, model_version):
    start_time = time.time()
    image = pipe(prompt).images[0]
    end_time = time.time()
    image.save(os.path.join(output_dir, f"{model_version}.png"))
    return end_time - start_time


# Number of images to generate
num_images = 20

# List to store time taken for each image generation
generation_times_1_4 = []
generation_times_1_5 = []
generation_times_2 = []

# Generate and measure time for each image
for i in range(len(pipes)):
    for j in range(num_images):
        generation_time = generate_and_measure_time(prompt, output_dirs[i], pipes[i], f"v{model_version[i]}_n{j}")
        if i == 0:
            generation_times_1_4.append(generation_time)
        elif i == 1:
            generation_times_1_5.append(generation_time)
        else:
            generation_times_2.append(generation_time)

# Calculate average generation time
average_generation_time_1_4 = sum(generation_times_1_4) / num_images
average_generation_time_1_5 = sum(generation_times_1_5) / num_images
average_generation_time_2 = sum(generation_times_2) / num_images

gen = [generation_times_1_4, generation_times_1_5, generation_times_2]
avgs = [average_generation_time_1_4, average_generation_time_1_5, average_generation_time_2]

for i in range(len(model_version)):
    print(f"Average generation time per image for version {model_version[i]} (for {num_images} images): ", avgs[i])
    print(f"Times v{model_version[i]}: ", gen[i])