# IRP_practical
Individual Research Project: Work with Diffusion models and test different versions of Stable Diffusion (v1.4, v1.5, v2)


## gen.py

Use for generating images using a prompt. It accesses the Stable Diffusion models with versions 1.4, 1.5 and 2. 
The prompt is set to "a photo of an astronaut riding a horse on mars" and can be changed within the file.
The generated images are placed in folders 1_4, 1_5 and 2 respectively.


## iteration.py

Use for generating images during the denoising process of Stable Diffusion (for each version).
Placed images in folders 1_4_iter, 1_5_iter and 2_iter respectively.

## concat_img.py

Used to concatenate the denoising images from iteration.py into a single image and grid.
