import os
from PIL import Image, ImageOps
import re

def sort_key(path):
    # Extract numeric part from the filename
    match = re.search(r'\d+', os.path.basename(path))
    if match:
        return int(match.group())
    else:
        return 0  # If no numeric part found, consider it as 0

def concat_images(image_paths, size, shape=None):
    # Open images and resize them
    width, height = size
    images = map(Image.open, image_paths)
    images = [ImageOps.fit(image, size, Image.LANCZOS) for image in images]
    
    # Create canvas for the final image with total size
    shape = shape if shape else (1, len(images))
    image_size = (width * shape[1], height * shape[0])
    image = Image.new('RGB', image_size)
    
    # Paste images into final image
    for row in range(shape[0]):
        for col in range(shape[1]):
            offset = width * col, height * row
            idx = row * shape[1] + col
            image.paste(images[idx], offset)
    
    return image

folders = ["1_4_iter", "1_5_iter", "2_iter"]
names = ["1_4", "1_5", "2"]

for i in range(3):
    # Get list of image paths
    folder = f"IRP_practical/{folders[i]}/"
    image_paths = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.png')]

    image_paths.sort(key=sort_key)
    
    # Create and save image grid
    image = concat_images(image_paths, (512, 512), (2, 5))
    image.save(os.path.join("IRP_practical/concat",f'con_{names[i]}.png'))