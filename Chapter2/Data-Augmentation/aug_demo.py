import imgaug as ia
from imgaug import augmenters as iaa
import numpy as np

# This seed can be changed - random seed
ia.seed(1)

# Example batch of 100 images
images = np.array(
    [ia.quokka(size=(64, 64)) for _ in range(100)],
    dtype=np.uint8
)

# Create the transformer function by specifying the different augmentations
seq = iaa.Sequential([
    # Horizontal Flips
    iaa.Fliplr(0.5), 

    # Random Crops
    iaa.Crop(percent=(0, 0.1)), 

    # Gaussian blur for 50% of the images
    iaa.Sometimes(0.5,
        iaa.GaussianBlur(sigma=(0, 0.5))
    ),
    # Strengthen or weaken the contrast in each image.
    iaa.ContrastNormalization((0.75, 1.5)),

    # Add gaussian noise.
    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),

    # Make some images brighter and some darker.
    iaa.Multiply((0.8, 1.2), per_channel=0.2),

    # Apply affine transformations to each image.
    iaa.Affine(
        scale={"x": (0.5, 1.5), "y": (0.5, 1.5)},
        translate_percent={"x": (-0.5, 0.5), "y": (-0.5, 0.5)},
        rotate=(-10, 10),
        shear=(-10, 10)
    )
], 
# apply augmenters in random order
random_order=True) 

# This should display a random set of augmentations in a window
images_aug = seq.augment_images(images)
seq.show_grid(images[0], cols=8, rows=8)

