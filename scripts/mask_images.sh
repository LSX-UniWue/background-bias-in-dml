#!/bin/bash
# Masks the dataset by removing the background
# Run from root folder with: bash scripts/mask_images.sh

# Cars196
# =======
rembg p -om "data/cars196/car_ims" "data/cars196/car_ims_masked"

# Cub200
# ======
rembg p -om "data/cub2011/CUB_200_2011/images/" "data/cub2011/images_masked"

# Stanford Online Products
# ========================
# Since this takes very long to process, we go through each folder separately.
# If you need to stop processing, you can later start the script for the remaining folders.
rembg p -om "data/Stanford_Online_Products/" "data/SOP_masked"