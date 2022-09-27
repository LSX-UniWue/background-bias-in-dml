<div align="center">

# On Background Bias in Deep Metric Learning

[![Conference](http://img.shields.io/badge/ICMV-2022-F77A4B.svg)](http://icmv.org/)

</div>

## Description

Code for our ICMV 2022 paper "On Background Bias in Deep Metric Learning" (Konstantin Kobs and Andreas Hotho).

Abstract:
>Deep Metric Learning trains a neural network to map input images to a lower-dimensional embedding space such that similar images are closer together than dissimilar images.
>When used for item retrieval, a query image is embedded using the trained model and the closest items from a database storing their respective embeddings are returned as the most similar items for the query.
>Especially in product retrieval, where a user searches for a certain product by taking a photo of it, the image background is usually not important and thus should not influence the embedding process.
>Ideally, the retrieval process always returns fitting items for the photographed object, regardless of the environment the photo was taken in.
>In this paper, we analyze the influence of the image background on Deep Metric Learning models by utilizing five common loss functions and three common datasets.
>We find that Deep Metric Learning networks are prone to so-called background bias, which can lead to a severe decrease in retrieval performance when changing the image background during inference.
>We also show that replacing the background of images during training with random background images alleviates this issue.
>Since we use an automatic background removal method to do this background replacement, no additional manual labeling work and model changes are required while inference time stays the same.
>Qualitative and quantitative analyses, for which we introduce a new evaluation metric, confirm that models trained with replaced backgrounds attend more to the main object in the image, benefitting item retrieval systems.


## How to train

### 1. Clone this repository and install dependencies
Install the dependencies given in `requirements.txt`

### 2. Put datasets into the `data/` subfolders
- `data/cars196` (Cars196)
- `data/cub200` (CUB200)
- `data/Stanford_Online_Products` (Stanford Online Products)

### 3. Generate masks of images
Use the commands in `scripts/mask_images.sh`.

### 4. Train models
Run the `train.py` using the desired experiment configs.
For example, `python3 train.py experiment=cars196/arcface/base trainer.gpus=1` trains a model with the ArcFace loss on the Cars196 dataset using one GPU.
For each dataset (`cars196`/`cub200`/`stanford_online_products`), you can choose five loss functions (`contrastive`/`triplet`/`multi_similarity`/`arcface`/`normalized_softmax`) and whether the model should be trained with (`bgaugment`) or without (`base`) BGAugment, our proposed method to alleviate background bias in Deep Metric Learning (DML) models.

The background images used for our test setting and BGAugment are stored in `data/unsplash_backgrounds`.

## How to generate results table (Table 1 in the paper)

After training, you can use the `notebooks/results.ipynb` notebook to generate the results table.

## Analysis

In the paper, we perform qualitative and quantitative analyses to understand the performance of the trained models.

After training all models, the `logs` folder should contain all trained models.
Update the `trained_models.py` file in the root folder to reflect the paths to the corresponding checkpoint files.

### Qualitative analysis (Figure 2 in the paper)
Run the `notebooks/qualitative_analysis.ipynb` notebook to generate some images and their corresponding attribution maps for the base and BGAugment models.

### Quantitative analysis (Table 2 in the paper)
Run `analysis.py` (see file for options) to generate attribution maps and compute their attention scores for all test images in the datasets.
This script will create the `attention_scores` folder, containing scores as computed by the scoring method that is described in Section 6 of the paper.
Then, run the `notebooks/analysis.ipynb` notebook to generate the table.

### Quality of automatically generated masks
For our test setting and analyses, we automatically created masks using `rembg`.
In order to understand the quality of those masks, we hand-labelled some images from each dataset and compared them with the generated masks.
The ground truth masks are in `notebooks/gt_masks` and you can generate the metrics reported in the paper using the `notebooks/rembg_check.ipynb` notebook.


## Citation
If you use the code provided here, please cite our paper:

```
TODO
```