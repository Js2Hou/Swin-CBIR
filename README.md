# Swin-transformer based CBIR

This repository contains a CBIR(content-based image retrieval) system. Here we use [Swin-transformer](https://github.com/microsoft/Swin-Transformer) to extract query image's feature, and retrieve similar ones from image database. Notably, our program achieves intelligent user interaction, including selecting an image by opening explorer dialog and cropping interested region by drafting mouse.

<img src="database/framework.png" width = 60%  alt="framework" div align=center />

## Structure

```python
SWIN_CBIR/
|-- checkpoints/
|
|-- database/
|   |-- data/
|   |   |-- 1.jpg
|   |   |-- 2.jpg
|   |  
|   |-- DB.npz
|   |-- index.txt
|
|-- models/
|   |-- __init__.py
|   |-- build.py
|   |-- swin_transformer.py
|
|-- scripts/
|   |-- generate_DB.sh
|
|-- test/
|
|-- config.py
|-- database.py
|-- generate_DB.py
|-- main.py
|-- requirements.txt
|-- README
```

## Getting Started

1. Prepare images database

    Just find out some images and put them into `database/data/`.
2. Download swin-transformer checkpoint at [swin_tiny_patch4_window7_224.pth](https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth), and then move it into `checkpoints`.

3. run `./script/generate_DB.sh` in linux machine to extract features of all images and package them into `DB.npz`.

4. run `main.py`, open an image and select interested region, then program will find similar images in database automatically!

**Pay Attention To**: we recommend you do step 4 on a local mechine, because it involves graphic user interface.

## Results

Here we show two image retrieval results. Two images in the first row are original image and cropped image respectively while the others are retrieval results (have been sorted by similarity).

Note: all images are resize to square for visual requirement, so there would be distorted in some of the images.

<img src="database/result1.png" width = 50%  alt="framework" div align=center ><img src="database/result2.png" width = 50%  alt="framework" div align=center />

## Acknowledgments

Part of code in this repository are copied from [Swin-transformer](https://github.com/microsoft/Swin-Transformer), thank the authors for their exquiste code.
