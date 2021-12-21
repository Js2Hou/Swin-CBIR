# Swin-transformer based CBIR

This repository contains a CBIR(content-based image retrieval) system. Here we use [Swin-transformer](https://github.com/microsoft/Swin-Transformer) to extract query image's feature, and retrieve similar ones from image database. Notably, our program achieves intelligent user interaction, including selecting an image by opening explorer dialog and cropping interested region by drafting mouse.

![avatar](database/image.png)

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

2. run `./script/generate_DB.sh` in linux machine to extract features of all images and package them into `DB.npz`.

3. run `main.py`, open an image and select interested region, then program will find similar images in database automatically!

## Acknowledgments

Part of code in this repository are copied from [Swin-transformer](https://github.com/microsoft/Swin-Transformer), thank the authors for their exquiste code.