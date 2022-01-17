import math
import argparse

import cv2.cv2 as cv
import numpy as np
import torch
import matplotlib.pyplot as plt
from tkinter import filedialog

from config import get_config
from database import DB
from utils import ROISelector, cos_similarity, get_pretrained_model, set_subplot_border
from models import build_model


def parse_option():
    parser = argparse.ArgumentParser(
        'image retrieve script', add_help=False)
    parser.add_argument('--cfg', default='configs/swin_tiny_patch4_window7_224.yaml', type=str,
                        help='path to config file', )

    parser.add_argument('--resume', default='checkpoints/swin_tiny_patch4_window7_224.pth',
                        help='resume from checkpoint')

    # todo: useless, to be deleted
    parser.add_argument("--local_rank", type=int,
                        help='local rank for DistributedDataParallel')
    parser.add_argument('--batch-size', type=int,
                        help="batch size for single GPU")
    parser.add_argument('--data-path', default='database/data',
                        type=str, help='path to dataset')

    args, unparsed = parser.parse_known_args()

    config = get_config(args)

    return args, config


def extract_feat(config, model, img):
    """
    Extract feature of input image by `model`.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    img = cv.resize(img, (config.DATA.IMG_SIZE, config.DATA.IMG_SIZE))
    img = img.transpose(2, 0, 1)
    img = img[np.newaxis]
    img = torch.from_numpy(img)
    img.to(device)

    with torch.no_grad():
        output = model(img)
    feat = output.cpu().numpy()
    return feat


def retrive(config, feat, n_return=5):
    """
    Retrive similar image in database based on image feature.

    Parameters:
        feat (np.array): feature of retrive target
        n_return (int): number of most similar images returned

    Returns:
        similaritys_sorted (list)
        imgs (list of np.ndarray): retrived results
    """
    db = DB(config.DATA.DATABASE_PATH)
    db_feat, db_path = db.database()
    n = len(db)

    similaritys = []
    for i in range(n):
        similaritys.append(cos_similarity(db_feat[i], feat))
    similaritys = np.array(similaritys)
    index = np.argsort(similaritys)[::-1]
    index = index[:n_return]
    similaritys_sorted = similaritys[index]

    # return top `n_return` similar images
    db_path_sorted = db_path[index]
    imgs = []
    for path_ in db_path_sorted:
        imgs.append(plt.imread(path_))
    return similaritys_sorted, imgs


def show_images(ori_img, cropped_img, retrived_imgs, similarity, col=None):
    """
    Plot original image, cropped image, and retrived images in `col` columns.
    """
    assert len(retrived_imgs) == len(similarity), \
        f'Size of images {len(retrived_imgs)} should be same with size of sims {len(similarity)}.'
    n = len(retrived_imgs)
    col = int(col) if col else 5
    row = math.ceil(n / col) + 1
    h, w, _ = ori_img.shape
    h = w = min(h, w)

    # plot to show
    plt.figure()
    plt.subplots_adjust(wspace=.2, hspace=.2)
    for i in range(row):
        for j in range(col):
            idx = i * col + j
            ax = plt.subplot(row, col, idx + 1)

            if idx == 0:
                title_, img_ = 'raw img', ori_img
                set_subplot_border(ax, 'green', 4)
            elif idx == 1:
                title_, img_ = 'cropped img', cropped_img
                set_subplot_border(ax, 'red', 4)
            elif idx < col:
                plt.axis('off')
                continue
            else:
                title_, img_ = f'{similarity[idx - col]:.4f}', retrived_imgs[idx - col]
                set_subplot_border(ax, 'blue', 4)
            
            plt.title(title_)
            plt.xticks([])
            plt.yticks([])
            img_ = cv.resize(img_, (h, w))
            plt.imshow(img_)
    plt.show()


def main():
    _, config = parse_option()

    # 1. open image and select interested region
    path = filedialog.askopenfilename()
    # path = r'database/sample.jpg'
    roisor = ROISelector(path)
    plt.show()
    ori_img = roisor.img
    roi = roisor.cropped_img

    # 2. extract feature of roi
    model = get_pretrained_model(config)
    feat = extract_feat(config, model, roi)

    # 3. retrive in database
    similarity, retrived_imgs = retrive(config, feat, n_return=10)

    # 4. display
    show_images(ori_img, roi, retrived_imgs, similarity, col=5)


if __name__ == '__main__':
    main()
