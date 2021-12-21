import sys
import os

import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.getcwd())
from utils import test, cos_similarity, ROISelector


@test
def test_cos_similarity():
    x = np.random.randn(10)
    y = np.random.randn(10)
    sim1 = cos_similarity(x, y)
    sim2 = cos_similarity(x, x)
    print(
        f'sim with self ({sim2:.4f}) should be 1, sim with another random vector ({sim1:.4f})')


@test
def test_ROISelector():
    path = 'database/sample.jpg'
    a = ROISelector(path)
    plt.show()
    img = a.img
    img_cropped = a.cropped_img

    print(f'img: {img.shape} img_cropped: {img_cropped.shape}')

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.axis('off')
    plt.imshow(img)

    plt.subplot(1, 2, 2)
    plt.axis('off')
    plt.imshow(img_cropped)
    
    plt.show()


if __name__ == '__main__':
    test_ROISelector()
