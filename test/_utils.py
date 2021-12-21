import sys
import os

import numpy as np
import matplotlib.pyplot as plt

sys.path.append(sys.path.append(os.getcwd()))
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
    img = a.output_image()
    print(f'img: {img.shape}')


if __name__ == '__main__':
    test_ROISelector()
