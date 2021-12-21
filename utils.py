from functools import wraps

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from tkinter import filedialog

from models import build_model


def test(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        print(f"{'=' * 20}> Function Test ({func.__name__}) <{'=' * 20}")
        func(*args, **kwargs)
    return wrapper


def get_pretrained_model(config):
    
    model = build_model(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print(f"==============> Resuming form {config.MODEL.RESUME}....................")
    if config.MODEL.RESUME.startswith('https'):
        checkpoint = torch.hub.load_state_dict_from_url(
            config.MODEL.RESUME, map_location='cpu', check_hash=True)
    else:
        checkpoint = torch.load(config.MODEL.RESUME, map_location='cpu')

    model.load_state_dict(checkpoint['model'], strict=False)
    print(f"=> loaded successfully '{config.MODEL.RESUME}'")

    del checkpoint
    torch.cuda.empty_cache()

    model.eval()
    return model


def cos_similarity(x, y):
    """
    余弦相似度，范围在[0, 1]之间，越大相似程度越高
    """
    similarity = np.dot(x, y.T) / (np.linalg.norm(x) * np.linalg.norm(y))
    return similarity.flatten()[0]


class ROISelector(object):
    """实现功能：框选图像感兴趣部分并返回

    Example:

    ```python
    import matplotlib.pyplot as plt
    
    
    img_path = 'img.jpg'
    roisor = ROISelector(img_path)
    plt.show()
    ```
    """

    def __init__(self, img_path):
        self.img_path = img_path
        self.img = plt.imread(img_path)

        self.init_image()
        self.init_rectangle()

        self.ax = plt.gca()
        self.x0 = self.x1 = None
        self.y0 = self.y1 = None
        self.ax.add_patch(self.rect)
        self.ax.figure.canvas.mpl_connect('button_press_event', self.on_press)
        self.ax.figure.canvas.mpl_connect(
            'button_release_event', self.on_release)
        self.ax.figure.canvas.mpl_connect(
            'motion_notify_event', self.on_mouse_move)

    def init_image(self):
        plt.axis('off')
        plt.imshow(self.img)

    def init_rectangle(self):
        self.rect = Rectangle((0, 0), 1, 1, color='red',
                              linewidth=2, fill=False)

    def output_image(self):
        return self.cropped_img

    def on_mouse_move(self, event):
        if not self.x0:
            return
        x0, y0 = self.x0, self.y0
        x1, y1 = event.xdata, event.ydata

        self.rect.set_width(x1 - x0)
        self.rect.set_height(y1 - y0)
        self.rect.set_xy((x0, y0))
        self.rect.set_linestyle('dashed')
        self.ax.figure.canvas.draw()

    def on_press(self, event):
        print('press')
        self.x0 = event.xdata
        self.y0 = event.ydata

    def on_release(self, event):
        print('release')
        x0, y0 = self.x0, self.y0
        x1, y1 = event.xdata, event.ydata
        self.x1, self.y1 = x1, y1

        self.rect.set_width(x1 - x0)
        self.rect.set_height(y1 - y0)
        self.rect.set_xy((x0, y0))
        self.rect.set_linestyle('dashed')
        self.ax.figure.canvas.draw()
        self.cropped_img = self.img[int(y0) : int(y1), int(x0) : int(x1), :]
        self.x0 = self.y0 = None
