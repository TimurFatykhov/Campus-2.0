import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
import torchvision
import os
import glob
from PIL import Image

def load_gestures(path, size=(32, 32)):
    """
    Загружает данные с жестами рук
    
    Принимает один параметр:
    -------------------------
    - path: строка
        путь к папкам с цифрами на языке жестов (каждая
        папка должна иметь название в соответствии классу
        картинок, которые она хранит)
    
    
    Возвращает две переменные:
    -------------------------
    - X: тензор (четырех-мерная матрица) Nx3x64x64
        массив картинок
        (N - кол-во картинок)
        
    - y: вектор длинной N
        true вектор, хранящий класс каждой картинки, 
        (N - кол-во картинок)
        
    
    Пример использования:
    ---------------------
    >>> X, y = load_gestures('data')
    >>>
    """
    X = []
    y = []
    
    if not hasattr(path, '__iter__'):
        path = [path]
        
    for p in path:
        to_classes = sorted(glob.glob(os.path.join(p, '*')))
        for c in to_classes:
            print(c, 'загружен')
            to_imgs = glob.glob(os.path.join(c, '*'))
            for to_img in to_imgs:
                img = Image.open(to_img)
                width, height = img.size
                diff = height - width

                if diff < 0:
                    # width > height
                    print('w > h')
                    diff = - diff
                    half = diff // 2
                    img = img.crop((half, 0, width - diff + half, height))

                elif diff > 0:
                    # height > width
                    half = diff // 2
                    img = img.crop((0, half, width, height - diff + half))

                img = img.resize(size=size, resample=Image.BICUBIC)

                pix_arr = np.asarray(img)
                pix_arr = pix_arr - pix_arr.min()
                pix_arr = pix_arr / pix_arr.max()

                X.append(pix_arr)
                y.append(int(os.path.basename(c)))
    
    
    X = np.stack(X)
    
    return X, y


def __center_crop__(pil_img):
    width, height = pil_img.size
    diff = height - width

    if diff < 0:
        # width > height
        print('w > h')
        diff = - diff
        half = diff // 2
        pil_img = pil_img.crop((half, 0, width - diff + half, height))

    elif diff > 0:
        # height > width
        half = diff // 2
        pil_img = pil_img.crop((0, half, width, height - diff + half))
        
    return pil_img
    
    

def load_mri(path, size=(32, 32)):
    """
    Загружает изображения с МРТ и соответствующие им
    сегментационные маски желудочков
    
    Принимает один параметр:
    -------------------------
    - path: строка
        путь к папке с папками: images, masks
    
    
    Возвращает две переменные:
    -------------------------
    - X: тензор 
        
    - y: тензор
        
    
    Пример использования:
    ---------------------
    >>> X, Y = load_mri('mri_data')
    >>>
    """
    X = []
    y = []
        
    to_images = sorted(glob.glob(os.path.join(path, 'images', '*')))
    
    for to_img in to_images:
        name = os.path.basename(to_img)
        
        img = __center_crop__(Image.open(to_img))
        mask = __center_crop__(Image.open(os.path.join(path, 'masks', 'mask_' + name)))

        img = img.resize(size=size, resample=Image.BICUBIC)
        mask = mask.resize(size=size, resample=Image.BICUBIC)

        pix_img = np.asarray(img)
        pix_img = pix_img - pix_img.min()
        pix_img = pix_img / pix_img.max()
                               
        pix_mask = np.asarray(mask)
        if pix_mask.max() != 0:
            pix_mask = pix_mask / pix_mask.max()

        X.append(pix_img)
        y.append(pix_mask)
    
    
    X = np.expand_dims(np.stack(X), 1)
    y = np.stack(y)
    
    return X, y


def load_mnist_8():
    """
    Загружает данные с цифрами
    
    
    Возвращает две переменные:
    -------------------------
    - X: матрица 1797x64
        картинки в виде векторов длинной 64
        
    - y: вектор длинной 1797
        true вектор, хранящий класс каждой картинки
        
    
    Пример использования:
    ---------------------
    >>> X, y = load_data()
    >>>
    """
    X, y = load_digits(return_X_y=True)
    return X, y

def load_fmnist():
    local = os.getcwd() # ../ in Linux
    root = os.path.join(local,'data', 'fmnist')
    if not os.path.exists(root):
        os.makedirs(root)
    fmnist = torchvision.datasets.FashionMNIST(root, train=True, download=True)
    x_train = fmnist.train_data.numpy()
    y_train = fmnist.train_labels.numpy()
    
    fmnist = torchvision.datasets.FashionMNIST(root, train=False, download=True)
    x_test = fmnist.test_data.numpy()
    y_test = fmnist.test_labels.numpy()
    
    x = np.concatenate((x_train, x_test))
    x = np.expand_dims(x, 1)
    y = np.concatenate((y_train, y_test))
    return x, y


def load_mnist(shape='tensor'):
    local = os.getcwd() # ../ in Linux
    root = os.path.join(local,'data', 'mnist')
    if not os.path.exists(root):
        os.makedirs(root)
    mnist = torchvision.datasets.MNIST(root, train=True, download=True)
    x_train = mnist.data.numpy()
    y_train = mnist.targets.numpy()
    
    mnist = torchvision.datasets.MNIST(root, train=False, download=True)
    x_test = mnist.data.numpy()
    y_test = mnist.targets.numpy()
    
    x = np.concatenate((x_train, x_test))
    x = np.expand_dims(x, 1)
    y = np.concatenate((y_train, y_test))
    
    if shape == 'vector':
        x = x.reshape((len(x), 28*28))
    elif shape == 'matrix':
        x = x.reshape((len(x), 28, 28))
        
    return x, y


def calculate_pad(input_size, kernel_size, stride, output_size):
    pad = output_size * stride - input_size + kernel_size - 1
    pad /= 2
    if int(pad) != pad:
        print('С такими параметрами нереально подобрать размер pad-а!')
    else:
        return int(pad)


def show_image(img, figsize=(5,5)):
    """
    Показывает изображение
    
    Параметры:
    - img: numpy.array
        массив numpy, с тремя или одним каналом (цветное или ч/б фото)
    """
    if len(img.shape) < 2:
        s = np.sqrt(len(img)).astype(int)
        img = img.reshape((s,s))
        
    plt.figure(figsize=figsize)
    plt.imshow(img)
    plt.axis('off')
    plt.show()
    
def show_history(train_history, valid_history=None, hide_left=0, figsize=None, fontsize=30, title=None, width=4):
    if figsize is not None:
        plt.figure(figsize=figsize)
        
    N = len(train_history)
    plt.plot(np.arange(hide_left, N), train_history[hide_left:], color='blue', label='train', linewidth=width)
    
    if valid_history is not None:
        plt.plot(np.arange(hide_left, N), valid_history[hide_left:], color='green', label='val', linewidth=width)
        
    plt.title(title)
    plt.legend(fontsize=fontsize)
    plt.grid()
    plt.show()
        
    
        
    
    
    
    
    
    