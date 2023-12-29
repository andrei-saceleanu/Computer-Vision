import os
import cv2

import matplotlib.pyplot as plt
import numpy as np

from argparse import ArgumentParser, RawTextHelpFormatter
from itertools import product
from tqdm import tqdm

def gaussian(sigma: float = 1, ksize: int = None):

    half_width = ksize // 2 if ksize is not None else 3 * sigma
    xy = np.stack(
        np.meshgrid(
            np.arange(-half_width, half_width+1),
            np.arange(-half_width, half_width+1),
            indexing='ij'
        ),
        axis=-1
    )
    r = np.linalg.norm(xy, axis=-1) ** 2
    scale_factor = 1/(2 * np.pi * sigma ** 2)
    return scale_factor * np.e ** -(r / (2 * sigma ** 2))

def box(ksize: int = 3):

    return np.ones((ksize, ksize)) / (ksize * ksize)

### auxiliary function for filter_cv
def _conv_same_channels(img, filter):
    h, w, c = img.shape
    ksize_y, ksize_x = filter.shape[0], filter.shape[1]
    pad_y, pad_x = ksize_y // 2, ksize_x // 2
    padded_img = np.zeros((h + 2 * pad_y, w + 2 * pad_x, c))
    padded_img[pad_y:pad_y+h, pad_x:pad_x+w, :] = img
    res = np.empty_like(img)

    for y in range(h):
        for x in range(w):
            res[y, x, :] = np.sum(
                padded_img[
                    y:(y + 2 * pad_y + 1),
                    x:(x + 2 * pad_x + 1),
                    :
                ]
                *
                filter
            )
    return res

### same operation as conv_same_channels but somewhat faster(more numpy vectorization)
def _conv_same_channels2(img, filter):
    h, w, c = img.shape
    ksize_y, ksize_x = filter.shape[0], filter.shape[1]
    pad_y, pad_x = ksize_y // 2, ksize_x // 2
    padded_img = np.zeros((h + 2 * pad_y, w + 2 * pad_x, c))
    padded_img[pad_y:pad_y+h, pad_x:pad_x+w, :] = img

    x = np.lib.stride_tricks.sliding_window_view(padded_img, window_shape=(ksize_y, ksize_x, c))
    y = x * filter
    y = y.sum(axis=(-3,-2,-1))
    return y

### apply convolutional filter on img
def filter_cv(img: np.array, filter: np.array):

    ndim = len(filter.shape)
    if ndim >= 4 or ndim <= 1:
        print("Invalid number of dimensions for filter")
        return None

    if ndim == 2:
        filter = np.expand_dims(filter, -1)
 
    if filter.shape[2] not in [1, img.shape[2]]:
        print("Invalid number of channels for filter")
        return None
    
    h, w, c = img.shape
    if c == filter.shape[2]:
        res = _conv_same_channels(img, filter)
    else:
        res = np.empty_like(img)
        for k in range(c):
            res[:,:,k] = _conv_same_channels(img[:,:,k][:,:,np.newaxis], filter)[:,:,0]
    return res

### same operation as filter_cv but somewhat faster
def filter_cv2(img: np.array, filter: np.array):

    ndim = len(filter.shape)
    if ndim >= 4 or ndim <= 1:
        print("Invalid number of dimensions for filter")
        return None

    if ndim == 2:
        filter = np.expand_dims(filter, -1)
 
    if filter.shape[2] not in [1, img.shape[2]]:
        print("Invalid number of channels for filter")
        return None
    
    h, w, c = img.shape
    if c == filter.shape[2]:
        res = _conv_same_channels2(img, filter)
    else:
        res = np.empty_like(img)
        for k in range(c):
            res[:,:,k] = _conv_same_channels2(img[:,:,k][:,:,np.newaxis], filter)[:,:,0]
    return res

def parse_args():
    parser = ArgumentParser(description="Homework 2", formatter_class=RawTextHelpFormatter)

    parser.add_argument(
        "--mode",
        type=str,
        required=False,
        default="generate",
        choices=["test", "generate", "single_generate"],
        help="test=confirm own implementation matches OpenCV's filter2D results\ngenerate=generate result images\nsingle_generate=generate one image with sigma and ksize for scale image"
    )
    parser.add_argument(
        "--filter_type",
        type=str,
        required=False,
        default="gaussian",
        choices=["box", "gaussian"],
        help="filter type to apply to each image"
    )
    parser.add_argument(
        "--sigma",
        type=float,
        required=False,
        default=1.0,
        help="gaussian filter sigma"
    )
    parser.add_argument(
        "--ksize",
        type=int,
        required=False,
        default=3,
        help="kernel size"
    )
    parser.add_argument(
        "--scale",
        type=str,
        required=False,
        default="100",
        choices=["010", "020", "0025", "040", "080", "100"],
        help="scale of selected image to be filtered"
    )
    return parser.parse_args()

def main():
    
    args = parse_args()

    if args.mode == "test":

        img = cv2.cvtColor(cv2.imread("Gura_Portitei_Scara_020.jpg"),cv2.COLOR_BGR2RGB)
        filter = gaussian(sigma=2, ksize=11)

        img_filt = cv2.filter2D(img, ddepth=-1, kernel=filter, borderType=cv2.BORDER_CONSTANT)
        my_filt = filter_cv2(img, filter)
        total_size = img.shape[0] * img.shape[1] * img.shape[2]
        print(f"All elements within tolerance:{(np.abs(img_filt - my_filt) < 2).sum() == total_size}" )
    elif args.mode == "generate":

        ftype = args.filter_type
        os.makedirs(f"results/task1/{ftype}", exist_ok=True)
        scales = ["0025", "010", "020", "040", "080", "100"]
        sigmas = [1.0, 3.0, 5.0]
        ksizes = [3, 7, 15]
        
        for scale in scales:
            fname = f"Gura_Portitei_Scara_{scale}.jpg"
            img = cv2.imread(fname)

            if ftype == "gaussian":
            
                for sigma, ksize in tqdm(list(product(sigmas, ksizes))):
                    kernel = gaussian(sigma, ksize)

                    img_filt = cv2.filter2D(img, ddepth=-1, kernel=kernel, borderType=cv2.BORDER_CONSTANT)
                    cv2.imwrite(f"results/task1/{ftype}/{scale}_sigma_{int(sigma)}_ksize_{ksize}.jpg", img_filt)
            else:

                for ksize in tqdm(ksizes):
                    kernel = box(ksize)

                    img_filt = cv2.filter2D(img, ddepth=-1, kernel=kernel, borderType=cv2.BORDER_CONSTANT)
                    cv2.imwrite(f"results/task1/{ftype}/{scale}_ksize_{ksize}.jpg", img_filt)
    else:
        
        os.makedirs("results/task1/single_experiments", exist_ok=True)
        fname = f"Gura_Portitei_Scara_{args.scale}.jpg"
        img = cv2.imread(fname)
        ftype = args.filter_type
        res_name = None
        if ftype == "gaussian":
            kernel = gaussian(args.sigma, args.ksize)
            res_name = f"results/task1/single_experiments/{args.scale}_sigma_{int(args.sigma)}_ksize_{args.ksize}.jpg"
        else:
            kernel = box(args.ksize)
            res_name = f"results/task1/single_experiments/{args.scale}_ksize_{args.ksize}.jpg"
        img_filt = cv2.filter2D(img, ddepth=-1, kernel=kernel, borderType=cv2.BORDER_CONSTANT)
        cv2.imwrite(res_name, img_filt)




if __name__ == "__main__":
    main()