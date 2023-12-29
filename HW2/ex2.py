import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from argparse import ArgumentParser, RawTextHelpFormatter

from ex1 import gaussian

def parse_args():
    parser = ArgumentParser(description="Homework 2", formatter_class=RawTextHelpFormatter)

    parser.add_argument(
        "--scale",
        type=str,
        required=False,
        default="100",
        choices=["010", "020", "0025", "040", "080", "100"],
        help="scale of selected image to be filtered"
    )
    parser.add_argument(
        "--filter_type",
        type=str,
        required=False,
        default="sobel",
        choices=["sobel", "laplacian", "canny"],
        help="edge detection options"
    )
    parser.add_argument(
        "--save_result",
        action="store_true",
        help="enable/disable saving of processed image"
    )
    return parser.parse_args()


def main():

    args = parse_args()
    scale = args.scale
    fname = f"Gura_Portitei_Scara_{scale}.jpg"
    img = cv2.cvtColor(cv2.imread(fname),cv2.COLOR_BGR2GRAY)

    if args.filter_type == "sobel":
        kernel = np.array([
            [-1,0,1],
            [-2,0,2],
            [-1,0,1]
        ])
        img = cv2.filter2D(img, ddepth=-1, kernel=kernel)
    elif args.filter_type == "laplacian":
        kernel = np.array([
            [0,1,0],
            [1,-4,1],
            [0,1,0]
        ])
        img = cv2.filter2D(img, ddepth=-1, kernel=kernel)
        img *= 2
    else:
        img = cv2.Canny(img, 100, 200)
        
    plt.imshow(img,cmap="gray") 
    plt.show()

    if args.save_result:
        os.makedirs("results/task2", exist_ok=True)
        res_name = f"results/task2/{scale}_{args.filter_type}_edges.jpg"
        cv2.imwrite(res_name, img)

if __name__ == "__main__":
    main()