import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from argparse import ArgumentParser, RawTextHelpFormatter

def parse_args():
    parser = ArgumentParser(description="Homework 2", formatter_class=RawTextHelpFormatter)

    parser.add_argument(
        "--object",
        type=str,
        required=False,
        default="pool",
        choices=["pool", "roof", "helipad"],
        help="object to be outlined in the input image"
    )
    parser.add_argument(
        "--scale",
        type=str,
        required=False,
        default="100",
        choices=["010", "020", "0025", "040", "080", "100"],
        help="scale of selected image to be filtered"
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
    img = cv2.cvtColor(cv2.imread(fname),cv2.COLOR_BGR2RGB)

    
    thickness = 5 if scale in ["040","080","100"] else 1
    if scale == "0025":
        scale = 0.025
    else:
        scale = int(scale)/100
    w = int(scale * 150)
    h = int(scale * 80)

    if args.object == "pool":
        kernel = np.ones((h,w))
        a = img[:, :, 2].astype(np.float64) / 255
        filtered = cv2.filter2D(a, ddepth=-1,kernel=kernel)
        pos = np.unravel_index(np.argmax(filtered), filtered.shape)
        tl = (pos[1] - w//2, pos[0] - h//2)
        br = (pos[1] + w//2, pos[0] + h//2)
        img = cv2.rectangle(img, tl, br, color=(255,0,0), thickness=thickness)
        plt.imshow(img)
        plt.show()

        if args.save_result:
            os.makedirs("results/task3", exist_ok=True)
            res_name = f"results/task3/{scale}_{args.object}_sol2.jpg"
            cv2.imwrite(res_name, img[:,:,::-1])

if __name__ == "__main__":
    main()

