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

    if args.object == "pool":
        a = np.where(img[:,:,2] > 250, 255, 0).astype(np.uint8)
        kernel = np.ones((3,3),dtype=np.uint8)
        a = cv2.dilate(a, kernel, iterations=1)
        a = cv2.erode(a, kernel, iterations=2)
    elif args.object == "roof":
        a = np.where((img[:,:,0] > 170) & (img[:,:,1] > 90) & (img[:,:,2] < 130), 255, 0).astype(np.uint8)
        kernel = np.ones((3,3),dtype=np.uint8)
        a = cv2.dilate(a, kernel, iterations=2)
    else:
        a = np.where((img[:,:,0] > 195) & (img[:,:,0] < 215) & (img[:,:,2] > 130) & (img[:,:,2] < 145), 255, 0).astype(np.uint8)
        kernel = np.ones((3,3),dtype=np.uint8)
        a = cv2.dilate(a, kernel, iterations=2)
        a = cv2.erode(a, kernel, iterations=1)

    cnts, _ = cv2.findContours(a,mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
    areas = []
    for idx, cnt in enumerate(cnts):
        areas.append(cv2.contourArea(cnt))
    if not areas:
        print("Object undetected")
        exit(1)

    thickness = 3 if scale in ["040","080","100"] else 1
    idx = np.argmax(areas)
    cv2.drawContours(img, cnts, contourIdx=idx, color=(0,255,0), thickness=thickness)
    plt.imshow(img) 
    plt.show()

    if args.save_result:
        os.makedirs("results/task3", exist_ok=True)
        res_name = f"results/task3/{scale}_{args.object}.jpg"
        res_mask = f"results/task3/{scale}_{args.object}_mask.jpg"
        cv2.imwrite(res_name, img[:,:,::-1])
        cv2.imwrite(res_mask, a)

if __name__ == "__main__":
    main()

