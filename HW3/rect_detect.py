import os
import argparse

import cv2
import matplotlib.pyplot as plt
import numpy as np

from itertools import combinations
from line_detect import line_detect_hough, get_section

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_img",
        type=str,
        required=True,
        help="input image for rectangle detection"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=False,
        default="img_config.ini",
        help="input config file"
    )
    parser.add_argument(
        "--show",
        action="store_true",
        required=False,
        help="plot detected rectangles"
    )

    parser.add_argument(
        "--save",
        action="store_true",
        required=False,
        help="save result image"
    )

    return parser.parse_args()

def polar2xy(rho, theta, size):
    """
    Convert polar representation of line to cartesian.
    Parameters
    ----------
    rho - radial distance
    theta - angle of radius relative to Ox
    size - image shape

    Returns
    -------
    start_point - point on line with x=0
    end_point - point on line with x=width of image

    """

    h, w = size
    theta = np.deg2rad(theta)
    if np.abs(theta) > 1e-3:
        m = -np.cos(theta)/np.sin(theta)
        b = rho/np.sin(theta)

        start_point = (0, int(m*0 + b))
        end_point = (w, int(m*w + b))
    else: # handle vertical lines
        start_point = (rho, 0)
        end_point = (rho, h)

    return start_point, end_point

def points2abc(p1, p2):
    """
    Obtain line coefficients A,B,C (Ax+By+C=0) from endpoints
    """

    A = (p1[1] - p2[1])
    B = (p2[0] - p1[0])
    C = (p1[0] * p2[1] - p2[0] * p1[1])
    return A, B, -C # return -C for convenience in intersection computation

def intersection(L1, L2, size):
    """
    Find point of intersection between lines expressed as (A,B,-C) from Ax+By+C=0
    If the point doesn't lie in the first quadrant defined by the image size
    or lines are parallel, return False
    """

    D  = L1[0] * L2[1] - L1[1] * L2[0]
    Dx = L1[2] * L2[1] - L1[1] * L2[2]
    Dy = L1[0] * L2[2] - L1[2] * L2[0]
    if D != 0:
        x = Dx / D
        y = Dy / D
        if (0 <= y <= size[0]) and (0 <= x <= size[1]):
            return int(x), int(y)
        return False
    else:
        return False


def main():

    args = parse_args()
    input_img = args.input_img
    img = cv2.cvtColor(cv2.imread(input_img), cv2.COLOR_BGR2RGB)

    section = get_section(input_img, args.config)
    threshold = int(section["thresh"])
    ksize, sigmaX, sigmaY = eval(section["blur"])
    low, high = eval(section["canny"])

    # blur for better edge detection
    blurred_img = cv2.GaussianBlur(img, ksize, sigmaX, sigmaY)
    edge_img = cv2.Canny(blurred_img, low, high)

    # each line is (r,ang) where ang is expressed in degrees
    lines, _ = line_detect_hough(edge_img, img, threshold)

    size = edge_img.shape
    rect_img = img.copy()
    
    # find pairs of parallel lines(angs absolute difference should be ideally 0)
    parallel_pairs = []
    for l1, l2 in combinations(lines, 2):
        if abs(l1[1] - l2[1]) < 2: 
            parallel_pairs.append((l1, l2))

    # find pairs of pairs that are approximately orthogonal
    # find the 4 rectangle corners and draw appropriate lines
    for p1, p2 in combinations(parallel_pairs, 2):
        if abs(abs(p1[0][1] - p2[0][1]) - 90) < 1:
            group4 = p1 + p2
            converted = []
            for elem in group4:
                converted.append(points2abc(*polar2xy(*elem, size=size)))
            l1, l2, l3, l4 = converted
            i1 = intersection(l1, l3, size)
            i2 = intersection(l1, l4, size)
            i3 = intersection(l2, l4, size)
            i4 = intersection(l2, l3, size)
            rect_img = cv2.line(rect_img, i1, i2, color=(0, 0, 255), thickness=3)
            rect_img = cv2.line(rect_img, i2, i3, color=(0, 0, 255), thickness=3)
            rect_img = cv2.line(rect_img, i3, i4, color=(0, 0, 255), thickness=3)
            rect_img = cv2.line(rect_img, i4, i1, color=(0, 0, 255), thickness=3)
    
    if args.show:
        plt.imshow(rect_img)
        plt.show()

    if args.save:
        os.makedirs("results/rectangles", exist_ok=True)
        res_img = cv2.cvtColor(rect_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(
            os.path.join(
                "results/rectangles",
                os.path.splitext(os.path.basename(input_img))[0] + "_rectangles.jpg"
            ),
            res_img
        )

if __name__ == "__main__":
    main()
