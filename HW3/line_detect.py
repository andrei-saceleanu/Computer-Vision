import os
import argparse

import cv2
import matplotlib.pyplot as plt
import numpy as np

from configparser import ConfigParser

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_img",
        type=str,
        required=True,
        help="input image for line detection"
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
        help="plot detected lines"
    )

    parser.add_argument(
        "--save",
        action="store_true",
        required=False,
        help="save result image"
    )

    return parser.parse_args()

def line_detect_hough(img, init_img, threshold=120, show=False):
    """
    Hough Line Transform - Line detection

    Parameters
    ----------
    img - binary image of detected edges (from Canny or other edge detector)
    init_img - original image
    threshold - threshold for accumulated values (a line is kept if it has at least threshold votes)
    show - plot image or not
    """

    h, w = img.shape

    max_r = int(round(np.sqrt(h**2 + w**2)))

    angles = np.deg2rad(np.arange(-90, 90))
    dists = np.linspace(-max_r, max_r, 2*max_r)

    # accumulator matrix
    mat = np.zeros((2 * max_r, len(angles)))

    edge_pixel_coords = np.stack(np.where(img > 0),axis=-1)
    sin_cos = np.stack([np.sin(angles),np.cos(angles)],axis=-1)
    res = (edge_pixel_coords[:,np.newaxis,:] * sin_cos[np.newaxis,:,:]).sum(-1)

    m, n = res.shape
    x_values = res.reshape(-1).astype(np.int64) + max_r
    y_values = np.tile(np.arange(n), reps=(m,))

    # each edge point casts its vote to each member line of its pencil
    mat = np.histogram2d(x_values, y_values, bins=(np.arange(2*max_r+1), np.arange(len(angles)+1)))[0]
    idxs = np.argsort(mat, axis=None)[::-1]

    line_img = init_img.copy()
    lines = []
    for idx in idxs:
        pos_y, pos_x = np.unravel_index(idx, mat.shape)

        if mat[pos_y, pos_x] > threshold:
            rho = int(dists[pos_y])
            theta = angles[pos_x]
            
            lines.append((rho, np.rad2deg(theta)))

            # convert from polar to cartesian and handle vertical lines
            if np.abs(theta) > 1e-3:
                m = -np.cos(theta)/np.sin(theta)
                b = rho/np.sin(theta)

                start_point = (0, int(m*0 + b))
                end_point = (w, int(m*w + b))
            else:
                start_point = (rho, 0)
                end_point = (rho, h)
            line_img = cv2.line(line_img, start_point, end_point, color=(255,0,0), thickness=3)

    if show:
        plt.imshow(line_img)
        plt.show()

    return lines, line_img

def get_section(input_img, config):

    cparser = ConfigParser()
    cparser.read(config)
    section_name = os.path.splitext(os.path.basename(input_img))[0]
    section = cparser[section_name] if cparser.has_section(section_name) else cparser.default_section
    return section

if __name__ == "__main__":

    args = parse_args()
    input_img = args.input_img
    img = cv2.cvtColor(cv2.imread(input_img), cv2.COLOR_BGR2RGB)

    section = get_section(input_img, args.config)
    threshold = int(section["thresh"])
    ksize, sigmaX, sigmaY = eval(section["blur"])
    low, high = eval(section["canny"])

    blurred_img = cv2.GaussianBlur(img, ksize, sigmaX, sigmaY)
    edge_img = cv2.Canny(blurred_img, low, high)

    _, res_img = line_detect_hough(edge_img, img, threshold, show=args.show)
    
    if args.save:
        os.makedirs("results/lines", exist_ok=True)
        res_img = cv2.cvtColor(res_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(
            os.path.join(
                "results/lines",
                os.path.splitext(os.path.basename(input_img))[0] + "_lines.jpg"
            ),
            res_img
        )
