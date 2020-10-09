import click
import os
import json
import cv2
import numpy as np
from tqdm import tqdm

import albumentations as A
from collections import Counter


from common_tools import make_square


def get_data(json_path):
    with open(json_path) as f:
        data = json.load(f)
    return data


@click.command()
@click.argument("json_path")
@click.option("--savedir", default="resized_alpacas")
@click.option("--size", default=256)
@click.option("--ignore_small", default=True)
def run(json_path, savedir, size, ignore_small=True):
    data = get_data(json_path)
    transform = A.Compose([
        A.LongestMaxSize(size)
    ])
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    sizes = []
    for info in tqdm(data):
        filename = info['file_name']
        identifier = os.path.basename(filename).split("_")[0]
        height = info['height']
        width = info['width']
        image = cv2.imread(filename)
        for i, a in enumerate(info['annotations']):
            x, y, u, v = make_square(a['bbox'], height, width)
            # TODO: maybe save segmentation? and blur background
            selected_img = image[y:v, x:u].copy()
            if np.min(selected_img.shape[:2]) >= size:
                selected_img = transform(image=selected_img)['image']
            else:
                if ignore_small:
                    continue
            #sizes.append()
            sizes.append(np.min(selected_img.shape[:2]))

            savepath = os.path.join(savedir, f"{identifier}_{i}.png")
            # coords = []
            # for points in a['segmentation']:
            #     xx = [k-x for k in points[::2]]
            #     yy = [k-y for k in points[1::2]]
            #     p = np.stack([xx, yy], -1).astype('int32').reshape((1, -1, 2))
            #     #print(p.shape)
            #     #print(selected_img.dtype, selected_img.shape)
            #     cv2.fillPoly(selected_img, p, (255, 255, 255), 8)
            cv2.imwrite(savepath, selected_img)
        #print(filename)
        #return
    print(Counter(sizes))
    #print(list(data[0]))

if __name__ == "__main__":
    run()
