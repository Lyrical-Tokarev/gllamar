import click
import os
import json
import cv2
import numpy as np
from tqdm import tqdm

def get_data(json_path):
    with open(json_path) as f:
        data = json.load(f)
    return data

def make_square(bbox, height, width):
    """Extends bounding box to square if possible and returns new shape
    """
    x, y, u, v = bbox
    if x > u:
        x, u = u, x
    if y > v:
        y, v = v, y
    dx = u - x
    dy = v - y
    if dx == dy:
        return x, y, u, v
    size = max(dx, dy)
    #print(dx, dy)
    pad = int(np.abs(dx - dy) / 2)
    #print(pad)
    if size == dx:
        # extend dy
        y = max(y - pad, 0)
        v = min(y + size, height - 1)
    else:
        # extend dx
        x = max(x - pad, 0)
        u = min(x + size, width - 1)
    return x, y, u, v


@click.command()
@click.argument("json_path")
@click.option("--savedir", default="resized_alpacas")
def run(json_path, savedir="saves"):
    data = get_data(json_path)
    if not os.path.exists(savedir):
        os.makedirs(savedir)
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
    #print(list(data[0]))

if __name__ == "__main__":
    run()
