"""
Sample call:

python scripts/labelme2detectron.py alpacas --save_path annotations/validation.json --dataset_name validation --csv processing-notebooks/final.csv  --masks_dir raw-openimages/annotations/correct-masks/

"""

import click
import json
import pandas as pd
import cv2
import os
from tqdm import tqdm
from skimage.measure import find_contours, approximate_polygon
from detectron2.structures import BoxMode
import numpy as np

def read_labelme_annotation(path):
    with open(path) as f:
        data = json.load(f)
    assert data['version']=='4.5.6'
    image_path = os.path.join(os.path.dirname(path), data['imagePath'])
    #image_path = data['imagePath']
    height = data['imageHeight']
    width = data['imageWidth']
    shapes = data['shapes']

    return image_path, height, width, shapes

def make_square(bbox, height, width):
    """Extends bounding box to square if possible and returns new shape
    """
    (x, y), (u, v) = bbox
    if x > u:
        x, u = u, x
    if y > v:
        y, v = v, y
    dx = u - x
    dy = v - y
    if dx == dy:
        return (x, y), (u, v)
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
    return (x, y), (u, v)


def process_shapes(df, image_path, shapes, height, width, masks_dir):
    image_id, _ = os.path.basename(image_path).split("_")

    mask_rectangles = dict()
    for i, shape in enumerate(shapes):
        #start, end = shape['points']
        mask_rectangles[i] = make_square([ [int(x) for x in xs] for xs in shape['points']], height, width)

    if not (df.ImageID == image_id).any():
        return
    mask_data = {i: [] for i in mask_rectangles}
    for mask_path, dataset_name in df.loc[df.ImageID == image_id, ["MaskPath", "dataset"]].drop_duplicates().values:
        segmentation_mask = cv2.imread(os.path.join(masks_dir, mask_path), 0)/255.0
        if segmentation_mask.shape != (height, width):
            segmentation_mask = cv2.resize(segmentation_mask, (width, height))

        for i, ((x,y), (u, v)) in mask_rectangles.items():

            mask = segmentation_mask[y: v, x:u]
            #segmentation_mask[:y] = 0
            #segmentation_mask[v:] = 0
            #segmentation_mask[:, :x] = 0
            #segmentation_mask[:, u:] = 0

            #print(np.unique(mask))
            mask = np.pad(mask, 1, mode='constant')
            lbls = np.unique(mask)
            if len(lbls) == 1 and lbls[0] ==0:
                continue
            mask_data[i].append((mask.sum(), mask > 0))

    for i, ((x,y), (u, v)) in mask_rectangles.items():
        if len(mask_data[i]) == 0:
            continue
        k = np.argmax([x[0] for x in mask_data[i]])
        mask = mask_data[i][k][1]
        contours = find_contours(mask, 0.5)
        #contours = [
        #    approximate_polygon(contour, 2.5)
        #    for contour in contours]
        if len(contours) > 0:
            data = {
                #'image_id': str(image_id),
                'bbox': [x, y, u, v],
                'bbox_mode': BoxMode.XYXY_ABS, #<BoxMode.XYXY_ABS: 0>,
                'segmentation': [],
                'category_id': 0,
                'mask_path': mask_path
            }
            for contour in contours:
                points = [
                    min(max(p+q - 1, q), r) for xs in contour
                    for p, q, r in zip(xs[::-1], (x, y), (u, v))
                ]
                if len(points) < 6:
                    continue
                assert len(points) % 2 == 0
                assert len(points) >= 6, f"{points}"
                data['segmentation'].append(points)
            yield data

        else:
            continue


@click.command()
@click.argument('src')
@click.option('--masks_dir',
    default="../raw-openimages/annotations/correct-masks",
    help="directory where the masks are located")
@click.option('--csv', default="../processing-notebooks/final.csv",
    help="file with collected annotations from OpenImages"
)
@click.option('--save_path', default=None)
@click.option('--dataset_name', default=None)
def process_files(src, masks_dir, csv, save_path, dataset_name="train"):
    """Print FILENAME."""

    df = pd.read_csv(csv)
    df = df[(df.LabelName == "/m/0pcr") & (df.dataset == dataset_name)]
    image_ids = np.unique(df.ImageID.values)
    json_paths = [
        os.path.join(src, x)
        for x in os.listdir(src)
        if x.find("json") >= 0 and x.split("_")[0] in image_ids
    ]
    all_shapes = []
    for path in tqdm(json_paths):
        image_path, height, width, shapes = read_labelme_annotation(path)
        prepared = {
            'file_name': image_path,
            'height':height,
            'width': width,
            'annotations': []
        }
        for x in process_shapes(df, image_path, shapes, height, width, masks_dir):
            prepared['annotations'].append(x)
        if len(prepared['annotations']) > 0:
            all_shapes.append(prepared)
    if save_path is None:
        print(all_shapes)
    else:
        with open(save_path, "w") as f:
            json.dump(all_shapes, f)
    print(len(all_shapes))


if __name__ == '__main__':
    process_files()
