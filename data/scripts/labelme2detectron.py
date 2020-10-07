import click
import json
import pandas as pd
import cv2
import os
from tqdm import tqdm
from skimage.measure import find_contours, approximate_polygon
from detectron2.structures import BoxMode


def read_labelme_annotation(path):
    with open(path) as f:
        data = json.load(f)
    assert data['version']=='4.5.6'
    #image_path = os.path.join(os.path.dirname(path), data['imagePath'])
    image_path = data['imagePath']
    height = data['imageHeight']
    width = data['imageWidth']
    shapes = data['shapes']

    return image_path, height, width, shapes


def process_shapes(df, image_path, shapes, height, width, masks_dir):
    image_id, _ = os.path.basename(image_path).split("_")

    mask_rectangles = dict()
    for i, shape in enumerate(shapes):
        start, end = shape['points']
        a, b = [int(x) for x in start]
        u, v = [int(x) for x in end]
        if a > u:
            a, u = u, a
        if b > v:
            b, v = v, b

        mask_rectangles[i] = ((a, b), (u, v))

    for mask_path, dataset_name in df.loc[df.ImageID == image_id, ["MaskPath", "dataset"]].values:
        segmentation_mask = cv2.imread(os.path.join(masks_dir, mask_path), 0)
        if segmentation_mask.shape!= (height, width):

            segmentation_mask = cv2.resize(segmentation_mask, (width, height)) > 0

        for i, ((x,y), (u, v)) in mask_rectangles.items():
            mask = segmentation_mask[y: v, x:u]
            contours = find_contours(mask, 0.5)
            contours = [
                approximate_polygon(contour, 2.5)
                for contour in contours]
            if len(contours) > 0:
                data = {
                    #'image_id': str(image_id),
                    'bbox': [x, y, u, v],
                    'bbox_mode': BoxMode.XYXY_ABS, #<BoxMode.XYXY_ABS: 0>,
                    'segmentation': [
                        [
                            p+q for xs in contour for p, q in zip(xs[::-1], (x, y))
                        ]
                        for contour in contours
                    ],
                    'category_id': 0,
                    'mask_path': mask_path
                }
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
def process_files(src, masks_dir, csv, save_path):
    """Print FILENAME."""
    json_paths = [
        os.path.join(src, x)
        for x in os.listdir(src)
        if x.find("json") >= 0
    ]

    df = pd.read_csv(csv)
    df = df[df.LabelName=="/m/0pcr"]
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
            all_shapes.append(prepared)
    if save_path is None:
        print(all_shapes)
    else:
        with open(save_path, "w") as f:
            json.dump(all_shapes, f)


if __name__ == '__main__':
    process_files()