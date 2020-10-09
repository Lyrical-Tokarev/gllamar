"""
Usage:

To download data:

  python scripts/download_images.py download face-detection-ds/face_detection.json face-detection-ds/images

To extract bounding boxes and to save images:

  python scripts/download_images.py extract face-detection-ds/face_detection.json

"""
import click
import os
import json
import pandas as pd
import numpy as np
import requests
from tqdm import tqdm
from PIL import Image
from io import BytesIO
import cv2
import albumentations as A

from common_tools import make_square

@click.group()
def cli():
    pass

@cli.command()
@click.argument("json_path")
@click.argument("savedir", default="face-detection-ds/images")
def download(json_path, savedir):
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    df = pd.read_json(json_path, lines=True)
    for i, url in tqdm(enumerate(df['content'].values), total=df.shape[0]):
        savepath = os.path.join(savedir, f"image_{i}.png")
        if os.path.exists(savepath):
            continue
        response = requests.get(url)
        if response.status_code != 200:
            print(response.status_code, url)
        with Image.open(BytesIO(response.content)) as i:
            i.save(savepath)
        pass

@cli.command()
@click.argument("json_path")
@click.argument("datadir", default="face-detection-ds/images")
@click.argument("savedir", default="face-detection-ds/images_cropped")
@click.argument("size", default=256)
def extract(json_path, datadir, savedir, size):
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    transform = A.Compose([
        A.LongestMaxSize(size)
    ])
    df = pd.read_json(json_path, lines=True)
    for i, annotations in tqdm(enumerate(df['annotation'].values), total=df.shape[0]):
        image_path = os.path.join(datadir, f"image_{i}.png")
        image = cv2.imread(image_path)
        for k, annotation in enumerate(annotations):
            savepath = os.path.join(savedir, f"image_{i}_{k}.png")
            height = annotation['imageHeight']
            width = annotation['imageWidth']
            assert image.shape[0] == height and image.shape[1] == width
            if not 'Face' in annotation['label']:
                print(i, annotation)
                continue
            points = annotation['points']
            assert len(points) == 2
            start, end = points
            x, y = start['x'], start['y']
            u, v = end['x'], end['y']
            res = make_square((x, y, u, v), height, width)
            x, y, u, v = res
            selected_img = image[y:v, x: u]
            if np.min(selected_img.shape[:2]) > size:
                selected_img = transform(image=selected_img)['image']
            else:
                continue
            cv2.imwrite(savepath, selected_img)

            #print(annotation)
            #break
        #break
    pass


if __name__ == "__main__":
    cli()
