# Data: origins and collection

We use OpenImages dataset, since it provides annotations (both segmentation and bounding boxes) for the class named 'Alpaca'.
The annotations describe the whole animal bodies, and we've decided to use only animal faces, we've also used labelme to select bounding boxes for faces by hand.

For now we have 432 images with alpaca, llama or guanaco. These images are not published, since the dataset is quite big.

## Next steps to reproduce/process the data (all scripts are run from the current directory):

1. Prepare annotations to detectron2-compatible dictionaries and save them to .json.
We suppose that images are located at `alpacas` folder, we want to save json files to `annotations` folder, our csv file with markup was prepared at jupyter notebook and is stored at the repository; we've also collected segmentation masks from OpenImages for every image with alpaca on them and store them at `raw-openimages/annotations/correct-masks/`.

```
python scripts/labelme2detectron.py alpacas --save_path annotations/train.json --dataset_name train --csv processing-notebooks/final.csv  --masks_dir raw-openimages/annotations/correct-masks/

python scripts/labelme2detectron.py alpacas --save_path annotations/test.json --dataset_name test --csv processing-notebooks/final.csv  --masks_dir raw-openimages/annotations/correct-masks/

python scripts/labelme2detectron.py alpacas --save_path annotations/validation.json --dataset_name validation --csv processing-notebooks/final.csv  --masks_dir raw-openimages/annotations/correct-masks/
```
As a result, we have `train.json`, `test.json`, `validation.json`.

2. Training/validation of detectron2 detector
This step is not necessary, although we think that it might be useful for extending our train dataset for cyclegan.

  2.1. train detectron2 detector on these images:

  ```
  python scripts/train_detector.py annotations/train.json
  ```

  for now script has many default parameters and stores trained detector to `output` folder. We might to modify this behavior in future.

  2.2. Show predictions from recently trained model on one of the datasets:

  ```
  streamlit run scripts/check_model.py -- --model_name output/model_final.pth --json_path annotations/test.json
  ```

3. Convert annotation data and images from 2.1 to square images.
```
python scripts/detectron2cyclegan.py annotations/train.json
```

4. Extract and save images with human faces

I've decided to try [this dataset](https://www.kaggle.com/dataturks/face-detection-in-images) from kaggle. Download as:
```
kaggle datasets download -d dataturks/face-detection-in-images -p face-detection-ds
cd face-detection-ds
unzip face-detection-in-images.zip
```
