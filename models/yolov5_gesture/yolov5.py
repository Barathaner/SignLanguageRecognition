import cv2
import numpy as np
import pandas as pd
from loguru import logger
import shutil
import os

if __name__ == "__main__":

    # create lables for yolov5
    data_src = "../data/features/skeletons/"
    train_dest = "train/"
    print(os.getcwd())
    # valid_dest = "./valid/"
    # 15 256 256 512 512
    df = pd.read_csv(data_src + "train.csv")
    for index, row in df.iterrows():
        file_name = row['file_name']
        file_name_label = file_name.rsplit(".", 1)[0] + '.txt'
        label = row['word']

        src = data_src + file_name

        # extract bounding box for skeleton
        image = cv2.imread(src)
        w, h, d = image.shape
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        non_zeros = np.transpose(np.nonzero(image_gray))

        width_px = np.max(non_zeros[:, 0]) - np.min(non_zeros[:, 0])
        height_px = np.max(non_zeros[:, 1]) - np.min(non_zeros[:, 1])
        x_px = np.min(non_zeros[:, 0]) + 0.5 * width_px
        y_px = np.min(non_zeros[:, 1]) + 0.5 * height_px

        width = width_px / w
        height = height_px / h
        x = x_px / w
        y = y_px / h

        # copies image to destination
        dest = "yolov5_gesture/" + train_dest + "images/" + file_name
        try:
            shutil.copyfile(src, dest)
        except:
            logger.warning("File not found: " + file_name)
            continue
        # creates a text file with similar name from corresponding image if not already created, 'w' to replace
        with open("yolov5_gesture/" + train_dest + 'labels/' + file_name_label, 'w') as f:
            logger.debug("create yolov5 label for image " + row['file_name'])
            f.write(f"{label} {x} {y} {width} {height}")

    # Model
    # model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # or yolov5n - yolov5x6, custom

    # Images
    # img = 'https://ultralytics.com/images/zidane.jpg'  # or file, Path, PIL, OpenCV, numpy, list

    # Inference
    # results = model(img)

    # Results
    # results.print()  # or .show(), .save(), .crop(), .pandas(), etc.
    # TODO: Retrain model as command in yolov5/ repository with the new labels in yolov5_gesture/
    # python train.py --img 512 --batch 2 --epochs 3 --data ../yolov5_gesture/dataset.yaml --weights ../yolov5_gesture/yolov5s.pt

    # model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
