# SignLanguageRecognition

Live classification of sign language into words via webcam.  

## Installation

### Install dependencies

```sh
pip install -r requirements.txt
```

### Clone yolov5 into `models/`

```sh
git clone git@github.com:ultralytics/yolov5.git models/yolov5
```

## Run

- <kbd>Esc</kbd> or <kbd>q</kbd> to close the window 

```sh
python livedetection.py
```
> Tested with words: `friend`, `sister`

- It opens a side-by-side window
  - Left: Live-Webcam with ROI, Right: Skeleton-Flow

## Functionality

- [Dataset](https://chalearnlap.cvc.uab.cat/dataset/40/description/) for train and test: 
- [Mediapipe](https://google.github.io/mediapipe/) for skeleton
  - [Pose](https://google.github.io/mediapipe/solutions/pose.html)
  - [Hands](https://google.github.io/mediapipe/solutions/hands.html)
- [OpenCV](https://opencv.org/) for image processing
- [YOLOv5](https://github.com/ultralytics/yolov5) for object detection (classification)
  - Model: [YOLOv5s](https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data#2-select-a-model)
  - trained model with YOLOv5 with command (with best results):

  ```sh
  python train.py --img 512 --batch 16 --epochs 100 --data models/yolov5_gesture/dataset.yaml --weights models/yolov5_gesture/yolov5s.pt
  ```
  