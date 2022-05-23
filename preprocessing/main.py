import cv2
import numpy as np
import mediapipe as mp
import pandas as pd
import glob,os
from typing import Optional

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose

def clear_data():
    for f in glob.glob("train/signer*_depth.mp4"):
        os.remove(f)


def create_skeleton(video_path: Optional[str] = None, videoName):
    """
    Das macht hier sachen machen

    :param video_path:
    :param videoName:
    :return:
    """
    cap = cv2.VideoCapture(cv2.samples.findFile(video_path))  # cv2.samples.findFile("signer0_sample1_color.mp4")

    if cap.isOpened():
        # get vcap property
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        mask = np.zeros([width, height, 3], dtype="uint8")

    with mp_hands.Hands(
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as hands:
        with mp_pose.Pose(
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5) as pose:
            while cap.isOpened():
                success, image = cap.read()
                if not success:
                    print("Ignoring empty camera frame.")
                    # If loading a video, use 'break' instead of 'continue'.
                    cv2.imwrite("train/skeleton/" + videoName + "_color.jpg", mask)
                    break

                # To improve performance, optionally mark the image as not writeable to
                # pass by reference.
                image.flags.writeable = False
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = hands.process(image)
                # Draw the hand annotations on the image.
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            mask,
                            hand_landmarks,
                            mp_hands.HAND_CONNECTIONS,
                            mp_drawing_styles.get_default_hand_landmarks_style(),
                            mp_drawing_styles.get_default_hand_connections_style())
                # Flip the image horizontally for a selfie-view display.
                results = pose.process(image)
                mp_drawing.draw_landmarks(
                    mask,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

                cv2.imshow('Sign Language Recognition', cv2.flip(mask, 1))
                # Quit with 'q'
                if cv2.waitKey(5) & 0xFF == ord('q'):
                    break
    cap.release()


if __name__ == "__main__":

    df = pd.read_csv("data/train_labels.csv", sep=",", engine="python", encoding="utf-8", header=None)
    print(df)
    """
    Train-Videos in train/ kopieren
    Ordner "train/skeleton" und "data" erzeugen
    Ordner "data/" enthält die train_labels.csv und SignList_Classid_TR_EN.csv
    """
    for i in range(0, len(df)):
        videoName = df.iloc[i][0]
        videoPath = "train/" + videoName + "_color.mp4"
        print(videoPath)
        createSkeleton(videoPath, videoName)




    #clearData()
    #path = "signer0_sample1_color.mp4"
    #createSkeleton(path)
