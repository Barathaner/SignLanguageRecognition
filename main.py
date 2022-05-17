import cv2
import numpy as np
import mediapipe as mp
import pandas as pd
import glob,os

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose

def createList() -> pd.DataFrame:
    """
    train_labels.csv einelesen und Dataframe returnen
    WortID ersetzen
    :return: 1. Spalte: Dateiname, 2. Spalte: Wort
    """
    return pd.DataFrame()

def clearData():

    for f in glob.glob("train/signer*_depth.mp4"):
        os.remove(f)



def createSkeleton(videoPath, styleFunc):
    cap = cv2.VideoCapture(cv2.samples.findFile(videoPath))  # cv2.samples.findFile("signer0_sample1_color.mp4")

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
                    cv2.imwrite("skeleton.jpg", mask)
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
    # Daten holen (manuell downloaden und entzippen)
    # Funktion, die einen Dataframe mit Dateinamen zurückgibt
    # createList()
    # Dataframe in Funktion und erstellt vom Video ein Skelett-Frame
    # createSkeleton()

    # clearData()
    path = "signer0_sample1_color.mp4"
    createSkeleton(path)
