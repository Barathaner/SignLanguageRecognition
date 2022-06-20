import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import torch
import os

cap = cv2.VideoCapture(0)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)


def crop_img_size(img, img_size_to_crop=512):
    if ret == True:
        a1 = width / height
        a2 = height / width

        if (a1 > a2):

            # if width greater than height
            r_img = cv2.resize(img, (round(img_size_to_crop * a1), img_size_to_crop), interpolation=cv2.INTER_AREA)
            margin = int(r_img.shape[1] / 6)
            crop_img = r_img[0:img_size_to_crop, margin:(margin + img_size_to_crop)]

        elif (a1 < a2):

            # if height greater than width
            r_img = cv2.resize(img, (img_size_to_crop, round(img_size_to_crop * a2)), interpolation=cv2.INTER_AREA)
            margin = int(r_img.shape[0] / 6)
            crop_img = r_img[margin:(margin + img_size_to_crop), 0:img_size_to_crop]

        elif (a1 == a2):

            # if height and width are equal
            r_img = cv2.resize(img, (img_size_to_crop, round(img_size_to_crop * a2)), interpolation=cv2.INTER_AREA)
            crop_img = r_img[0:img_size_to_crop, 0:img_size_to_crop]

        if (crop_img.shape[0] != img_size_to_crop or crop_img.shape[1] != img_size_to_crop):
            crop_img = r_img[0:img_size_to_crop, 0:img_size_to_crop]

        return crop_img


def detectSkeleton(mp_drawing, mp_drawing_styles, mp_hands, mp_pose, img):
    mask = np.zeros([512, 512, 3], dtype="uint8")

    with mp_hands.Hands(
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as hands:
        with mp_pose.Pose(
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5) as pose:
            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            img.flags.writeable = False
            image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # hands
            results = hands.process(image)

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
            results = pose.process(image)
            mp_drawing.draw_landmarks(
                mask,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
    return cv2.flip(mask, 1)


if __name__ == "__main__":
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_hands = mp.solutions.hands
    mp_pose = mp.solutions.pose
    imagestream = []
    root = os.getcwd()
    model = torch.hub.load(root + '/models/yolov5', 'custom',
                           path=root + '/models/weights/best_b16_e100.pt',
                           source='local', device='cpu')
    # model = torch.hub.load('C:/Users\karl-/PycharmProjects/Mustererkennung/SignLanguageRecognition/models/yolov5', 'custom',path='C:/Users/karl-/PycharmProjects/Mustererkennung/SignLanguageRecognition/models/weights/best_b16_e100.pt', source='local', device='cpu')
    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        frame = crop_img_size(frame, 512)
        skeletonFlowFused = np.zeros([512, 512, 3], dtype="uint8")
        skeletonFlow = np.zeros([512, 512, 3], dtype="uint8")
        if not ret:
            print("Ignoring empty camera frame.")
            continue
        if len(imagestream) < 15:
            skeletonFlow = detectSkeleton(mp_drawing, mp_drawing_styles, mp_hands, mp_pose, frame)
            imagestream.append(skeletonFlow)
        if len(imagestream) >= 15:
            del imagestream[0]
            del imagestream[0]

        for skeletons in imagestream:
            skeletonFlowFused += skeletons
        if 13 <= len(imagestream) <= 17:
            skeletonFlowFused = cv2.cvtColor(skeletonFlowFused, cv2.COLOR_BGR2RGB)
            results = model(skeletonFlowFused, size=512)
            boxes = results.pandas().xyxy[0]
            if len(boxes) > 0:
                maxob = boxes['confidence'].idxmax()
                if boxes.iat[maxob, 4] > 0.5:
                    labelstring = "Class: " + boxes.iat[maxob, 6] + " {:.9f}".format(boxes.iat[maxob, 4])
                    cv2.putText(frame, labelstring, (int(boxes.iat[maxob, 0]), int(boxes.iat[maxob, 1])),
                                cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (255, 255, 255), 2, cv2.LINE_AA)
                    frame = cv2.rectangle(frame, (int(boxes.iat[maxob, 0]), int(boxes.iat[maxob, 1])),
                                          (int(boxes.iat[maxob, 2]), int(boxes.iat[maxob, 3])), (255, 255, 255), 3)

        preview = np.concatenate((frame, skeletonFlowFused), axis=1)
        # resize preview for better visibility
        resize_multiplier = 1.3
        preview = cv2.resize(preview, (0, 0), fx=resize_multiplier, fy=resize_multiplier)
        # show preview
        cv2.imshow('Gesture Recognition', preview)
        if cv2.waitKey(1) == 27 or cv2.waitKey(2) & 0xFF == ord("q"):
            break  # esc or 'q' to quit
cap.release()
cv2.destroyAllWindows()
