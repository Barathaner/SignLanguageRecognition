from __future__ import annotations
from abc import ABC, abstractmethod
import cv2
import numpy as np
import mediapipe as mp
import os


class FeatureExtraction():
    """
    The Context defines the interface of interest to clients.
    TASK:
    INPUT: Cleaned data as Mp4
    OUTPUT: Matrix für jedes Video mit n Kanälen (Bild+Kanalxy)
    png.shape()= (width,height,3 = kanäle)
    BSP featurefromvideo.shape() = (width,height,3 )

    """

    def __init__(self, strategy: Strategy) -> None:
        """
        Usually, the Context accepts a strategy through the constructor, but
        also provides a setter to change it at runtime.
        """

        self._strategy = strategy

    @property
    def strategy(self) -> Strategy:
        """
        The Context maintains a reference to one of the Strategy objects. The
        Context does not know the concrete class of a strategy. It should work
        with all strategies via the Strategy interface.
        """

        return self._strategy

    @strategy.setter
    def strategy(self, strategy: Strategy) -> None:
        """
        Usually, the Context allows replacing a Strategy object at runtime.
        """

        self._strategy = strategy

    def extractFeature(self, cleaned_data_dir, source_data_dir) -> None:
        """
        The Context delegates some work to the Strategy object instead of
        implementing multiple versions of the algorithm on its own.
        """

        # ...

        print("Featureextraction: get Features with Strategy")
        self._strategy.do_algorithm(cleaned_data_dir)
        # print(",".join(result))

        # ...


class Strategy(ABC):
    """
    The Strategy interface declares operations common to all supported versions
    of some algorithm.

    The Context uses this interface to call the algorithm defined by Concrete
    Strategies.
    """

    @abstractmethod
    def do_algorithm(self, cleaned_data_dir: str):
        pass


"""
Concrete Strategies implement the algorithm while following the base Strategy
interface. The interface makes them interchangeable in the Context.
"""


class Skelleting_as_image(Strategy):
    """
    TASK: generate marker for each joint saved as image
    """

    def do_algorithm(self, cleaned_data_dir: str):
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles
        mp_hands = mp.solutions.hands
        mp_pose = mp.solutions.pose

        video_names = os.listdir(cleaned_data_dir)
        video_paths = [cleaned_data_dir + "/" + vid for vid in video_names]
        # os.chdir(cleaned_data_dir)
        # os.chdir("../features/skeletons")
        for video_name in video_names:
            try:
                cap = cv2.VideoCapture(cv2.samples.findFile(
                    cleaned_data_dir + "/" + video_name))  # cv2.samples.findFile("signer0_sample1_color.mp4")
            except:
                print("Shitted in my pants")
            finally:
                continue

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
                            cv2.imwrite("data/features/skeletons/" + video_name[:-4] + "_skeleton" + ".jpg", mask)
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
        pass


class anotherfeature(Strategy):
    # TODO
    def do_algorithm(self, cleaned_data_dir):
        pass
