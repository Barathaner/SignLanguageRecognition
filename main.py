
#read video
#start timer
#start copying
#after ... time saving


import cv2
from datetime import datetime, timedelta

from preprocessing.featureextraction import *
# the duration (in seconds)
duration = 5
cap = cv2.VideoCapture(0)
qu = 0
countdown = 4
counting = False
start_time_capturestart = None
end_time_capturestart = None
diff = None
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
img_size = 512
video_duration = 2
recording = False
extract = False
start_time_recording = None
fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
out = cv2.VideoWriter('fromWebcam/output.mp4',fourcc, 30, (512,512))

while True:
    ret, frame = cap.read()
    cv2.imshow('frame', frame)
    k = cv2.waitKey(10)
    if k & 0xFF == ord("s"):  # reset the timer
        start_time_capturestart = datetime.now()
        end_time_capturestart = datetime.now() + timedelta(seconds=countdown)
        diff = (end_time_capturestart - datetime.now()).seconds  # converting into seconds
        counting = True

    if k & 0xFF == ord("q"):  # quit all
        qu = 1
        break

    if k & 0xFF == ord("x"):  # quit all
        fetest = FeatureExtraction(Skelleting_as_image())
        fetest.extractFeature("fromWebcam/", "fromWebcam/")
        break

    while recording:
        ret, frame = cap.read()
        k = cv2.waitKey(3)
        if (datetime.now() - start_time_recording).seconds <= video_duration:

            if ret == True:
                a1 = width / height
                a2 = height / width

                if (a1 > a2):

                    # if width greater than height
                    r_img = cv2.resize(frame, (round(img_size * a1), img_size), interpolation=cv2.INTER_AREA)
                    margin = int(r_img.shape[1] / 6)
                    crop_img = r_img[0:img_size, margin:(margin + img_size)]

                elif (a1 < a2):

                    # if height greater than width
                    r_img = cv2.resize(frame, (img_size, round(img_size * a2)), interpolation=cv2.INTER_AREA)
                    margin = int(r_img.shape[0] / 6)
                    crop_img = r_img[margin:(margin + img_size), 0:img_size]

                elif (a1 == a2):

                    # if height and width are equal
                    r_img = cv2.resize(frame, (img_size, round(img_size * a2)), interpolation=cv2.INTER_AREA)
                    crop_img = r_img[0:img_size, 0:img_size]

                if (crop_img.shape[0] != img_size or crop_img.shape[1] != img_size):
                    crop_img = r_img[0:img_size, 0:img_size]

                if (crop_img.shape[0] == img_size and crop_img.shape[1] == img_size):
                    out.write(crop_img)
                out.write(crop_img)
                cv2.putText(crop_img, "recording", (70, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2,
                    cv2.LINE_AA)  # adding timer text
                cv2.imshow('frame', crop_img)
        else:
            recording = False
            out.release()

    while counting:

        ret, frame = cap.read()
        cv2.putText(frame, str(diff), (70, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2,
                    cv2.LINE_AA)  # adding timer text
        cv2.imshow('frame', frame)
        diff = (end_time_capturestart - datetime.now()).seconds  # converting into seconds
        k = cv2.waitKey(10)

        if k & 0xFF == ord("q"):  # quit all
            qu = 1
            break

        if diff == 0:
            recording = True
            start_time_recording=datetime.now()
            counting = False



    if qu == 1:
        break

cap.release()
cv2.destroyAllWindows()