import os
import logging
import imutils
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import time
import cv2

ASSETS_PATH = 'assets/'
MODEL_PATH = os.path.join(ASSETS_PATH, 'frozen_inference_graph.pb')
CONFIG_PATH = os.path.join(ASSETS_PATH, 'ssd_mobilenet_v2_coco_2018_03_29.pbtxt')
LABELS_PATH = os.path.join(ASSETS_PATH, 'labels.txt')
SCORE_THRESHOLD = 0.4
NETWORK_INPUT_SIZE = (300, 300)
NETWORK_SCALE_FACTOR = 1

logger = logging.getLogger('detector')
logging.basicConfig(level=logging.INFO)

# Reading coco labels
with open(LABELS_PATH, 'rt') as f:
    labels = f.read().rstrip('\n').split('\n')
logger.info(f'Available labels: \n{labels}\n')
COLORS = np.random.uniform(0, 255, size=(len(labels), 3))

# Loading model from file
logger.info('Loading model from tensorflow...')
ssd_net = cv2.dnn.readNetFromTensorflow(model=MODEL_PATH, config=CONFIG_PATH)
# Initiating camera
logger.info('Starting video stream...')
vs = VideoStream(src=0).start()
time.sleep(2.0)
fps = FPS().start()

while True:
    # Reading frames
    frame = vs.read()
    frame = imutils.resize(frame, width=1200)
    height, width, channels = frame.shape

    # Converting frames to blobs using mean standardization
    blob = cv2.dnn.blobFromImage(image=frame,
                                 scalefactor=NETWORK_SCALE_FACTOR,
                                 size=NETWORK_INPUT_SIZE,
                                 mean=(127.5, 127.5, 127.5),
                                 crop=False)

    # Passing blob through neural network
    ssd_net.setInput(blob)
    network_output = ssd_net.forward()

    # Looping over detections
    for detection in network_output[0, 0]:
        score = float(detection[2])
        class_index = np.int(detection[1])
        label = f'{labels[class_index]}: {score:.2%}'

        # Drawing likely detections
        if score > SCORE_THRESHOLD:
            left = np.int(detection[3] * width)
            top = np.int(detection[4] * height)
            right = np.int(detection[5] * width)
            bottom = np.int(detection[6] * height)

            cv2.rectangle(img=frame,
                          rec=(left, top, right, bottom),
                          color=COLORS[class_index],
                          thickness=4,
                          lineType=cv2.LINE_AA)

            cv2.putText(img=frame,
                        text=label,
                        org=(left, np.int(top*0.9)),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=2,
                        color=COLORS[class_index],
                        thickness=2,
                        lineType=cv2.LINE_AA)

    cv2.imshow("Detector", frame)

    # Exit loop by pressing "q"
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    fps.update()

fps.stop()
logger.info(f'\nElapsed time: {fps.elapsed() :.2f}')
logger.info(f' Approx. FPS: {fps.fps():.2f}')
cv2.destroyAllWindows()
vs.stop()
