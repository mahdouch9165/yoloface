import os
import sys
import cv2
import numpy as np
from utils import *
from djitellopy import Tello
import queue
import threading

def update_frame_queue(drone, frame_queue):
    while True:
        frame = drone.get_frame_read().frame
        if frame is not None:
            if not frame_queue.empty():
                try:
                    frame_queue.get_nowait()
                except queue.Empty:
                    pass
            frame_queue.put(frame)


# Hardcoded parameters
model_cfg = './cfg/yolov3-face.cfg'
model_weights = './model-weights/yolov3-wider_16000.weights'
output_dir = 'outputs/'

# Check outputs directory
if not os.path.exists(output_dir):
    print(f'==> Creating the {output_dir} directory...')
    os.makedirs(output_dir)
else:
    print(f'==> Skipping create the {output_dir} directory...')

# Load the YOLOv3 network
net = cv2.dnn.readNetFromDarknet(model_cfg, model_weights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Connect to the Tello drone
drone = Tello()
drone.connect()
print(f"Drone battery: {drone.get_battery()}%")

# Start the video stream from the drone
drone.streamon()

# Get data from the drone's camera
#cap = drone.get_video_capture()
frame_queue = queue.Queue(maxsize=1)
tello_frame_thread = threading.Thread(target=update_frame_queue, args=(drone, frame_queue))
tello_frame_thread.daemon = True
tello_frame_thread.start()


# Define output_file based on whether it's an image or video
output_file = 'tello_yoloface.avi'

# Initialize the video writer
video_writer = cv2.VideoWriter(os.path.join(output_dir, output_file),
                               cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                               30, (960, 720))

wind_name = 'face detection using YOLOv3'
cv2.namedWindow(wind_name, cv2.WINDOW_NORMAL)

frame_skip = 5
frame_count = 0

while True:
    if frame_count % frame_skip == 0:
        if not frame_queue.empty():
            frame = frame_queue.get()
            has_frame = True
        else:
            continue

        # Create a 4D blob from a frame.
        frame_resized = cv2.resize(frame, (480, 360))  # Resize the frame
        blob = cv2.dnn.blobFromImage(frame_resized, 1 / 255, (IMG_WIDTH, IMG_HEIGHT),
                                    [0, 0, 0], 1, crop=False)

        # Sets the input to the network
        net.setInput(blob)

        # Runs the forward pass to get output of the output layers
        outs = net.forward(get_outputs_names(net))

        # Remove the bounding boxes with low confidence
        faces = post_process(frame, outs, CONF_THRESHOLD, NMS_THRESHOLD)
        print('[i] ==> # detected faces: {}'.format(len(faces)))
        print('#' * 60)

        # initialize the set of information we'll displaying on the frame
        info = [
            ('number of faces detected', '{}'.format(len(faces)))
        ]

        for (i, (txt, val)) in enumerate(info):
            text = '{}: {}'.format(txt, val)
            cv2.putText(frame, text, (10, (i * 20) + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_RED, 2)

        # Save the output video to file
        video_writer.write(frame.astype(np.uint8))

        cv2.imshow(wind_name, frame)

        key = cv2.waitKey(1)
        if key == 27 or key == ord('q'):
            print('[i] ==> Interrupted by user!')
            break
    frame_count += 1


video_writer.release()
drone.streamoff()
cv2.destroyAllWindows()

print('==> All done!')
print('***********************************************************')
