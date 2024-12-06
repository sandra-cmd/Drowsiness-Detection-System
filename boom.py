import numpy as np
import cv2
import dlib
import imutils
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import argparse
import os
from pygame import mixer
import time
import mediapipe as mp

mixer.init()
mixer.music.load("music.wav")

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

mp_drawing = mp.solutions.drawing_utils

drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# Function to detect head position
# Function to detect head position
def detect_head_position(shape, image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Get the result
    results = face_mesh.process(image_rgb)

    img_h, img_w, img_c = image.shape
    face_3d = []
    face_2d = []

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for idx, lm in enumerate(face_landmarks.landmark):
                if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                    if idx == 1:
                        nose_2d = (lm.x * img_w, lm.y * img_h)
                        nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)

                    x, y = int(lm.x * img_w), int(lm.y * img_h)

                    # Get the 2D Coordinates
                    face_2d.append([x, y])

                    # Get the 3D Coordinates
                    face_3d.append([x, y, lm.z])

            # Convert it to the NumPy array
            face_2d = np.array(face_2d, dtype=np.float64)

            # Convert it to the NumPy array
            face_3d = np.array(face_3d, dtype=np.float64)

            if len(face_3d) >= 4 and len(face_2d) >= 4:
                # The camera matrix
                focal_length = 1 * img_w

                cam_matrix = np.array([[focal_length, 0, img_h / 2],
                                       [0, focal_length, img_w / 2],
                                       [0, 0, 1]])

                # The distortion parameters
                dist_matrix = np.zeros((4, 1), dtype=np.float64)

                # Solve PnP
                success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

                # Get rotational matrix
                rmat, jac = cv2.Rodrigues(rot_vec)

                # Get angles
                angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

                # Get the y rotation degree
                x = angles[0] * 360
                y = angles[1] * 360
                z = angles[2] * 360

                # See where the user's head tilting
                if y < -10:
                    text = "Looking Left"

                elif y > 10:
                    text = "Looking Right"
                elif x < -10:
                    text = "Looking Down"
                    mixer.music.play()
                elif x > 10:
                    text = "Looking Up"
                    mixer.music.play()
                else:
                    text = "Forward"

                # Display the nose direction
                nose_3d_projection, _ = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)

                p1 = (int(nose_2d[0]), int(nose_2d[1]))
                p2 = (int(nose_2d[0] + y * 10), int(nose_2d[1] - x * 10))

                cv2.line(image, p1, p2, (255, 0, 0), 3)

                # Add the text on the image
                cv2.putText(image, text, (20, img_h - 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
                cv2.putText(image, "x: " + str(np.round(x, 2)), (500, img_h - 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 0, 255), 2)
                cv2.putText(image, "y: " + str(np.round(y, 2)), (500, img_h), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),
                            2)
                cv2.putText(image, "z: " + str(np.round(z, 2)), (500, img_h + 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 0, 255), 2)

            FACE_CONNECTIONS = [
                (10, 338), (338, 297), (297, 332), (332, 284), (284, 251), (251, 389), (389, 356), (356, 454),
                (454, 323), (323, 361), (361, 288), (288, 397), (397, 365), (365, 379), (379, 378), (378, 400),
                (400, 377), (377, 152), (152, 148), (148, 176), (176, 149), (149, 150), (150, 136), (136, 172),
                (172, 58), (58, 132), (132, 93), (93, 234), (234, 127), (127, 162), (162, 21), (21, 54), (54, 103),
                (103, 67), (67, 109), (109, 10), (10, 338)
            ]
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=FACE_CONNECTIONS,
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=drawing_spec
            )

    return image

# Function to raise alarm
def alarm(msg):
    global alarm_status
    global alarm_status2
    global saying

    while alarm_status:
        print('user drowsy')
        s = 'espeak "' + msg + '"'
        os.system(s)
        mixer.music.play()

    if alarm_status2:
        print('user yawning')
        saying = True
        s = 'espeak "' + msg + '"'
        os.system(s)
        mixer.music.play()
        saying = False

# Function to calculate eye aspect ratio
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Function to calculate lip distance
def lip_distance(shape):
    top_lip = shape[50:53]
    top_lip = np.concatenate((top_lip, shape[61:64]))

    low_lip = shape[56:59]
    low_lip = np.concatenate((low_lip, shape[65:68]))

    top_mean = np.mean(top_lip, axis=0)
    low_mean = np.mean(low_lip, axis=0)

    distance = abs(top_mean[1] - low_mean[1])
    return distance

# Argument parsing
ap = argparse.ArgumentParser()
ap.add_argument("-w", "--webcam", type=int, default=0, help="index of webcam on system")
args = vars(ap.parse_args())

# Constants and variables
YAWN_THRESH = 25
alarm_status = False
alarm_status2 = False
saying = False
COUNTER = 0

# Loading the predictor and detector
detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")  # Faster but less accurate
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Starting video stream
vs = VideoStream(src=args["webcam"]).start()
time.sleep(1.0)

while True:
    start = time.time()  # Define start time

    frame = vs.read()
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    rects = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

    for (x, y, w, h) in rects:
        rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))

        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        left_eye = shape[36:42]
        right_eye = shape[42:48]

        leftEAR = eye_aspect_ratio(left_eye)
        rightEAR = eye_aspect_ratio(right_eye)

        ear = (leftEAR + rightEAR) / 2.0

        distance = lip_distance(shape)

        if ear < 0.25:  # Adjusted the threshold
            COUNTER += 1
            if COUNTER >= 20:  # Adjusted the number of frames
                if not alarm_status:
                    alarm_status = True
                    t = Thread(target=alarm, args=('wake up sir',))
                    t.daemon = True
                    t.start()
                cv2.putText(frame, "DROWSINESS ALERT!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            COUNTER = 0
            alarm_status = False

        if distance > YAWN_THRESH:
            cv2.putText(frame, "Yawn Alert", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if not alarm_status2 and not saying:
                alarm_status2 = True
                t = Thread(target=alarm, args=('take some fresh air sir',))
                t.daemon = True
                t.start()
        else:
            alarm_status2 = False

        cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "YAWN: {:.2f}".format(distance), (300, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Detect head position
        frame = detect_head_position(shape, frame)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cv2.destroyAllWindows()
vs.stop()
