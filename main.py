import face_recognition
import cv2
import numpy as np


def lambda_handler(event, context):

    frame = cv2.imdecode(np.fromstring(event["body"], dtype=np.uint8), 1)
    face_locations = []

    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    face_locations = face_recognition.face_locations(small_frame, model="cnn")

    for top, right, bottom, left in face_locations:
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        face_image = frame[top:bottom, left:right]

        face_image = cv2.GaussianBlur(face_image, (99, 99), 30)

        frame[top:bottom, left:right] = face_image

    _, image = cv2.imencode(".jpg", frame)
    return image.tostring()


with open("fussoir.jpg", "rb") as r:
    with open("/tmp/fussoir-res.jpg", "wb") as f:
        f.write(lambda_handler({"body": r.read()}, None))
