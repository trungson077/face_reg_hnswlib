import cv2
import numpy as np
import imutils
import face_recognition
from imutils import paths
import matplotlib.pyplot as plt
import os
from collections import Counter
import time
import hnswlib
from constant import DIM, NUM_ELEMENTS, FX, FY

output_name = []
known_face_names = []
video_capture = cv2.VideoCapture(0)
p = hnswlib.Index(space='l2', dim=DIM)  # the space can be changed - keeps the data, alters the distance function.
p.load_index("images.bin", max_elements = NUM_ELEMENTS)
imagePaths = list(paths.list_images('images'))

for i, imagePath in enumerate(imagePaths):
    name = imagePath.split(os.path.sep)[-2]
    known_face_names.append(name)

def append_names(frame):
        small_frame = cv2.resize(frame, (0, 0), fx=FX, fy=FY)
        rgb_small_frame = small_frame[:, :, ::-1]

        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        face_names = []

        for face_encoding in face_encodings:
            labels, distances = p.knn_query(np.expand_dims(face_encoding, axis = 0), k = 1)
            known_face_encoding = p.get_items([labels])
            name = "unknown"
            if distances < 0.13:
                name = known_face_names[labels[0][0]]
            face_names.append(name)

        return face_names

        #Draw rectangle in faces 
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

while True:  
    ret, frame = video_capture.read()
    #Test
    names =  append_names(frame)
    print(names)




        
        
