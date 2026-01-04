from sklearn.neighbors import KNeighborsClassifier
import cv2
import pickle
import numpy as np
import os
import csv
import time
from datetime import datetime
from win32com.client import Dispatch


# ================= SPEAK FUNCTION =================
def speak(text):
    speaker = Dispatch("SAPI.SpVoice")
    speaker.Speak(text)


# ================= CAMERA =================
video = cv2.VideoCapture(0)
if not video.isOpened():
    print(" Camera not opening")
    exit()


# ================= FACE DETECTOR =================
cascade_path = os.path.join("data", "haarcascade_frontalface_default.xml")
facedetect = cv2.CascadeClassifier(cascade_path)

if facedetect.empty():
    print(" Haarcascade not loaded:", cascade_path)
    exit()


# ================= LOAD DATA =================
with open('data/names.pkl', 'rb') as w:
    LABELS = pickle.load(w)

with open('data/faces_data.pkl', 'rb') as f:
    FACES = pickle.load(f)

print('Shape of Faces matrix --> ', FACES.shape)
print('Number of labels --> ', len(LABELS))


# ================= FIX 1: ENSURE MATCH =================
min_len = min(len(LABELS), FACES.shape[0])
LABELS = LABELS[:min_len]
FACES = FACES[:min_len]


# ================= TRAIN KNN =================
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(FACES, LABELS)


# ================= BACKGROUND =================
imgBackground = cv2.imread("background.png")
if imgBackground is None:
    print(" background.png not found")
    exit()


# ================= ATTENDANCE =================
os.makedirs("Attendance", exist_ok=True)
COL_NAMES = ['NAME', 'TIME']


# ================= MAIN LOOP =================
while True:
    ret, frame = video.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)

    attendance = None  # reset each frame

    for (x, y, w, h) in faces:
        crop_img = frame[y:y+h, x:x+w]
        resized_img = cv2.resize(crop_img, (50, 50)).flatten().reshape(1, -1)

        output = knn.predict(resized_img)[0]

        ts = time.time()
        date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
        timestamp = datetime.fromtimestamp(ts).strftime("%H:%M:%S")

        attendance = [output, timestamp]

        cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 2)
        cv2.rectangle(frame, (x, y-40), (x+w, y), (50, 50, 255), -1)
        cv2.putText(frame, output, (x, y-10),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)

    imgBackground[162:162+480, 55:55+640] = frame
    cv2.imshow("Frame", imgBackground)

    k = cv2.waitKey(1)

    # ================= MARK ATTENDANCE =================
    if k == ord('o') and attendance is not None:
        speak("Attendance Taken")
        filename = f"Attendance/Attendance_{date}.csv"
        file_exists = os.path.isfile(filename)

        with open(filename, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            if not file_exists:
                writer.writerow(COL_NAMES)
            writer.writerow(attendance)

        time.sleep(2)

    if k == ord('q'):
        break


video.release()
cv2.destroyAllWindows()
