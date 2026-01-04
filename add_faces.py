import cv2
import pickle
import numpy as np
import os

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print(" Camera not opening")
    exit()

cascade_path = os.path.join("data", "haarcascade_frontalface_default.xml")
facedetect = cv2.CascadeClassifier(cascade_path)

if facedetect.empty():
    print(" Haarcascade not loaded. Check path:", cascade_path)
    exit()

faces_data = []
i = 0

name = input("Enter Your Name: ")

print(" Camera started. Look at the camera...")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Frame not received")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        crop_img = frame[y:y+h, x:x+w]
        resized_img = cv2.resize(crop_img, (50, 50))

        if len(faces_data) < 100 and i % 10 == 0:
            faces_data.append(resized_img)

        i += 1
        cv2.putText(frame, f"Faces: {len(faces_data)}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow("Add Faces", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    if len(faces_data) >= 100:
        break

cap.release()
cv2.destroyAllWindows()

faces_data = np.asarray(faces_data).reshape(100, -1)

# ===== SAVE NAMES =====
os.makedirs("data", exist_ok=True)

if "names.pkl" not in os.listdir("data"):
    names = [name] * 100
else:
    with open("data/names.pkl", "rb") as f:
        names = pickle.load(f)
    names += [name] * 100

with open("data/names.pkl", "wb") as f:
    pickle.dump(names, f)

# ===== SAVE FACE DATA =====
if "faces_data.pkl" not in os.listdir("data"):
    faces = faces_data
else:
    with open("data/faces_data.pkl", "rb") as f:
        faces = pickle.load(f)
    faces = np.append(faces, faces_data, axis=0)

with open("data/faces_data.pkl", "wb") as f:
    pickle.dump(faces, f)

print(" Face data saved successfully")
