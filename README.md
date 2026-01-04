## Project Overview

Simple face-recognition attendance project using OpenCV and a KNN classifier.

## Overview

This repository captures face images for named users, trains a simple K-Nearest Neighbors model on flattened face images, and provides a live recognition script that writes attendance records to CSV files. A small Streamlit app displays the daily attendance CSV.

## Repository Structure

- `add_faces.py` — capture face images for a person and save them to `data/faces_data.pkl` and `data/names.pkl`.
- `test.py` — runs live recognition, shows a camera preview, and writes attendance to `Attendance/Attendance_DD-MM-YYYY.csv`. Press `o` to mark attendance and `q` to quit.
- `app.py` — simple Streamlit app to view the attendance CSV for today.
- `camera.py` — (helper for camera functions, if present).
- `data/haarcascade_frontalface_default.xml` — Haar cascade used for face detection.
- `Attendance/` — folder containing CSV attendance files.

## Requirements

- Python 3.8+
- Packages: `opencv-python`, `numpy`, `scikit-learn`, `streamlit`, `streamlit-autorefresh`, `pywin32` (Windows speech support)

Install dependencies with pip (recommended in a virtual environment):

```bash
python -m venv .venv
source .venv/Scripts/activate    # Windows: ".venv\Scripts\activate"
pip install opencv-python numpy scikit-learn streamlit streamlit-autorefresh pywin32
```

## Usage

1. Capture faces for a user

	Run:

	```bash
	python add_faces.py
	```

	Enter the user's name when prompted. The script collects ~100 face crops per user and appends them to `data/faces_data.pkl` and `data/names.pkl`.

2. Run recognition and mark attendance

	Ensure `background.png` exists in the project root (used by `test.py`). Then run:

	```bash
	python test.py
	```

	- The camera window shows recognized names above detected faces.
	- Press `o` to take attendance for the currently detected person(s). Attendance is appended to `Attendance/Attendance_DD-MM-YYYY.csv`.
	- Press `q` to quit.

	Notes:
	- `test.py` uses Windows SAPI (`pywin32`) for voice feedback. Remove or change the `speak()` function for non-Windows systems.
	- If `data/faces_data.pkl` or `data/names.pkl` are missing or empty, run `add_faces.py` first.

3. View attendance with Streamlit

	```bash
	python -m streamlit run app.py
	```

	The Streamlit app reads today's attendance CSV from the `Attendance/` folder and displays it.

## Troubleshooting

- Haar cascade not loaded: verify `data/haarcascade_frontalface_default.xml` exists.
- `background.png` missing: create or place an image named `background.png` in the project root or update `test.py` to use a different background.
- Webcam not opening: ensure no other application is using the camera and that the correct camera index is used (`cv2.VideoCapture(0)`).

## Notes & Next Steps

- This project uses a simple KNN on flattened color crops (50x50). For production-quality recognition, consider feature embeddings (FaceNet, MobileFaceNet) and proper dataset management.
- Optionally add a `requirements.txt` and a small setup script to automate environment creation.



# face_recognition_project