import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
from collections import deque

# Load trained model
model = load_model('age_regression_model.h5')
img_size = (64, 64)

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print('Error: Could not open webcam.')
    exit()

age_history = deque(maxlen=10)  # Averages over the last 10 frames

while True:
    ret, frame = cap.read()
    if not ret:
        print('Error: Failed to capture frame.')
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        face_img = frame[y:y+h, x:x+w]
        try:
            img = Image.fromarray(face_img).convert('RGB').resize(img_size)
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            pred_age = model.predict(img_array)[0][0]
            age_history.append(pred_age)
            avg_age = sum(age_history) / len(age_history)
            label = f"Age: {int(avg_age)}"
        except Exception as e:
            label = 'Age: N/A'
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow('Age Regression Webcam', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows() 