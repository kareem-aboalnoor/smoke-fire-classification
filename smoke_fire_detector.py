import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import joblib
import pygame

# Initialize pygame mixer for sound
pygame.mixer.init()

# Load alarm sound
pygame.mixer.music.load("mixkit-facility-alarm-sound-999.wav")

# Load model and label binarizer
model = load_model("smoke_fire_model.h5")
lb = joblib.load("label_binarizer.pkl")
classes = lb.classes_

# Open video
cap = cv2.VideoCapture("21985-323496013_medium.mp4")

# Get original FPS
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Display size
display_width = 800
display_height = 600

# Sound control
is_alarm_playing = False

# Frame counter for skip
frame_count = 0
skip_frames = 3  # Process every 3rd frame only

# Variables to store last prediction
last_label = "normal"
last_percent = 0.0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    
    # Only process every nth frame
    if frame_count % skip_frames == 0:
        # Prepare frame for model
        img = cv2.resize(frame, (224, 224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img_to_array(img)
        img = preprocess_input(img)
        img = np.expand_dims(img, axis=0)

        # Predict
        predictions = model.predict(img)[0]
        max_idx = np.argmax(predictions)
        last_label = classes[max_idx].lower()
        last_percent = predictions[max_idx] * 100
    
    # Use last prediction for display
    label = last_label
    percent = last_percent
    
    # Set color and manage alarm
    if label == "normal":
        color = (0, 255, 0)  # Green
        show_warning = False
        if is_alarm_playing:
            pygame.mixer.music.stop()
            is_alarm_playing = False
    elif label == "fire" or label == "smoke":
        color = (0, 0, 255)  # Red
        show_warning = True
        if not is_alarm_playing:
            pygame.mixer.music.play(-1)
            is_alarm_playing = True
    
    # Resize and display
    display_frame = cv2.resize(frame, (display_width, display_height))
    
    # Show detection text
    text = f"{label} ({percent:.2f}%)"
    cv2.putText(display_frame, text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    
    # Show WARNING on right side for fire/smoke
    if show_warning:
        cv2.putText(display_frame, "WARNING", (display_width - 150, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    
    cv2.imshow("Fire & Smoke Detection", display_frame)
    
    # Adjust wait time for smooth playback
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()