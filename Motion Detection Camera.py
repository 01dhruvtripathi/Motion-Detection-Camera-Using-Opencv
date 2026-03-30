import os

# Suppress pygame welcome message
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "1"

import cv2
import numpy as np
import pygame
import threading

# Initialize pygame mixer for sound playback
pygame.mixer.init()
alarm_sound = pygame.mixer.Sound('alarm.mp3.wav')  # Replace with your alarm sound file

# Load Haar cascade classifiers for face and eye detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Initialize the camera
cap = cv2.VideoCapture(0)

# Create a background subtractor object
fgbg = cv2.createBackgroundSubtractorMOG2()

# Flag to indicate if an alarm has been triggered
alarm_triggered = False

def play_alarm():
    while True:
        if alarm_triggered:
            alarm_sound.play()  # Play the alarm sound
            pygame.time.delay(1000)  # Delay to prevent the sound from playing too rapidly
        else:
            alarm_sound.stop()  # Stop the alarm sound if no movement is detected

# Start the alarm thread
alarm_thread = threading.Thread(target=play_alarm)
alarm_thread.daemon = True
alarm_thread.start()

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame. Exiting...")
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    for (x, y, w, h) in faces:
        # Draw a rectangle around the detected face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Region of interest (ROI) for the face
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

        # Detect eyes within the face ROI
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

    # Apply background subtraction for motion detection
    fgmask = fgbg.apply(frame)

    # Threshold the mask to get a binary image
    th = cv2.threshold(fgmask.copy(), 244, 255, cv2.THRESH_BINARY)[1]

    # Find contours in the thresholded image
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw bounding boxes around moving objects
    moving_detected = False
    for contour in contours:
        if cv2.contourArea(contour) < 300:
            continue
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        moving_detected = True

    # Color detection for blue objects
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([100, 150, 0])  # Lower bound for blue color
    upper_blue = np.array([140, 255, 255])  # Upper bound for blue color
    blue_mask = cv2.inRange(hsv_frame, lower_blue, upper_blue)

    # Find contours in the blue mask
    blue_contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    blue_detected = False
    for contour in blue_contours:
        if cv2.contourArea(contour) < 300:
            continue
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)  # Draw blue bounding box
        blue_detected = True

    # Trigger the alarm if a moving object or a blue object is detected
    alarm_triggered = moving_detected or blue_detected

    cv2.imshow('Object and Motion Detection', frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
