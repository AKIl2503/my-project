# Import Libraries
import cv2
import time
import mediapipe as mp

# Grabbing the Holistic Model from Mediapipe and initializing the model
mp_holistic = mp.solutions.holistic
holistic_model = mp_holistic.Holistic(
    min_detection_confidence=0.5,  # Confidence threshold for detection
    min_tracking_confidence=0.5  # Confidence threshold for tracking
)

# Initialize drawing utilities to visualize the landmarks
mp_drawing = mp.solutions.drawing_utils

# Initialize webcam video capture (0 for the default camera)
capture = cv2.VideoCapture(0)

# Initializing time variables for calculating FPS
previous_time = 0
current_time = 0

# Loop to continuously get frames from the webcam
while capture.isOpened():
    # Capture each frame from the webcam
    ret, frame = capture.read()

    if not ret:
        print("Failed to capture image")
        break

    # Resize the frame for better viewing experience
    frame = cv2.resize(frame, (800, 600))

    # Convert the image from BGR (OpenCV default) to RGB (Mediapipe requirement)
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Disable writing to the image to improve performance during processing
    image.flags.writeable = False

    # Perform holistic landmark detection
    results = holistic_model.process(image)

    # Re-enable writing to the image
    image.flags.writeable = True

    # Convert the processed image back to BGR for OpenCV display
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Remove facial landmarks code block

    # Draw right hand landmarks if detected
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            results.right_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS
        )

    # Draw left hand landmarks if detected
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            results.left_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS
        )

    # Calculate the frames per second (FPS)
    current_time = time.time()
    fps = 1 / (current_time - previous_time)
    previous_time = current_time

    # Display FPS on the image
    cv2.putText(image, f'{int(fps)} FPS', (10, 70), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow("Hand Landmarks", image)

    # Press 'q' key to exit the loop
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
capture.release()
cv2.destroyAllWindows()