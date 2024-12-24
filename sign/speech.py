import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os
import cv2
import HandDataCollecter
import mediapipe as mp
import numpy as np
import pyttsx3  # Import the text-to-speech library

# Initialize pyttsx3
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Set speech rate
engine.setProperty('volume', 1.0)  # Set volume level
# Initialize Random Forest
local_path = (os.path.dirname(os.path.realpath('__file__')))
file_name = 'data.csv'  # file of total data
data_path = os.path.join(local_path, file_name)
print(data_path)
df = pd.read_csv(r'' + data_path)

print(df)

units_in_data = 28  # no. of units in data

titles = ["unit-" + str(i) for i in range(units_in_data)]
X = df[titles]
y = df['letter']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=2)

clf = RandomForestClassifier(n_estimators=30)  # Random Forest
clf.fit(X_train, y_train)

# Mediapipe Hand Solutions
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands


def get_prediction(image):
    with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
        ImageData = HandDataCollecter.ImageToDistanceData(image, hands)
        DistanceData = ImageData['Distance-Data']
        image = ImageData['image']
        prediction = clf.predict([DistanceData])
        return prediction[0]


if __name__ == '__main__':
    cap = cv2.VideoCapture(0)

    SpelledWord = ""
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        try:
            SpelledWord = get_prediction(image)
            cv2.putText(image, SpelledWord, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (124, 252, 0), 5, cv2.LINE_AA)

            # Convert detected text to speech
            engine.say(SpelledWord)
            engine.runAndWait()

        except Exception as e:
            print(f"Error: {e}")
            pass

        cv2.imshow('frame', image)

        if cv2.waitKey(5) & 0xFF == 27:  # press escape to break
            break

    cap.release()
    cv2.destroyAllWindows()
