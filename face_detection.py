import os
import numpy as np
from datetime import datetime

# Ensure setuptools is installed
try:
    import importlib.metadata  # Newer replacement for pkg_resources
except ModuleNotFoundError:
    print("Error: importlib.metadata module is missing!")
    print("Fix it with this command:")
    print("pip3 install --upgrade pip setuptools")
    exit()

# Check for OpenCV (cv2) module
try:
    import cv2
except ModuleNotFoundError:
    print("Error: OpenCV (cv2) module is missing!")
    print("Fix it with this command:")
    print("pip3 install opencv-python")
    print("If you're on macOS or Raspberry Pi, use:")
    print("pip3 install opencv-python-headless")
    exit()

# Check for face_recognition module
try:
    import face_recognition
except ModuleNotFoundError:
    print("Error: face_recognition module is missing!")
    print("Fix it with the following commands:")
    print("\n1. Install Xcode Command Line Tools: xcode-select --install")
    print("2. Install dependencies: brew install cmake libomp boost boost-python3")
    print("3. Install dlib manually: pip3 install dlib --no-cache-dir")
    print("4. Install face_recognition: pip3 install face_recognition --no-cache-dir")
    exit()

# Check for face_recognition_models dependency
try:
    import face_recognition_models
except ModuleNotFoundError:
    print("Error: face_recognition_models package is missing!")
    print("Fix it with this command:")
    print("pip3 install git+https://github.com/ageitgey/face_recognition_models")
    exit()

# Force face_recognition to use the correct models path
os.environ['FACE_RECOGNITION_MODEL_PATH'] = face_recognition_models.__path__[0]

# Check for pytesseract module
try:
    import pytesseract
except ModuleNotFoundError:
    print("Error: pytesseract module is missing!")
    print("Fix it with this command:")
    print("pip3 install pytesseract")
    print("Also, install Tesseract OCR on macOS with:")
    print("brew install tesseract")
    exit()

# Simple Face Detection Mode with Face Saving

KNOWN_FACES_DIR = "detected_faces"
os.makedirs(KNOWN_FACES_DIR, exist_ok=True)
known_face_encodings = []

def get_camera_index():
    for index in range(5):  # Check first 5 camera indices
        cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
        if cap.isOpened():
            cap.release()
            return index
    return 0  # Default to 0 if no valid camera found

def detect_faces():
    print("Starting Face Detection. Press 'q' to exit.")
    video_capture = cv2.VideoCapture(get_camera_index(), cv2.CAP_AVFOUNDATION)
    if not video_capture.isOpened():
        print("Error: Could not open webcam.")
        return
    
    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("Error: Could not read frame from camera.")
            break
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            
            if not any(matches):  # New face detected
                known_face_encodings.append(face_encoding)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                face_img = frame[top:bottom, left:right]
                face_filename = os.path.join(KNOWN_FACES_DIR, f"face_{timestamp}.jpg")
                cv2.imwrite(face_filename, face_img)
                print(f"New face detected and saved: {face_filename}")
            
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        
        cv2.imshow("Face Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    video_capture.release()
    cv2.destroyAllWindows()

# Run face detection mode
detect_faces()
