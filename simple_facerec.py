import cv2
import mediapipe as mp
import os
import glob
import numpy as np
import face_recognition

class SimpleFacerec:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []
        self.frame_resizing = 0.25  # Resize frame for faster processing

        # Initialize MediaPipe Face Detection
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(min_detection_confidence=0.7)

        self.encoding_file = 'face_encodings.npy'  # Path to save/load encodings

    def load_encoding_images(self, images_path):
        # Check if encoding file exists
        if os.path.exists(self.encoding_file):
            # Load encodings and names from file
            print(f"Loading face encodings from {self.encoding_file}")
            data = np.load(self.encoding_file, allow_pickle=True)
            self.known_face_encodings = data.item().get('encodings', [])
            self.known_face_names = data.item().get('names', [])
            print(f"Loaded {len(self.known_face_encodings)} encodings from file.")
        else:
            # Otherwise, process the images and create encodings
            print(f"Encoding file not found. Generating encodings from images.")
            person_folders = [os.path.join(images_path, d) for d in os.listdir(images_path) if os.path.isdir(os.path.join(images_path, d))]

            for folder_path in person_folders:
                person_name = os.path.basename(folder_path)
                image_files = glob.glob(os.path.join(folder_path, "*.*"))

                print(f"Found {len(image_files)} images for {person_name}.")
                for img_path in image_files:
                    img = cv2.imread(img_path)
                    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    encodings = face_recognition.face_encodings(rgb_img)
                    if encodings:
                        img_encoding = encodings[0]
                        self.known_face_encodings.append(img_encoding)
                        self.known_face_names.append(person_name)

            # Save the encodings to the file
            print("Saving face encodings to file.")
            encodings_data = {'encodings': self.known_face_encodings, 'names': self.known_face_names}
            np.save(self.encoding_file, encodings_data)

            print(f"All encoding images loaded and saved.")

    def detect_known_faces(self, frame):
        small_frame = cv2.resize(frame, (0, 0), fx=self.frame_resizing, fy=self.frame_resizing)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        face_locations = []
        face_names = []

        results = self.face_detection.process(rgb_small_frame)
        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = rgb_small_frame.shape
                x1 = int(bboxC.xmin * iw)
                y1 = int(bboxC.ymin * ih)
                x2 = int((bboxC.xmin + bboxC.width) * iw)
                y2 = int((bboxC.ymin + bboxC.height) * ih)
                face_locations.append([y1, x2, y2, x1])

                # Extract the full frame region for encoding
                face_roi = rgb_small_frame[y1:y2, x1:x2]
                face_encodings = face_recognition.face_encodings(rgb_small_frame, [(y1, x2, y2, x1)])  # Use the entire frame and pass locations

                if face_encodings:
                    face_encoding = face_encodings[0]
                    matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding , tolerance=0.38)
                    name = "Unknown"
                    face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index] and face_distances[best_match_index] < 0.5:
                        name = self.known_face_names[best_match_index]

                    face_names.append(name)

        face_locations = np.array(face_locations) / self.frame_resizing
        return face_locations.astype(int), face_names