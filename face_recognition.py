import cv2
import numpy as np
import os
from typing import List, Tuple, Optional, Dict
import json

class FaceRecognition:
    def __init__(self, database_dir: str = "faces/"):
        self.database_dir = database_dir
        self.faces_dir = os.path.join(database_dir, "faces")
        self.metadata_file = os.path.join(database_dir, "metadata.json")
        os.makedirs(self.faces_dir, exist_ok=True)
        
        # Initialize face detector
        self.face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        
        # Initialize face recognizer
        self.face_recognizer = cv2.face.LBPHFaceRecognizer_create()
        
        # Load face database
        self.known_face_encodings: List[np.ndarray] = []
        self.known_face_names: List[str] = []
        self.metadata = self._load_metadata()
        self._load_database()

    def _load_metadata(self) -> Dict:
        """Load metadata about registered faces"""
        if os.path.exists(self.metadata_file):
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {}

    def _save_metadata(self):
        """Save metadata about registered faces"""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f)

    def _load_database(self):
        """Load all face encodings from the database directory"""
        faces = []
        labels = []
        label_dict = {}
        current_label = 0
        
        for name, data in self.metadata.items():
            for angle, filename in data['photos'].items():
                image_path = os.path.join(self.faces_dir, filename)
                if not os.path.exists(image_path):
                    continue
                    
                # Read and convert image to grayscale
                image = cv2.imread(image_path)
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                
                # Detect face
                face_locations = self.face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                
                if len(face_locations) > 0:
                    # Use the first face detected
                    x, y, w, h = face_locations[0]
                    face = gray[y:y+h, x:x+w]
                    
                    # Add to training data
                    faces.append(face)
                    labels.append(current_label)
                    label_dict[current_label] = name
                    current_label += 1
        
        if faces:
            # Train the recognizer
            self.face_recognizer.train(faces, np.array(labels))
            self.label_dict = label_dict

    def detect_faces(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces in an image using OpenCV"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        return faces

    def recognize_face(self, image: np.ndarray) -> List[Tuple[str, float, Tuple[int, int, int, int]]]:
        """
        Recognize faces in the image and return list of (name, confidence, face_location) tuples
        Returns: List of tuples (name, confidence, (x, y, w, h)) or empty list if no faces recognized
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.detect_faces(image)
        
        if len(faces) == 0:
            return []
            
        # Store results for each face
        results: List[Tuple[str, float, Tuple[int, int, int, int]]] = []
        
        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            
            try:
                # Predict the face
                label, confidence = self.face_recognizer.predict(face)
                
                # If confidence is too low, skip this face
                if confidence > 30:  # Lower confidence means better match
                    continue
                    
                name = self.label_dict.get(label)
                if name:
                    results.append((name, confidence, (x, y, w, h)))
            except:
                continue
        
        # Sort by confidence (lower is better) and return only the best match
        if results:
            results.sort(key=lambda x: x[1])
            return [results[0]]
        return []

    def add_face(self, image: np.ndarray, name: str, angle: str) -> bool:
        """Add a new face photo to the database"""
        # Generate filename
        filename = f"{name}_{angle}.jpg"
        image_path = os.path.join(self.faces_dir, filename)
        
        # Save the image
        cv2.imwrite(image_path, image)
        
        # Update metadata
        if name not in self.metadata:
            self.metadata[name] = {'photos': {}}
        self.metadata[name]['photos'][angle] = filename
        self._save_metadata()
        
        # Reload the database to include the new face
        self._load_database()
        return True

    def delete_face(self, name: str) -> bool:
        """Delete a face from the database"""
        if name in self.metadata:
            # Delete all photos
            for filename in self.metadata[name]['photos'].values():
                file_path = os.path.join(self.faces_dir, filename)
                if os.path.exists(file_path):
                    os.remove(file_path)
            
            # Remove from metadata
            del self.metadata[name]
            self._save_metadata()
            
            # Reload the database
            self._load_database()
            return True
        return False

    def get_registered_angles(self, name: str) -> List[str]:
        """Get list of registered angles for a person"""
        if name in self.metadata:
            return list(self.metadata[name]['photos'].keys())
        return [] 