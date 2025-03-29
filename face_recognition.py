import cv2
import numpy as np
import os
from typing import List, Tuple, Optional, Dict
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor

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
        
        # Initialize thread pool for CPU-intensive tasks
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        
        # Initialize state variables
        self.is_loading = False
        self.loading_task = None
        self.is_trained = False
        
        # Initialize data structures
        self.known_face_encodings: List[np.ndarray] = []
        self.known_face_names: List[str] = []
        self.label_dict = {}
        
        # Load initial metadata
        self.metadata = self._load_metadata()
        print(f"Initialized with metadata: {self.metadata}")
        
        # Start initial database loading if metadata exists
        if self.metadata:
            self.start_loading_database()

    def _load_metadata(self) -> Dict:
        """Load metadata about registered faces"""
        try:
            if os.path.exists(self.metadata_file):
                with open(self.metadata_file, 'r') as f:
                    data = json.load(f)
                    # Convert list to dictionary for easier access
                    metadata_dict = {}
                    for person in data:
                        if isinstance(person, dict) and 'name' in person and 'photos' in person:
                            metadata_dict[person['name']] = person
                    print(f"Loaded metadata for {len(metadata_dict)} persons")
                    return metadata_dict
        except Exception as e:
            print(f"Error loading metadata: {e}")
        return {}

    def _save_metadata(self):
        """Save metadata about registered faces"""
        try:
            # Convert dictionary back to list for saving
            metadata_list = list(self.metadata.values())
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.metadata_file), exist_ok=True)
            with open(self.metadata_file, 'w') as f:
                json.dump(metadata_list, f, indent=4)
                f.flush()  # Force flush to disk
                os.fsync(f.fileno())  # Ensure it's written to disk
            print(f"Saved metadata for {len(metadata_list)} persons")
        except Exception as e:
            print(f"Error saving metadata: {e}")

    def _detect_face(self, gray_image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Wrapper function for face detection with specific parameters"""
        return self.face_detector.detectMultiScale(
            gray_image,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

    async def _process_face(self, image_path: str, name: str, current_label: int) -> Optional[Tuple[np.ndarray, int, str]]:
        """Process a single face image asynchronously"""
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            return None
            
        # Run CPU-intensive tasks in thread pool
        loop = asyncio.get_event_loop()
        image = await loop.run_in_executor(self.thread_pool, cv2.imread, image_path)
        if image is None:
            print(f"Failed to read image: {image_path}")
            return None
            
        # Convert to grayscale
        gray = await loop.run_in_executor(self.thread_pool, cv2.cvtColor, image, cv2.COLOR_BGR2GRAY)
        
        # Equalize histogram for better face detection
        gray = await loop.run_in_executor(self.thread_pool, cv2.equalizeHist, gray)
        
        # Detect face using wrapper function
        face_locations = await loop.run_in_executor(
            self.thread_pool,
            self._detect_face,
            gray
        )
        
        if len(face_locations) > 0:
            x, y, w, h = face_locations[0]
            face = gray[y:y+h, x:x+w]
            
            # Resize face to standard size for better recognition
            face = await loop.run_in_executor(
                self.thread_pool,
                cv2.resize,
                face,
                (100, 100)
            )
            
            return face, current_label, name
        else:
            print(f"No face detected in: {image_path}")
        return None

    async def _train_recognizer(self, faces: List[np.ndarray], labels: List[int], label_dict: Dict[int, str]):
        """Train the face recognizer with given faces and labels"""
        if not faces:
            print("No faces to train on")
            self.is_trained = False
            return
            
        print(f"Training recognizer with {len(faces)} faces")
        faces_array = np.array(faces)
        labels_array = np.array(labels)
        
        print(f"Faces array shape: {faces_array.shape}")
        print(f"Labels array shape: {labels_array.shape}")
        
        # Train the recognizer in thread pool
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            self.thread_pool,
            self.face_recognizer.train,
            faces_array,
            labels_array
        )
        
        # Update all in-memory data structures
        self.label_dict = label_dict.copy()  # Make a copy to avoid reference issues
        self.known_face_names = list(set(label_dict.values()))  # Update known face names
        self.known_face_encodings = faces.copy()  # Update face encodings
        
        self.is_trained = True
        print(f"Training completed with {len(faces)} faces")
        print(f"Label dictionary: {self.label_dict}")
        print(f"Known face names: {self.known_face_names}")

    async def _load_database(self):
        """Load all face encodings from the database directory asynchronously"""
        if self.is_loading:
            print("Database loading already in progress")
            return
            
        self.is_loading = True
        try:
            # Read fresh metadata
            fresh_metadata = self._load_metadata()
            if not fresh_metadata:
                print("No metadata found in database")
                return
                
            print(f"Loading database with metadata: {fresh_metadata}")
            faces = []
            labels = []
            label_dict = {}
            current_label = 0
            
            # Process all faces in the database
            for name, person_data in fresh_metadata.items():
                print(f"Processing person: {name}")
                for photo in person_data['photos']:
                    image_path = os.path.join(self.faces_dir, name, photo)
                    print(f"Processing photo: {image_path}")
                    result = await self._process_face(image_path, name, current_label)
                    if result is not None:
                        face, label, name = result
                        faces.append(face)
                        labels.append(label)
                        label_dict[label] = name
                        print(f"Added face for {name} with label {label}")
                        current_label += 1
            
            # Train with collected faces
            if faces:
                await self._train_recognizer(faces, labels, label_dict)
                self.metadata = fresh_metadata.copy()
                print(f"Database loading completed with {len(faces)} faces")
            else:
                print("No faces were successfully processed")
                self.is_trained = False
        except Exception as e:
            print(f"Error during database loading: {e}")
            self.is_trained = False
        finally:
            self.is_loading = False

    def start_loading_database(self):
        """Start loading the database in the background"""
        if not self.is_loading:
            print("Starting initial database loading...")
            self.loading_task = asyncio.create_task(self._load_database())
            return self.loading_task
        return None

    def detect_faces(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces in an image using OpenCV"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector.detectMultiScale(
            gray,
            scaleFactor=1.05,  # More sensitive scaling
            minNeighbors=3,    # Less strict neighbor requirement
            minSize=(30, 30)
        )
        print(f"Detected {len(faces)} faces")
        return faces

    async def _preprocess_face(self, face: np.ndarray) -> np.ndarray:
        """Preprocess face image for recognition"""
        loop = asyncio.get_event_loop()
        # Resize to standard size
        face = await loop.run_in_executor(
            self.thread_pool,
            cv2.resize,
            face,
            (100, 100)
        )
        # Equalize histogram for better recognition
        face = await loop.run_in_executor(
            self.thread_pool,
            cv2.equalizeHist,
            face
        )
        return face

    async def _predict_face(self, face: np.ndarray) -> Tuple[Optional[int], float]:
        """Predict a single face asynchronously"""
        if not self.is_trained:
            return None, float('inf')
            
        try:
            loop = asyncio.get_event_loop()
            label, confidence = await loop.run_in_executor(
                self.thread_pool,
                self.face_recognizer.predict,
                face
            )
            return label, confidence
        except Exception as e:
            print(f"Error predicting face: {e}")
            return None, float('inf')

    async def recognize_face(self, image: np.ndarray) -> List[Tuple[str, float, Tuple[int, int, int, int]]]:
        """
        Recognize faces in the image and return list of (name, confidence, face_location) tuples
        Returns: List of tuples (name, confidence, (x, y, w, h))
        """
        if not self.is_trained:
            print("Face recognizer is not trained yet")
            return []

        loop = asyncio.get_event_loop()
        # Convert to grayscale in thread pool
        gray = await loop.run_in_executor(
            self.thread_pool,
            cv2.cvtColor,
            image,
            cv2.COLOR_BGR2GRAY
        )
        
        # Detect faces
        faces = self.detect_faces(image)
        if len(faces) == 0:
            return []
            
        # Process each face concurrently
        tasks = []
        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            # Create task for face preprocessing and prediction
            task = asyncio.create_task(self._process_single_face(face, (x, y, w, h)))
            tasks.append(task)
        
        # Wait for all face processing tasks to complete
        results = []
        for completed_task in await asyncio.gather(*tasks):
            if completed_task is not None:
                results.append(completed_task)
        
        # Sort all results by confidence (lower is better)
        results.sort(key=lambda x: x[1])
        return results

    async def _process_single_face(self, face: np.ndarray, face_location: Tuple[int, int, int, int]) -> Optional[Tuple[str, float, Tuple[int, int, int, int]]]:
        """Process a single face for recognition"""
        try:
            # Preprocess face
            processed_face = await self._preprocess_face(face)
            
            # Predict face
            label, confidence = await self._predict_face(processed_face)
            
            if label is not None and confidence <= 100:  # Lower confidence means better match
                name = self.label_dict.get(label)
                if name:
                    print(f"Recognized face: {name} with confidence {confidence}")
                    return (name, confidence, face_location)
                else:
                    print(f"No name found for label {label}")
            else:
                print(f"Face skipped due to low confidence: {confidence}")
        except Exception as e:
            print(f"Error processing face: {e}")
        return None

    def add_face(self, image: np.ndarray, name: str, angle: str) -> bool:
        """Add a new face photo to the database"""
        try:
            # Generate filename
            filename = f"{name}_{angle}.jpg"
            image_path = os.path.join(self.faces_dir, name, filename)
            os.makedirs(os.path.dirname(image_path), exist_ok=True)
            
            # Save the image with flush
            with open(image_path, 'wb') as f:
                cv2.imwrite(image_path, image)
                f.flush()
                os.fsync(f.fileno())
            print(f"Saved new face image: {image_path}")
            
            # Update metadata
            if name not in self.metadata:
                self.metadata[name] = {'name': name, 'photos': []}
            if filename not in self.metadata[name]['photos']:
                self.metadata[name]['photos'].append(filename)
            self._save_metadata()
            print(f"Updated metadata for {name}")
            
            # Reload metadata to ensure we have the latest state
            self.metadata = self._load_metadata()
            print("Reloaded metadata with new user")
            
            # Start retraining process
            if not self.is_loading:
                asyncio.create_task(self._retrain_model())
            
            return True
        except Exception as e:
            print(f"Error adding face: {e}")
            return False

    async def _retrain_model(self):
        """Retrain the model with all faces in the database"""
        try:
            # Mark as loading to prevent concurrent retraining attempts
            if self.is_loading:
                print("Another loading/training process is already running")
                return
                
            self.is_loading = True
            
            # Read the latest metadata from disk
            fresh_metadata = self._load_metadata()
            print(f"Reloaded metadata from disk for retraining: {len(fresh_metadata)} persons")
            
            # Reset all in-memory data structures
            faces = []
            labels = []
            label_dict = {}
            self.known_face_names = []
            self.known_face_encodings = []
            current_label = 0
            
            print("Starting model retraining...")
            
            # Process all faces in the database using the fresh metadata
            for person_name, person_data in fresh_metadata.items():
                print(f"Processing person: {person_name}")
                for photo in person_data['photos']:
                    photo_path = os.path.join(self.faces_dir, person_name, photo)
                    result = await self._process_face(photo_path, person_name, current_label)
                    if result is not None:
                        face, label, name = result
                        faces.append(face)
                        labels.append(label)
                        label_dict[label] = name
                        print(f"Added face for {name} with label {label}")
                        current_label += 1
            
            # Train with all faces
            if faces:
                await self._train_recognizer(faces, labels, label_dict)
                # Update the main metadata reference with the fresh one
                self.metadata = fresh_metadata.copy()  # Make a copy to avoid reference issues
                print(f"Successfully retrained model with {len(faces)} faces")
                print(f"Final label dictionary: {self.label_dict}")
                print(f"Final known face names: {self.known_face_names}")
            else:
                print("No faces to train on")
                self.is_trained = False
                # Reset all data structures if no faces
                self.label_dict = {}
                self.known_face_names = []
                self.known_face_encodings = []
                self.metadata = {}
        except Exception as e:
            print(f"Error during model retraining: {e}")
            self.is_trained = False
            # Reset all data structures on error
            self.label_dict = {}
            self.known_face_names = []
            self.known_face_encodings = []
            self.metadata = {}
        finally:
            # Always release the loading lock
            self.is_loading = False

    def delete_face(self, name: str) -> bool:
        """Delete a face from the database"""
        if name in self.metadata:
            # Delete all photos
            for filename in self.metadata[name]['photos']:
                file_path = os.path.join(self.faces_dir, name, filename)
                if os.path.exists(file_path):
                    os.remove(file_path)
            
            # Remove from metadata
            del self.metadata[name]
            self._save_metadata()
            
            # Start reloading the database
            self.start_loading_database()
            return True
        return False

    def get_registered_angles(self, name: str) -> List[str]:
        """Get list of registered angles for a person"""
        if name in self.metadata:
            return [photo.split('_')[1].split('.')[0] for photo in self.metadata[name]['photos']]
        return [] 