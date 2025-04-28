import cv2
import numpy as np
import os
from typing import List, Tuple, Optional, Dict
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor
import shutil

# Helper function for creating asyncio tasks
def create_task(coro):
    """Create a task that will run in the asyncio event loop"""
    loop = asyncio.get_event_loop()
    if loop.is_running():
        return asyncio.create_task(coro)
    else:
        return asyncio.ensure_future(coro)

class FaceRecognition:
    def __init__(self, database_dir: str = "faces/", use_gpu: bool = True):
        self.database_dir = database_dir
        self.faces_dir = os.path.join(database_dir, "faces")
        self.metadata_file = os.path.join(database_dir, "metadata.json")
        os.makedirs(self.faces_dir, exist_ok=True)
        
        # GPU usage configuration
        self.use_gpu = use_gpu
        self.has_gpu = False
        self.gpu_device_id = 0
        
        # Check if GPU (CUDA) is available if requested
        if self.use_gpu:
            try:
                # Check if OpenCV was built with CUDA support
                cuda_devices = cv2.cuda.getCudaEnabledDeviceCount()
                if cuda_devices > 0:
                    self.has_gpu = True
                    print(f"✅ GPU acceleration enabled with {cuda_devices} CUDA device(s)")
                    # Initialize CUDA device
                    cv2.cuda.setDevice(self.gpu_device_id)
                    # Create a CUDA Stream for asynchronous operations
                    self.cuda_stream = cv2.cuda.Stream()
                else:
                    self.has_gpu = False
                    self.use_gpu = False
                    print("⚠️ GPU acceleration requested but no CUDA devices found. Falling back to CPU.")
                    print("   Please check if your GPU drivers are properly installed.")
                    print("   For NVIDIA GPUs, run 'nvidia-smi' to verify driver status.")
            except Exception as e:
                self.has_gpu = False
                self.use_gpu = False
                print(f"⚠️ Error initializing GPU: {e}. Falling back to CPU.")
                print("   Your OpenCV installation might not be compiled with CUDA support.")
                print("   Try running: python -c \"import cv2; print(cv2.getBuildInformation())\" | grep -i cuda")
                print("   to check if CUDA modules are available in your OpenCV build.")
        
        # Initialize face detector
        self.face_detector = cv2.CascadeClassifier("/home/wudado/opencv/data/haarcascades_cuda/haarcascade_frontalface_default.xml")
        
        # Initialize GPU-accelerated face detector if available
        if self.has_gpu:
            try:
                # For GPU, we can use CUDA-accelerated functions
                # Note: OpenCV doesn't have a direct CUDA CascadeClassifier, but we can accelerate other parts
                print("✅ GPU acceleration ready for image processing operations")
            except Exception as e:
                print(f"⚠️ Could not initialize GPU acceleration: {e}. Using CPU only.")
                self.has_gpu = False  # Disable GPU if initialization fails
                self.use_gpu = False
        
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
                        if isinstance(person, dict) and 'name' in person:
                            name = person['name']
                            metadata_dict[name] = person
                            
                            # Ensure features field exists - convert from photos if needed
                            if 'features' not in person:
                                if 'photos' in person:
                                    # Copy photos field to features for backward compatibility
                                    person['features'] = person['photos'].copy()
                                    print(f"Copied {len(person['photos'])} photos to features for {name}")
                                else:
                                    # Initialize empty features list
                                    person['features'] = []
                    
                    # Debug output after loading
                    for name, person_data in metadata_dict.items():
                        if 'features' in person_data:
                            print(f"📂 LOADED metadata for {name} with {len(person_data['features'])} features: {person_data['features']}")
                    
                    print(f"✅ Loaded metadata for {len(metadata_dict)} persons")
                    return metadata_dict
        except Exception as e:
            print(f"❌ Error loading metadata: {e}")
            import traceback
            traceback.print_exc()  # Print full stack trace
        
        return {}

    def _save_metadata(self):
        """Save metadata about registered faces"""
        try:
            print(self.metadata)
            # Before saving, ensure all fields are properly structured
            for name, person_data in self.metadata.items():
                if 'name' not in person_data:
                    person_data['name'] = name
                    
                # Convert any 'photos' to 'features' for consistency
                if 'photos' in person_data and 'features' not in person_data:
                    person_data['features'] = person_data['photos']
                elif 'features' not in person_data:
                    person_data['features'] = []
            
            # Debug output before saving
            for name, person_data in self.metadata.items():
                if 'features' in person_data:
                    print(f"💾 SAVING metadata for {name} with {len(person_data['features'])} features: {person_data['features']}")
            
            # Convert dictionary to list for saving
            metadata_list = list(self.metadata.values())
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.metadata_file), exist_ok=True)
            
            # Write directly to the file
            with open(self.metadata_file, 'w') as f:
                json.dump(metadata_list, f, indent=4)
                f.flush()  # Force flush to disk
                os.fsync(f.fileno())  # Ensure it's written to disk
            
            print(f"✅ Saved metadata for {len(metadata_list)} persons")
        except Exception as e:
            print(f"❌ Error saving metadata: {e}")
            import traceback
            traceback.print_exc()  # Print full stack trace

    def _detect_face(self, gray_image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Wrapper function for face detection with specific parameters"""
        # Use GPU acceleration if available
        if self.has_gpu:
            try:
                # Upload image to GPU memory
                gpu_gray = cv2.cuda_GpuMat()
                gpu_gray.upload(gray_image)
                
                # Apply CUDA-accelerated operations for preprocessing
                # Use GPU-accelerated image equalization to improve detection
                gpu_equalized = cv2.cuda.equalizeHist(gpu_gray)
                
                # Since OpenCV's GPU CascadeClassifier is not fully compatible,
                # we'll download back to CPU for the actual detection
                equalized_gray = gpu_equalized.download()
                
                # Perform detection on the preprocessed image
                faces = self.face_detector.detectMultiScale(
                    equalized_gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(30, 30)
                )
                
                print(f"GPU-accelerated face detection found {len(faces)} faces")
                return faces
            except Exception as e:
                print(f"Error in GPU face detection: {e}. Falling back to CPU.")
                # Fall back to CPU if GPU detection fails
        
        # Standard CPU detection
        faces = self.face_detector.detectMultiScale(
            gray_image,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        print(f"CPU face detection found {len(faces)} faces")
        return faces

    async def _process_face(self, feature_path: str, name: str, current_label: int) -> Optional[Tuple[np.ndarray, int, str]]:
        """Process a single face feature file asynchronously"""
        if not os.path.exists(feature_path):
            print(f"Feature file not found: {feature_path}")
            return None
            
        # Run CPU-intensive tasks in thread pool
        loop = asyncio.get_event_loop()
        try:
            # Load the feature vector directly from .npy file
            face_features = await loop.run_in_executor(
                self.thread_pool,
                np.load,
                feature_path
            )
            
            return face_features, current_label, name
        except Exception as e:
            print(f"Failed to load feature file: {feature_path}, error: {e}")
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
        """Load all face feature vectors from the database directory asynchronously"""
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
                for feature_file in person_data['features']:
                    feature_path = os.path.join(self.faces_dir, name, feature_file)
                    print(f"Processing feature file: {feature_path}")
                    result = await self._process_face(feature_path, name, current_label)
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
                # self.metadata = fresh_metadata.copy()
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
            self.loading_task = create_task(self._load_database())
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
        
        # Use GPU acceleration if available
        if self.has_gpu:
            try:
                # Process on GPU directly without using thread pool
                # Upload to GPU
                gpu_face = cv2.cuda_GpuMat()
                gpu_face.upload(face)
                
                # Resize on GPU
                gpu_resized = cv2.cuda.resize(gpu_face, (100, 100))
                
                # Equalize histogram on GPU
                gpu_equalized = cv2.cuda.equalizeHist(gpu_resized)
                
                # Download result back to CPU
                processed_face = gpu_equalized.download()
                return processed_face
            except Exception as e:
                print(f"Error in GPU face preprocessing: {e}. Falling back to CPU.")
                # Fall back to CPU if GPU processing fails
        
        # Standard CPU processing with thread pool
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

        # Convert to grayscale (using GPU if available)
        if self.has_gpu:
            try:
                # Upload to GPU
                gpu_image = cv2.cuda_GpuMat()
                gpu_image.upload(image)
                
                # Convert to grayscale on GPU
                gpu_gray = cv2.cuda.cvtColor(gpu_image, cv2.COLOR_BGR2GRAY)
                
                # Download result
                gray = gpu_gray.download()
            except Exception as e:
                print(f"Error in GPU grayscale conversion: {e}. Falling back to CPU.")
                # Fall back to CPU
                loop = asyncio.get_event_loop()
                gray = await loop.run_in_executor(
                    self.thread_pool,
                    cv2.cvtColor,
                    image,
                    cv2.COLOR_BGR2GRAY
                )
        else:
            # CPU conversion
            loop = asyncio.get_event_loop()
            gray = await loop.run_in_executor(
                self.thread_pool,
                cv2.cvtColor,
                image,
                cv2.COLOR_BGR2GRAY
            )
        
        # Detect faces (already GPU-accelerated in the _detect_face method if possible)
        faces = self._detect_face(gray)
        if len(faces) == 0:
            return []
            
        # Process each face concurrently
        tasks = []
        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            # Create task for face preprocessing and prediction
            task = create_task(self._process_single_face(face, (x, y, w, h)))
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

    async def _extract_face_features(self, face: np.ndarray) -> np.ndarray:
        """Extract features from a face image"""
        # The LBPH face recognizer already extracts features internally
        # This method is added for extensibility in the future
        # For now, we just return the preprocessed face image
        return await self._preprocess_face(face)

    def add_face(self, image: np.ndarray, name: str, angle: str) -> bool:
        """Add a new face feature vector to the database but don't save to metadata yet"""
        try:
            # Find the next available ID for this person
            next_id = 0
            
            # Ensure person directory exists
            person_dir = os.path.join(self.faces_dir, name)
            os.makedirs(person_dir, exist_ok=True)
            
            # Use set to store features to automatically prevent duplicates
            features_set = set()
            
            # Add existing features to set
            if name in self.metadata and 'features' in self.metadata[name]:
                existing_features = self.metadata[name]['features']
                features_set.update(existing_features)
                print(f"📊 Added {len(existing_features)} existing features to set")
                
                # Find max ID by parsing existing filenames in metadata
                for feature_file in existing_features:
                    try:
                        # Extract ID from filename format name_ID.npy
                        feature_id = int(feature_file.split('_')[1].split('.')[0])
                        next_id = max(next_id, feature_id + 1)
                    except (IndexError, ValueError):
                        # Skip files with non-standard naming
                        pass
            
            # Also check actual files in the directory
            if os.path.exists(person_dir):
                dir_files = os.listdir(person_dir)
                for filename in dir_files:
                    if filename.endswith('.npy') and filename.startswith(f"{name}_"):
                        # Add file to set if not already there
                        features_set.add(filename)
                        
                        try:
                            # Extract ID from filename format name_ID.npy
                            feature_id = int(filename.split('_')[1].split('.')[0])
                            next_id = max(next_id, feature_id + 1)
                        except (IndexError, ValueError):
                            # Skip files with non-standard naming
                            pass
            
            print(f"📊 Using next ID: {next_id} for person {name}")
            
            # Generate filename with sequential ID for feature vector
            feature_filename = f"{name}_{next_id}.npy"
            feature_path = os.path.join(self.faces_dir, name, feature_filename)
            
            # Convert image to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Detect face
            faces = self._detect_face(gray)
            if len(faces) == 0:
                print("No face detected in the image")
                return False
                
            # Extract the face from the image
            x, y, w, h = faces[0]
            face = gray[y:y+h, x:x+w]
            
            # Resize face to standard size
            face = cv2.resize(face, (100, 100))
            
            # Preprocess the face directly instead of using async method
            # This avoids event loop issues
            face_features = cv2.equalizeHist(face)
            
            # Save the feature vector to disk
            np.save(feature_path, face_features)
            print(f"✅ Saved new face feature vector: {feature_path}")
            
            # Add new feature to set
            features_set.add(feature_filename)
            print(f"📊 Added new feature {feature_filename} to set. Total features: {len(features_set)}")
            
            # Update metadata in memory only, but don't save to disk yet
            if name not in self.metadata:
                # Create a new entry if person doesn't exist
                self.metadata[name] = {'name': name, 'features': []}
                print(f"Created new metadata entry for {name}")
            else:
                # Ensure features list exists if not
                if 'features' not in self.metadata[name]:
                    if 'photos' in self.metadata[name]:
                        # Backward compatibility - copy photos to features
                        self.metadata[name]['features'] = self.metadata[name]['photos'].copy()
                        print(f"Copied {len(self.metadata[name]['photos'])} photos to features for backward compatibility")
                    else:
                        self.metadata[name]['features'] = []
                        print(f"Created new features list for {name}")
            
            # Update in-memory metadata but don't save to disk yet
            # Save only the list of features in memory for now
            self.metadata[name]['features'] = list(features_set)
            print(f"📊 Updated in-memory metadata with {len(features_set)} features")
            
            # Return the filename of the newly added feature for later confirmation
            return feature_filename
        except Exception as e:
            print(f"Error adding face: {e}")
            import traceback
            traceback.print_exc()  # Print full stack trace
            return False
            
    def finalize_face_addition(self, name: str, feature_filename: str) -> bool:
        """Save metadata to disk after face is confirmed (OK button pressed)"""
        try:
            # Verify that the feature was actually added to memory
            if (name not in self.metadata or 
                'features' not in self.metadata[name] or 
                feature_filename not in self.metadata[name]['features']):
                print(f"⚠️ Feature {feature_filename} not found in memory for {name}")
                return False
            
            # Check if this is a new user (only one feature) or an existing user with additional photos
            # If there's only one feature, this is likely a new user being added for the first time
            is_new_user = len(self.metadata[name]['features']) == 1
                
            # Only save metadata to disk if this is a new user
            if is_new_user:
                print(f"💾 Saving metadata to disk for new user {name}")
                self._save_metadata()
                print(f"✅ Metadata saved successfully for new user {name}")
            else:
                print(f"ℹ️ Skip saving metadata for existing user {name}")
            
            # Start retraining process
            try:
                print("Starting synchronous retraining...")
                # Use a direct synchronous approach to force immediate retraining
                # Pass save_metadata=True for new users to ensure metadata is saved after training
                self._sync_retrain_model(save_metadata=is_new_user)
                print("Completed model retraining after adding new face")
            except Exception as e:
                print(f"Error during synchronous retraining: {e}")
                # Fall back to async approach if sync fails
                try:
                    print("Falling back to async retraining...")
                    self.loading_task = asyncio.ensure_future(self._retrain_model(save_metadata=True))
                except Exception as inner_e:
                    print(f"Failed to create async retraining task: {inner_e}")
            
            return True
        except Exception as e:
            print(f"Error finalizing face addition: {e}")
            import traceback
            traceback.print_exc()  # Print full stack trace
            return False
            
    def _sync_retrain_model(self, save_metadata=False):
        """Synchronous version of retraining to ensure it completes immediately"""
        if not self.metadata:
            print("No metadata available for retraining")
            return
            
        print("Starting synchronous model retraining...")
        
        try:
            # Manually perform the steps from _load_database in a synchronous way
            faces = []
            labels = []
            label_dict = {}
            current_label = 0
            
            # Process all faces in the database
            for name, person_data in self.metadata.items():
                print(f"Processing person: {name}")
                if 'features' in person_data and len(person_data['features']) > 0:
                    features_list = person_data['features']
                    print(f"Found {len(features_list)} features for person {name}")
                    
                    # Assign a unique label for this person (not each photo)
                    person_label = current_label
                    current_label += 1
                    label_dict[person_label] = name
                    
                    for feature_file in features_list:
                        feature_path = os.path.join(self.faces_dir, name, feature_file)
                        print(f"Processing feature file: {feature_path}")
                        
                        if os.path.exists(feature_path):
                            try:
                                # Load feature directly
                                face_features = np.load(feature_path)
                                faces.append(face_features)
                                # Use the same label for all features of the same person
                                labels.append(person_label)
                                print(f"Added face for {name} with label {person_label}")
                            except Exception as e:
                                print(f"Error loading feature file {feature_path}: {e}")
                else:
                    print(f"No features found for person: {name}")
            
            # If we have faces, train the model
            if faces:
                faces_array = np.array(faces)
                labels_array = np.array(labels)
                
                print(f"Training recognizer with {len(faces)} faces")
                print(f"Faces array shape: {faces_array.shape}")
                print(f"Labels array shape: {labels_array.shape}")
                
                # Debug assertions to detect problems
                if len(faces_array) != len(labels_array):
                    print("ERROR: Mismatch between faces and labels array lengths")
                    return
                
                if len(faces_array) == 0:
                    print("ERROR: No faces to train on")
                    return
                
                # Train directly - this is synchronous
                self.face_recognizer.train(faces_array, labels_array)
                print("Face recognizer training completed")
                
                # Update all in-memory data structures
                self.label_dict = label_dict.copy()  # Make a copy to avoid reference issues
                self.known_face_names = list(set(label_dict.values()))  # Update known face names
                self.known_face_encodings = faces.copy()  # Update face encodings
                
                self.is_trained = True
                print(f"Training completed with {len(faces)} faces for {len(self.known_face_names)} people")
                print(f"Label dictionary: {self.label_dict}")
                print(f"Known face names: {self.known_face_names}")
                
                # Only save metadata if explicitly requested (e.g., after new user added)
                if save_metadata:
                    print("Saving metadata after retraining as requested...")
                    self._save_metadata()
            else:
                print("No faces were successfully processed")
                self.is_trained = False
        except Exception as e:
            print(f"Error during synchronous model retraining: {e}")
            import traceback
            traceback.print_exc()  # Print full stack trace
            self.is_trained = False

    async def add_face_and_wait(self, image: np.ndarray, name: str, angle: str = "") -> bool:
        """Add a new face photo to the database and wait for retraining to complete"""
        try:
            # Check if this is a new user (not in metadata yet)
            is_new_user = name not in self.metadata
            
            # First add the face normally
            feature_filename = self.add_face(image, name, angle)
            if not feature_filename:
                return False
                
            # Finalize the face addition (as if OK was pressed)
            if not self.finalize_face_addition(name, feature_filename):
                print(f"Failed to finalize face addition for {name}")
                return False
            
            # Check if we already have a loading task
            if self.loading_task and not self.loading_task.done():
                print("Waiting for existing retraining task to complete...")
                try:
                    # Wait for the existing task
                    await self.loading_task
                    print("Existing retraining task completed")
                    
                    # If this was a new user, make sure metadata is saved
                    if is_new_user:
                        self._save_metadata()
                        print(f"✅ Saved metadata for new user {name} after training completed")
                    
                    return True
                except Exception as e:
                    print(f"Error waiting for existing task: {e}")
                
            # No task was created or it failed - force retraining and wait for it to complete
            print("Forcing retraining and waiting for it to complete...")
            await self._retrain_model(save_metadata=True)
            
            # If this was a new user, make sure metadata is saved
            if is_new_user:
                self._save_metadata()
                print(f"✅ Saved metadata for new user {name} after model retraining")
                
            print("Model retraining completed after adding new face")
            
            return True
        except Exception as e:
            print(f"Error adding face and waiting: {e}")
            return False

    async def _retrain_model(self, save_metadata=False):
        """Retrain the model with the current metadata (after a face addition or deletion)"""
        if not self.metadata:
            print("No metadata available for retraining")
            return
            
        print("Starting model retraining...")
        try:
            # We'll simply reload the database, which will reconstruct the data and train
            await self._load_database()
            
            # Only save metadata if explicitly requested
            if save_metadata:
                print("Saving metadata after retraining as requested...")
                self._save_metadata()
                
            print("Model retrained successfully after face modification")
        except Exception as e:
            print(f"Error during model retraining: {e}")
            self.is_trained = False

    def delete_face(self, name: str) -> bool:
        """Delete a face from the database"""
        try:
            # Check if name exists
            if name not in self.known_face_names:
                print(f"Face '{name}' not found in database")
                return False
                
            # Remove the person's directory from disk
            person_dir = os.path.join(self.faces_dir, name)
            if os.path.exists(person_dir):
                shutil.rmtree(person_dir)
                print(f"Removed directory: {person_dir}")
                
            # Remove from in-memory data
            if name in self.known_face_names:
                self.known_face_names.remove(name)
                
            # Remove from label_dict
            new_label_dict = {}
            for label, label_name in self.label_dict.items():
                if label_name != name:
                    new_label_dict[label] = label_name
            self.label_dict = new_label_dict
                
            # Remove from metadata
            if name in self.metadata:
                del self.metadata[name]
                
            # Save the updated metadata - we must do this for deletions
            self._save_metadata()
            print(f"✅ Removed {name} from metadata")
                
            # Start retraining process with the remaining faces
            try:
                print("Starting synchronous retraining after deletion...")
                # Use a direct synchronous approach to force immediate retraining
                self._sync_retrain_model()
                print("Completed model retraining after face deletion")
            except Exception as e:
                print(f"Error during synchronous retraining: {e}")
                # Fall back to async approach if sync fails
                try:
                    print("Falling back to async retraining...")
                    self.loading_task = asyncio.ensure_future(self._retrain_model(save_metadata=True))
                except Exception as inner_e:
                    print(f"Failed to create async retraining task: {inner_e}")
                
            print(f"Face '{name}' successfully deleted")
            return True
        except Exception as e:
            print(f"Error deleting face: {e}")
            return False

    def get_photos_count(self, name: str) -> int:
        """Get the number of registered photos for a person"""
        if name in self.metadata:
            if 'features' in self.metadata[name]:
                return len(self.metadata[name]['features'])
            elif 'photos' in self.metadata[name]:
                # Backward compatibility
                return len(self.metadata[name]['photos'])
        return 0

    def get_photos_for_person(self, name: str) -> List[str]:
        """Get list of all photo filenames for a person"""
        if name in self.metadata:
            if 'features' in self.metadata[name]:
                return self.metadata[name]['features']
            elif 'photos' in self.metadata[name]:
                # Backward compatibility
                return self.metadata[name]['photos']
        return []

    def cleanup(self):
        """Clean up resources used by the face recognition system"""
        try:
            print("Cleaning up face recognition resources...")
            # Shutdown thread pool if it exists
            if hasattr(self, 'thread_pool'):
                self.thread_pool.shutdown(wait=False)
                print("Thread pool shutdown completed")
                
            # Cancel any pending loading task
            if self.loading_task and not self.loading_task.done():
                self.loading_task.cancel()
                print("Pending loading task cancelled")
                
            self.is_loading = False
            print("Face recognition cleanup completed")
        except Exception as e:
            print(f"Error during face recognition cleanup: {e}") 
