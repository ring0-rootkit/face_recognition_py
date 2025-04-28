import sys
import cv2
import numpy as np
import os
from face_recognition import FaceRecognition
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QPushButton, QLabel, QLineEdit, 
                            QFileDialog, QMessageBox, QFrame, QScrollArea, QDialog, QInputDialog, QGridLayout, QComboBox)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QImage, QPixmap, QFont, QIcon, QPalette, QColor
import time
from pathlib import Path
import json
import asyncio

# Wrapper function for asyncio.create_task to ensure proper threading
def create_task(coro):
    """Create a task that will run in the asyncio event loop"""
    loop = asyncio.get_event_loop()
    if loop.is_running():
        # Use the standard asyncio.create_task which will be patched in main()
        return asyncio.create_task(coro)
    else:
        # For tasks created before the event loop is running
        task = asyncio.ensure_future(coro)
        return task

class CameraThread(QThread):
    frame_ready = pyqtSignal(np.ndarray)
    
    def __init__(self, camera_index=0):
        super().__init__()
        self.running = False
        self.camera = None
        self.camera_index = camera_index
        
    def run(self):
        # When using the Android webcam, always use the direct device path
        device_path = f"/dev/video{self.camera_index}"
        print(f"Connecting to camera at {device_path}")
        
        # Try first with specific device path and more robust parameters
        self.camera = cv2.VideoCapture(device_path)
        
        # Configure the camera with specific settings for better Android webcam support
        if self.camera.isOpened():
            # Set specific properties for better compatibility
            self.camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))  # Use MJPG format
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Set resolution
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.camera.set(cv2.CAP_PROP_FPS, 30)  # Set framerate
            
            print(f"✅ Successfully connected to camera at {device_path}")
            self.running = True
            
            while self.running:
                ret, frame = self.camera.read()
                if ret:
                    self.frame_ready.emit(frame)
                    time.sleep(0.03)  # Add small delay to avoid overwhelming the system
        else:
            print(f"❌ Failed to open camera at {device_path}")
            
            # Try using a custom gstreamer pipeline as a last resort
            try:
                gst_pipeline = f"v4l2src device={device_path} ! video/x-raw,format=YUY2,width=640,height=480,framerate=30/1 ! videoconvert ! video/x-raw,format=BGR ! appsink"
                self.camera = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
                
                if self.camera.isOpened():
                    print(f"✅ Successfully connected to camera using GStreamer pipeline")
                    self.running = True
                    
                    while self.running:
                        ret, frame = self.camera.read()
                        if ret:
                            self.frame_ready.emit(frame)
                            time.sleep(0.03)  # Add small delay
                else:
                    print(f"❌ Failed to open camera with GStreamer pipeline")
            except Exception as e:
                print(f"Error trying GStreamer: {e}")
                print("Your Android phone must have a webcam app running and be connected via USB")

    def stop(self):
        self.running = False
        if self.camera:
            self.camera.release()

class ModernButton(QPushButton):
    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 5px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
            QPushButton:pressed {
                background-color: #0D47A1;
            }
            QPushButton:disabled {
                background-color: #BDBDBD;
            }
        """)

class ModernLineEdit(QLineEdit):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("""
            QLineEdit {
                padding: 8px;
                border: 2px solid #424242;
                border-radius: 5px;
                background-color: #424242;
                color: white;
                font-size: 14px;
            }
            QLineEdit:focus {
                border: 2px solid #2196F3;
            }
        """)

class ModernComboBox(QComboBox):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("""
            QComboBox {
                padding: 8px;
                border: 2px solid #424242;
                border-radius: 5px;
                background-color: #424242;
                color: white;
                font-size: 14px;
                min-height: 20px;
            }
            QComboBox:hover {
                border: 2px solid #2196F3;
            }
            QComboBox::drop-down {
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 25px;
                border-left-width: 1px;
                border-left-color: #424242;
                border-left-style: solid;
            }
            QComboBox::down-arrow {
                /* Use no image but define the arrow as a Unicode character */
                image: none;
                color: white;
                width: 12px;
                height: 12px;
            }
            QComboBox QAbstractItemView {
                background-color: #424242;
                color: white;
                selection-background-color: #2196F3;
                selection-color: white;
                border: none;
                outline: none;
                padding: 5px;
            }
            QComboBox QListView {
                background-color: #424242;
                color: white;
            }
            QComboBox QListView::item {
                background-color: #424242;
                color: white;
                padding: 5px;
            }
            QComboBox QListView::item:hover {
                background-color: #616161;
            }
            QComboBox QListView::item:selected {
                background-color: #2196F3;
            }
            QComboBox QScrollBar:vertical {
                border: none;
                background-color: #424242;
                width: 10px;
                margin: 0px;
            }
            QComboBox QScrollBar::handle:vertical {
                background-color: #666666;
                border-radius: 5px;
                min-height: 20px;
            }
            QComboBox QScrollBar::handle:vertical:hover {
                background-color: #888888;
            }
            QComboBox QScrollBar::add-line:vertical, QComboBox QScrollBar::sub-line:vertical {
                height: 0px;
            }
        """)
        
        # Set a proper arrow character
        self.setItemText(-1, "▼")
        
    def showPopup(self):
        """Override to apply dark theme to popup when it appears"""
        super().showPopup()
        
        # Find the view
        popup = self.findChild(QFrame)
        if popup:
            # Apply style to ensure the background is dark
            popup.setStyleSheet("""
                background-color: #424242;
                color: white;
                border: 1px solid #666666;
                border-radius: 5px;
            """)
            
        # Additional styling for the popup window
        view = self.view()
        if view and view.window():
            view.window().setStyleSheet("""
                background-color: #424242;
                border: 1px solid #666666;
            """)

class ModernLabel(QLabel):
    def __init__(self, text="", parent=None):
        super().__init__(text, parent)
        self.setStyleSheet("""
            QLabel {
                color: white;
                font-size: 14px;
            }
        """)

class ModernFrame(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("""
            QFrame {
                background-color: transparent;
                border-radius: 10px;
                padding: 10px;
            }
        """)

class FaceRegistrationDialog(QDialog):
    """Dialog for registering a new face"""
    def __init__(self, parent=None, camera_index=0):
        super().__init__(parent)
        self.setWindowTitle("Register New Face")
        self.setMinimumSize(640, 680)
        self.setModal(True)
        
        # Initialize variables
        self.name = ""
        self.face_recognizer = None
        self.captures = []
        self.is_capture_ready = False
        self.progress = 0
        self.progress_target = 3  # Number of angles to capture
        self.angles = ["Front", "Slight Left", "Slight Right"]
        self.current_angle_index = 0
        self.detected_faces = []
        self.camera_thread = None
        self.is_photo_taken = False
        self.allow_photo = False
        self.progress_timer = None
        self.stored_features = []
        self.camera_index = camera_index
        
        # Create layout
        main_layout = QVBoxLayout(self)
        
        # Add title
        title = QLabel("Face Registration")
        title.setStyleSheet("font-size: 18px; font-weight: bold; margin-bottom: 10px;")
        main_layout.addWidget(title)
        
        # Add instructions
        self.instructions = QLabel("Move your face in different positions for training.")
        self.instructions.setWordWrap(True)
        self.instructions.setStyleSheet("margin-bottom: 10px; color: white;")
        main_layout.addWidget(self.instructions)
        
        # Add image display
        self.image_label = QLabel()
        self.image_label.setMinimumSize(320, 240)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet("""
            QLabel {
                background-color: #2b2b2b;
                border-radius: 10px;
            }
        """)
        main_layout.addWidget(self.image_label)
        
        # Progress label
        self.progress_label = QLabel()
        self.progress_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.progress_label.setStyleSheet("margin-top: 10px; color: white;")
        main_layout.addWidget(self.progress_label)
        
        # Cancel button
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setStyleSheet("""
            QPushButton {
                background-color: #9e9e9e;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #757575;
            }
        """)
        self.cancel_button.clicked.connect(self.reject)
        main_layout.addWidget(self.cancel_button)
        
        self.setLayout(main_layout)
        
        # Initialize variables
        self.name = None
        self.current_image = None
        self.display_image_with_rect = None  # Image with rectangle overlay
        self.taken_photos = 0
        self.face_detector = cv2.CascadeClassifier("/home/wudado/opencv/data/haarcascades_cuda/haarcascade_frontalface_default.xml")

        
        # Recognition variables
        self.green_start_time = 0  # When the face started being recognized (green)
        self.required_recognition_time = 5.0  # Seconds of consecutive recognition required
        self.is_recognized = False  # Current recognition state
        self.last_face_location = None  # Track last detected face location
        self.original_face_location = None  # Original face coordinates in the full image
        
        # Timers
        self.process_timer = QTimer()  # Timer for processing frames
        self.process_timer.timeout.connect(self.process_current_frame)
        self.process_timer.setInterval(100)  # Process frames every 100ms
        
        self.photo_timer = QTimer()  # Timer for taking photos
        self.photo_timer.timeout.connect(self.check_take_photo)
        self.photo_timer.setInterval(500)  # Take photos every 500ms
        
        # Flag to track if we've shown a completion dialog
        self.completion_dialog_shown = False
        
    def set_name(self, name):
        """Set the name for the face being registered"""
        self.name = name
        self.person_dir = Path("faces/faces") / name
        self.person_dir.mkdir(parents=True, exist_ok=True)
        self.update_progress()
        
    def update_progress(self):
        """Update the progress label"""
        if self.is_recognized:
            time_left = max(0, self.required_recognition_time - (time.time() - self.green_start_time))
            self.progress_label.setText(f"Face recognized! Keep position for {time_left:.1f} seconds")
            self.progress_label.setStyleSheet("margin-top: 10px; color: #00CC00; font-weight: bold;")  # Darker green for better readability
        else:
            self.progress_label.setText(f"Photos taken: {self.taken_photos} - Move your face around")
            self.progress_label.setStyleSheet("margin-top: 10px; color: white;")
            
    def set_image(self, image):
        """Set the current camera image"""
        self.current_image = image.copy()
        
        # Detect faces immediately to crop image to face area
        if self.current_image is not None:
            gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
            faces = self.face_detector.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            
            if len(faces) == 1:
                # Store original face location for taking photos
                self.original_face_location = faces[0]
                x, y, w, h = self.original_face_location
                
                # Add padding around the face (30%)
                padding_x = int(w * 0.3)
                padding_y = int(h * 0.3)
                
                # Calculate padded coordinates with bounds checking
                start_x = max(0, x - padding_x)
                start_y = max(0, y - padding_y)
                end_x = min(self.current_image.shape[1], x + w + padding_x)
                end_y = min(self.current_image.shape[0], y + h + padding_y)
                
                # Crop image to face area with padding
                self.display_image_with_rect = self.current_image[start_y:end_y, start_x:end_x].copy()
                
                # Calculate the position of the face in the cropped image
                self.last_face_location = (x - start_x, y - start_y, w, h)
                
                # Draw rectangle on the cropped image
                self.display_image_with_rect = self.draw_face_rectangle(
                    self.display_image_with_rect, self.last_face_location, self.is_recognized)
                    
                self.display_image()
        
    def draw_face_rectangle(self, image, face_location, is_recognized):
        """Draw rectangle around detected face"""
        if face_location is None:
            return image.copy()
            
        display_image = image.copy()
        x, y, w, h = face_location
        
        # Set color based on recognition status
        if is_recognized:
            color = (0, 255, 0)  # Green for recognized
            text = "Recognized"
        else:
            color = (0, 0, 255)  # Red for unrecognized
            text = "Learning..."
            
        # Draw rectangle around face
        cv2.rectangle(display_image, (x, y), (x+w, y+h), color, 2)
        
        # Add text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        
        # Draw background for text
        cv2.rectangle(display_image, 
                     (x, y - text_height - 10), 
                     (x + text_width + 10, y - 5), 
                     color, 
                     -1)  # -1 fills the rectangle
        
        # Add text
        text_color = (0, 0, 0) if is_recognized else (255, 255, 255)  # Black on green, white on red
        cv2.putText(display_image, text, 
                   (x + 5, y - 10), 
                   font, 
                   font_scale, 
                   text_color,
                   thickness)
                   
        return display_image
        
    def display_image(self):
        """Display the current image with face rectangle"""
        if self.display_image_with_rect is not None:
            # Convert to RGB for display
            rgb_image = cv2.cvtColor(self.display_image_with_rect, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            
            # Create QImage
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            
            # Scale image to fit label while maintaining aspect ratio
            scaled_pixmap = QPixmap.fromImage(qt_image).scaled(
                self.image_label.size(), 
                Qt.AspectRatioMode.KeepAspectRatio, 
                Qt.TransformationMode.SmoothTransformation
            )
            
            self.image_label.setPixmap(scaled_pixmap)
            
    def showEvent(self, event):
        """Start camera when dialog is shown"""
        super().showEvent(event)
        # Start camera with the selected camera index
        if not self.camera_thread:
            self.camera_thread = CameraThread(self.camera_index)
            self.camera_thread.frame_ready.connect(self.set_image)
            self.camera_thread.start()
            
            # Add a delay to check if camera started properly
            QTimer.singleShot(2000, self.check_camera_started)
            
        # Start processing frames and taking photos
        self.process_timer.start()
        self.photo_timer.start()
        
    def check_camera_started(self):
        """Check if the camera started successfully"""
        if not hasattr(self, 'camera_thread') or not self.camera_thread or not self.camera_thread.isRunning():
            return
            
        # If we haven't received any frames yet, show an error
        if not hasattr(self, 'current_image') or self.current_image is None:
            QMessageBox.warning(
                self,
                "Camera Error",
                "Could not connect to the Android webcam.\n\n"
                "Please make sure:\n"
                "1. Your phone is connected via USB\n"
                "2. USB debugging is enabled\n"
                "3. You have a webcam app running on your phone\n"
                "4. You've allowed USB access permissions on your phone"
            )
            self.status_label.setText("Camera connection failed. Check phone connection.")
            
            # Stop the camera thread to clean up
            self.stop_camera()
        else:
            self.status_label.setText(f"Android webcam connected successfully!")
    
    def closeEvent(self, event):
        """Called when dialog is closed"""
        # Stop all timers when dialog closes
        self.process_timer.stop()
        self.photo_timer.stop()
        super().closeEvent(event)
    
    def check_take_photo(self):
        """Timer callback to take photo every 0.5 seconds if needed"""
        # Don't take photos if completion dialog has been shown or face is recognized for required time
        if self.completion_dialog_shown or (self.current_image is None or self.last_face_location is None):
            return
            
        # Skip taking photo if face is already recognized for required time
        if self.is_recognized and (time.time() - self.green_start_time) >= self.required_recognition_time:
            return
            
        # Take the photo
        self.take_photo(self.last_face_location)
            
    def take_photo(self, face_location):
        """Take a photo of the detected face and retrain"""
        # Prevent taking photos if the completion dialog has been shown
        if self.completion_dialog_shown:
            return
            
        x, y, w, h = face_location
            
        # If we're using a cropped image in display_image_with_rect, we need to extract
        # the face from the original current_image using the original coordinates
        if hasattr(self, 'original_face_location') and self.original_face_location is not None:
            # Use original coordinates
            x, y, w, h = self.original_face_location
            face_img = self.current_image[y:y+h, x:x+w]
        else:
            # Extract face region from whatever image we have
            if self.display_image_with_rect is not None and self.display_image_with_rect.size > 0:
                # If we have a cropped display image, use that
                face_img = self.display_image_with_rect[y:y+h, x:x+w]
            else:
                # Otherwise use the full image
                face_img = self.current_image[y:y+h, x:x+w]
        
        # Create a task to add the face and wait for retraining
        async def add_face_and_retrain():
            # Add the face and wait for retraining to complete
            await self.parent().face_recognizer.add_face_and_wait(face_img, self.name)
            # Update UI after retraining is complete
            if not self.completion_dialog_shown:  # Only update if not completed
                self.update_progress()
            
        # Start the task
        create_task(add_face_and_retrain())
        
        # Increment counter
        self.taken_photos += 1
        
        # Update progress label
        self.update_progress()
    
    def process_current_frame(self):
        """Process the current frame to detect faces and check recognition"""
        if self.current_image is None:
            return
            
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        # Process face detection for display
        if len(faces) == 1:
            x, y, w, h = faces[0]
            
            # Store original face location for taking photos
            self.original_face_location = faces[0]
            
            # Add padding around the face (30%)
            padding_x = int(w * 0.3)
            padding_y = int(h * 0.3)
            
            # Calculate padded coordinates with bounds checking
            start_x = max(0, x - padding_x)
            start_y = max(0, y - padding_y)
            end_x = min(self.current_image.shape[1], x + w + padding_x)
            end_y = min(self.current_image.shape[0], y + h + padding_y)
            
            # Crop image to face area with padding
            cropped_image = self.current_image[start_y:end_y, start_x:end_x].copy()
            
            # Calculate position of face in cropped image
            face_x = x - start_x
            face_y = y - start_y
            
            # Update face location for the cropped image
            self.last_face_location = (face_x, face_y, w, h)
            
            # Draw rectangle on cropped image
            self.display_image_with_rect = cropped_image.copy()
        else:
            # If no face or multiple faces, display full image
            self.display_image_with_rect = self.current_image.copy()
            self.last_face_location = None
            self.original_face_location = None
            
        if len(faces) == 0:
            # No faces detected
            self.is_recognized = False
            self.green_start_time = 0
            self.instructions.setText("No face detected. Please position your face in front of the camera.")
            self.display_image()
            return
            
        if len(faces) > 1:
            # Multiple faces detected
            self.is_recognized = False
            self.green_start_time = 0
            self.instructions.setText("Multiple faces detected. Please ensure only one face is visible.")
            self.display_image()
            return
            
        # We have exactly one face - use the original face location for recognition
        if hasattr(self.parent(), 'face_recognizer') and self.taken_photos >= 3:
            face_recognizer = self.parent().face_recognizer
            if face_recognizer.is_trained:
                # Create task for face recognition
                async def check_recognition():
                    recognized_faces = await face_recognizer.recognize_face(self.current_image)
                    
                    # Check if the face is recognized as the new person
                    recognized = False
                    for result in recognized_faces:
                        name, confidence, face_location = result
                        if name == self.name:
                            recognized = True
                            break
                            
                    # Update recognition state
                    if recognized:
                        # Face is recognized - should be green
                        if not self.is_recognized:
                            # Just started being recognized
                            self.is_recognized = True
                            self.green_start_time = time.time()
                        
                        # Check if recognized for required time
                        if (time.time() - self.green_start_time) >= self.required_recognition_time:
                            # Successfully recognized for required time
                            self.process_timer.stop()
                            self.photo_timer.stop()
                            
                            # Check if completion dialog has already been shown to avoid duplicates
                            if self.completion_dialog_shown:
                                return
                                
                            # Set flag to indicate completion dialog has been shown
                            self.completion_dialog_shown = True
                            
                            # Show success message and accept the dialog (close it)
                            QMessageBox.information(self, "Registration Complete", 
                                                  f"Face for {self.name} has been successfully registered!")
                            self.accept()
                            return
                    else:
                        # Face is not recognized - should be red
                        self.is_recognized = False
                        self.green_start_time = 0
                    
                    # Draw rectangle with recognition state on the cropped image
                    if self.display_image_with_rect is not None and self.last_face_location is not None:
                        self.display_image_with_rect = self.draw_face_rectangle(
                            self.display_image_with_rect, self.last_face_location, self.is_recognized)
                        self.display_image()
                    
                    # Update UI
                    self.update_progress()
                        
                create_task(check_recognition())
            else:
                # Recognizer not trained yet
                self.is_recognized = False
                if self.display_image_with_rect is not None and self.last_face_location is not None:
                    self.display_image_with_rect = self.draw_face_rectangle(
                        self.display_image_with_rect, self.last_face_location, False)
                    self.display_image()
        else:
            # Not enough photos yet or no recognizer
            self.is_recognized = False
            if self.display_image_with_rect is not None and self.last_face_location is not None:
                self.display_image_with_rect = self.draw_face_rectangle(
                    self.display_image_with_rect, self.last_face_location, False)
                self.display_image()
    
    def retrain_recognizer(self):
        """Retrain the face recognizer with the new photos"""
        if not hasattr(self.parent(), 'face_recognizer'):
            return
            
        # Update metadata
        self.parent().update_metadata()
        
        # Set instruction based on recognition state
        if self.is_recognized:
            self.instructions.setText("Face recognized! Keep position.")
            self.instructions.setStyleSheet("margin-bottom: 10px; color: #00CC00; font-weight: bold;")  # Darker green for better readability
        else:
            self.instructions.setText("Keep moving your face until it's recognized (green rectangle).")
            self.instructions.setStyleSheet("margin-bottom: 10px; color: white;")

class RecognitionPopupDialog(QDialog):
    def __init__(self, name: str, confidence: float, face_image: np.ndarray, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Face Recognized!")
        self.setMinimumSize(400, 300)
        self.setStyleSheet("""
            QDialog {
                background-color: #212121;
            }
        """)
        
        # Create layout
        layout = QVBoxLayout(self)
        layout.setSpacing(20)
        
        # Title
        title = ModernLabel("Face Recognized!")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("""
            QLabel {
                color: white;
                font-size: 24px;
                font-weight: bold;
            }
        """)
        layout.addWidget(title)
        
        # Face image
        self.face_label = ModernLabel()
        self.face_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.face_label.setMinimumSize(200, 200)
        self.face_label.setStyleSheet("""
            QLabel {
                background-color: #212121;
                border-radius: 10px;
            }
        """)
        layout.addWidget(self.face_label)
        
        # Display face image
        face_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        h, w, ch = face_rgb.shape
        bytes_per_line = ch * w
        qt_image = QImage(face_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        scaled_pixmap = pixmap.scaled(200, 200, Qt.AspectRatioMode.KeepAspectRatio)
        self.face_label.setPixmap(scaled_pixmap)
        
        # Convert confidence to percentage (higher is better)
        confidence_pct = max(0, min(100, 100 - confidence))
        
        # Info text with colorized confidence - use darker colors for better readability
        confidence_html = f"<span style='color:"
        if confidence_pct >= 90:
            confidence_html += "#00CC00'>Excellent Match"  # Darker Green
        elif confidence_pct >= 75:
            confidence_html += "#DDDD00'>Good Match"  # Darker Yellow
        else:
            confidence_html += "#DD7700'>Low Confidence Match"  # Darker Orange
        confidence_html += f" ({confidence_pct:.1f}%)</span>"
        
        info_text = ModernLabel(f"<html><body><h2>{name}</h2>{confidence_html}</body></html>")
        info_text.setAlignment(Qt.AlignmentFlag.AlignCenter)
        info_text.setStyleSheet("""
            QLabel {
                color: white;
                font-size: 16px;
                padding: 20px;
                background-color: rgba(66, 66, 66, 0.5);
                border-radius: 10px;
            }
        """)
        layout.addWidget(info_text)
        
        # OK button
        ok_button = ModernButton("OK")
        ok_button.clicked.connect(self.accept)
        layout.addWidget(ok_button)

class FaceRecognitionGUI(QMainWindow):
