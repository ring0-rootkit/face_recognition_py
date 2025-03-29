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
    
    def __init__(self):
        super().__init__()
        self.running = False
        self.camera = None
        
    def run(self):
        self.camera = cv2.VideoCapture(0)
        self.running = True
        
        while self.running:
            ret, frame = self.camera.read()
            if ret:
                self.frame_ready.emit(frame)
                
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
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Face Registration")
        self.setModal(True)
        self.setMinimumWidth(400)
        
        # Create layout
        layout = QVBoxLayout()
        
        # Add title
        title = QLabel("Face Registration")
        title.setStyleSheet("font-size: 18px; font-weight: bold; margin-bottom: 10px;")
        layout.addWidget(title)
        
        # Add instructions
        self.instructions = QLabel("Move your face in different positions for training.")
        self.instructions.setWordWrap(True)
        self.instructions.setStyleSheet("margin-bottom: 10px; color: white;")
        layout.addWidget(self.instructions)
        
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
        layout.addWidget(self.image_label)
        
        # Progress label
        self.progress_label = QLabel()
        self.progress_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.progress_label.setStyleSheet("margin-top: 10px; color: white;")
        layout.addWidget(self.progress_label)
        
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
        layout.addWidget(self.cancel_button)
        
        self.setLayout(layout)
        
        # Initialize variables
        self.name = None
        self.current_image = None
        self.display_image_with_rect = None  # Image with rectangle overlay
        self.taken_photos = 0
        self.face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        
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
            self.progress_label.setStyleSheet("margin-top: 10px; color: #00FF00; font-weight: bold;")
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
        """Called when dialog is shown"""
        super().showEvent(event)
        # Start processing frames and taking photos
        self.process_timer.start()
        self.photo_timer.start()
        
    def closeEvent(self, event):
        """Called when dialog is closed"""
        # Stop all timers when dialog closes
        self.process_timer.stop()
        self.photo_timer.stop()
        super().closeEvent(event)
    
    def check_take_photo(self):
        """Timer callback to take photo every 0.5 seconds if needed"""
        if self.current_image is None or self.last_face_location is None:
            return
            
        # Skip taking photo if face is already recognized for required time
        if self.is_recognized and (time.time() - self.green_start_time) >= self.required_recognition_time:
            return
            
        # Take the photo
        self.take_photo(self.last_face_location)
            
    def take_photo(self, face_location):
        """Take a photo of the detected face and retrain"""
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
            self.instructions.setStyleSheet("margin-bottom: 10px; color: #00FF00; font-weight: bold;")
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
        
        # Info text with colorized confidence
        confidence_html = f"<span style='color:"
        if confidence_pct >= 90:
            confidence_html += "#00FF00'>Excellent Match"  # Green
        elif confidence_pct >= 75:
            confidence_html += "#FFFF00'>Good Match"  # Yellow
        else:
            confidence_html += "#FFA500'>Low Confidence Match"  # Orange
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
    def __init__(self):
        super().__init__()
        self.detection_counter = 0
        self.setWindowTitle("Face Recognition System")
        self.setMinimumSize(1400, 800)
        
        # Initialize variables for face detection
        self.last_recognized_faces = []
        self.last_detected_faces = []
        self.is_locked = False
        self.locked_face = None
        self.locked_name = None
        self.locked_confidence = None
        self.is_scanning = True
        self.active_dialogs = 0
        self.current_image = None
        
        # Initialize variables for multiple image updates
        self.update_timer = None
        self.update_count = 0
        self.max_updates = 5
        
        # Ensure faces directory exists
        faces_dir = Path("faces/faces")
        faces_dir.mkdir(parents=True, exist_ok=True)
        
        # Ensure metadata.json exists
        metadata_path = Path("faces/metadata.json")
        if not metadata_path.exists():
            with open(metadata_path, 'w') as f:
                json.dump([], f)
        
        # Initialize face recognition
        self.face_recognizer = FaceRecognition("faces/")
        
        # Start loading database in background
        self.loading_task = self.face_recognizer.start_loading_database()
        
        # Create main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(20, 20, 20, 20)
        
        # Title
        title = ModernLabel("Face Recognition System")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("""
            QLabel {
                color: white;
                font-size: 24px;
                font-weight: bold;
                margin-bottom: 20px;
            }
        """)
        main_layout.addWidget(title)
        
        # Create content layout
        content_layout = QHBoxLayout()
        content_layout.setSpacing(20)
        
        # Left panel (Image and Camera)
        left_panel = ModernFrame()
        left_layout = QVBoxLayout(left_panel)
        
        # Image display
        self.image_label = ModernLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumSize(500, 400)
        self.image_label.setStyleSheet("""
            QLabel {
                background-color: #212121;
                border-radius: 10px;
            }
        """)
        left_layout.addWidget(self.image_label)
        
        # Camera controls
        camera_layout = QHBoxLayout()
        self.camera_button = ModernButton("Start Camera")
        self.camera_button.clicked.connect(self.toggle_camera)
        self.upload_button = ModernButton("Upload Image")
        self.upload_button.clicked.connect(self.load_image)
        
        camera_layout.addWidget(self.camera_button)
        camera_layout.addWidget(self.upload_button)
        left_layout.addLayout(camera_layout)
        
        # Middle panel (Controls)
        middle_panel = ModernFrame()
        middle_layout = QVBoxLayout(middle_panel)
        
        # Name input
        name_layout = QHBoxLayout()
        name_label = ModernLabel("Name:")
        self.name_entry = ModernLineEdit()
        name_layout.addWidget(name_label)
        name_layout.addWidget(self.name_entry)
        middle_layout.addLayout(name_layout)
        
        # Action buttons
        # Create buttons first
        self.add_button = ModernButton("Add Face")
        self.add_button.clicked.connect(self.add_face)
        self.delete_button = ModernButton("Delete Face")
        self.delete_button.clicked.connect(self.delete_face)
        
        # Create the person selection dropdown
        people_layout = QHBoxLayout()
        people_label = ModernLabel("Person:")
        self.people_combo = ModernComboBox()
        people_layout.addWidget(people_label)
        people_layout.addWidget(self.people_combo)
        middle_layout.addLayout(people_layout)
        
        # Lock button
        self.lock_button = ModernButton("Lock Recognition")
        self.lock_button.setCheckable(True)
        self.lock_button.clicked.connect(self.toggle_lock)
        
        # Create button layouts
        button_layout = QGridLayout()
        button_layout.addWidget(self.add_button, 0, 0)
        button_layout.addWidget(self.delete_button, 0, 1)
        button_layout.addWidget(self.lock_button, 1, 0, 1, 2)  # Span both columns
        middle_layout.addLayout(button_layout)
        
        # Right panel (Recognized Faces)
        right_panel = ModernFrame()
        right_layout = QVBoxLayout(right_panel)
        
        # Header for recognized faces
        recognition_header = ModernLabel("Recognized Faces")
        recognition_header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        recognition_header.setStyleSheet("""
            QLabel {
                font-size: 18px;
                font-weight: bold;
                margin-bottom: 10px;
            }
        """)
        right_layout.addWidget(recognition_header)
        
        # Scroll area for recognized faces
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("""
            QScrollArea {
                border: none;
                background-color: transparent;
                border-radius: 10px;
            }
            QScrollBar:vertical {
                border: none;
                background-color: #424242;
                width: 10px;
                margin: 0px;
            }
            QScrollBar::handle:vertical {
                background-color: #666666;
                border-radius: 5px;
                min-height: 20px;
            }
            QScrollBar::handle:vertical:hover {
                background-color: #888888;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
        """)
        
        self.recognized_faces_widget = QWidget()
        self.recognized_faces_widget.setStyleSheet("background-color: transparent;")
        self.recognized_faces_layout = QVBoxLayout(self.recognized_faces_widget)
        self.recognized_faces_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.recognized_faces_layout.setSpacing(10)
        scroll.setWidget(self.recognized_faces_widget)
        
        right_layout.addWidget(scroll)
        
        # Add panels to content layout
        content_layout.addWidget(left_panel)
        content_layout.addWidget(middle_panel)
        content_layout.addWidget(right_panel)
        main_layout.addLayout(content_layout)
        
        # Status label
        self.status_label = ModernLabel()
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(self.status_label)
        
        # Initialize variables
        self.camera_thread = None
        
        # Add recognition timer (using regular method, not async)
        self.recognition_timer = QTimer()
        self.recognition_timer.timeout.connect(self.check_for_faces)
        self.recognition_timer.start(500)  # Check every 3000ms
        
        # Initialize recognition variables
        self.last_recognition_time = 0
        self.face_detected_time = 0
        self.last_face_detection = 0
        
        # Set dark theme
        self.setStyleSheet("""
            QMainWindow {
                background-color: #212121;
            }
        """)
        
        # Populate the people combo box at startup
        self.update_people_combo()

    def toggle_camera(self):
        if not self.camera_thread or not self.camera_thread.isRunning():
            self.start_camera()
        else:
            self.stop_camera()

    def start_camera(self):
        self.camera_thread = CameraThread()
        self.camera_thread.frame_ready.connect(self.update_camera_frame)
        self.camera_thread.start()
        self.is_scanning = True
        
        self.camera_button.setText("Stop Camera")
        self.upload_button.setEnabled(False)

    def stop_camera(self):
        if self.camera_thread:
            self.camera_thread.stop()
            self.camera_thread.wait()
        self.is_scanning = False
        
        self.camera_button.setText("Start Camera")
        self.upload_button.setEnabled(True)

    def update_camera_frame(self, frame):
        self.current_image = frame
        if self.is_locked and self.locked_face is not None:
            # Display locked face in the recognized faces panel
            self.display_locked_face()
        
        # Only update the main camera frame if not in face registration mode
        if not hasattr(self, 'in_face_registration') or not self.in_face_registration:
            self.display_image(frame)

    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Image",
            "",
            "Image Files (*.jpg *.jpeg *.png *.bmp *.gif)"
        )
        if file_path:
            self.current_image = cv2.imread(file_path)
            if self.current_image is not None:
                # Сбросим счетчик для мгновенного распознавания
                self.detection_counter = 0
                
                # Выполним распознавание лиц сразу
                self.last_detected_faces = self.face_recognizer.detect_faces(self.current_image)
                
                # Создадим задачу для распознавания
                async def immediate_recognition():
                    self.last_recognized_faces = await self.face_recognizer.recognize_face(self.current_image)
                    # Если есть лица в кадре, их нужно обработать
                    if self.is_locked and self.locked_name:
                        for result in self.last_recognized_faces:
                            name, confidence, (x, y, w, h) = result
                            if name == self.locked_name:
                                self.locked_face = self.current_image[y:y+h, x:x+w]
                                self.locked_confidence = confidence
                                self.display_locked_face()
                                break
                
                # Запустим асинхронную задачу
                create_task(immediate_recognition())
                
                # Обновим отображение
                self.display_image(self.current_image)
                self.status_label.setText("Image loaded successfully")

                # Настроим счетчик для множественного обновления кадра
                self.update_count = 0
                self.max_updates = 5  # Количество обновлений
                
                # Остановим предыдущий таймер, если он существует и активен
                if self.update_timer and self.update_timer.isActive():
                    self.update_timer.stop()
                
                # Создаем таймер для периодического обновления
                self.update_timer = QTimer()
                self.update_timer.timeout.connect(self.update_loaded_image)
                self.update_timer.start(500)  # Интервал между обновлениями - 500 мс
            else:
                QMessageBox.critical(self, "Error", "Failed to load image")
                
    def update_loaded_image(self):
        """Обновляет отображение загруженного изображения несколько раз"""
        if self.current_image is not None:
            # Обновляем изображение
            self.display_image(self.current_image)
            
            # Увеличиваем счетчик обновлений
            self.update_count += 1
            
            # Если достигнуто максимальное количество обновлений, останавливаем таймер
            if self.update_count >= self.max_updates:
                self.update_timer.stop()
                self.status_label.setText("Image processing completed")

    def display_image(self, image):
        # Create a copy of the image for drawing
        display_image = image.copy()
        
        # Standard face detection and recognition
        # Update recognition status every 30 frames or when counter is 0
        self.detection_counter += 1
        
        # Если счетчик равен 1, значит он только что был сброшен в другом методе и распознавание уже запущено
        # Поэтому пропускаем обновление распознавания в этом случае
        if self.detection_counter >= 30 and self.detection_counter != 1:
            self.detection_counter = 0
            # Update last known faces
            self.last_detected_faces = self.face_recognizer.detect_faces(image)
            # Create task for face recognition
            async def update_recognition():
                self.last_recognized_faces = await self.face_recognizer.recognize_face(image)
                # Note: Lock button is now enabled/disabled in update_people_combo method
            create_task(update_recognition())
        
        # Создаем словарь для быстрого поиска распознанных лиц по координатам
        recognized_faces_dict = {}
        for result in self.last_recognized_faces:
            name, confidence, face_location = result
            # Используем координаты лица как ключ для быстрого поиска
            recognized_faces_dict[face_location] = (name, confidence)
        
        # Draw rectangles for all detected faces
        for (x, y, w, h) in self.last_detected_faces:
            # Проверяем, распознано ли это лицо
            face_location = (x, y, w, h)
            if face_location in recognized_faces_dict:
                # Лицо распознано
                name, confidence = recognized_faces_dict[face_location]
                
                # Convert confidence to percentage (higher is better)
                confidence_pct = max(0, min(100, 100 - confidence))
                
                # If confidence is less than 10%, treat as unknown
                if confidence_pct < 10:
                    # Если уверенность слишком низкая, пропускаем кадр чтобы избежать мерцания
                    continue
                
                # Skip if locked to a specific person and this is not that person
                if self.is_locked and name != self.locked_name:
                    # Draw gray rectangle for ignored faces when locked
                    cv2.rectangle(display_image, (x, y), (x+w, y+h), (128, 128, 128), 2)
                    continue
                
                # Draw green rectangle for recognized faces
                cv2.rectangle(display_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Color gradient based on confidence
                if confidence_pct >= 90:
                    bg_color = (0, 255, 0)  # Green for high confidence
                    text_color = (0, 0, 0)  # Black text
                elif confidence_pct >= 75:
                    bg_color = (0, 255, 255)  # Yellow for medium confidence
                    text_color = (0, 0, 0)  # Black text
                else:
                    bg_color = (0, 165, 255)  # Orange for low confidence
                    text_color = (255, 255, 255)  # White text
                
                text = f"{name} ({confidence_pct:.1f}%)"
                
                # Calculate text size to position it properly
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                thickness = 2
                (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
                
                # Draw background rectangle for text
                cv2.rectangle(display_image, 
                            (x, y - text_height - 10), 
                            (x + text_width, y - 5), 
                            bg_color, 
                            -1)  # -1 fills the rectangle
                
                # Add text
                cv2.putText(display_image, text, 
                            (x, y - 10), 
                            font, 
                            font_scale, 
                            text_color,
                            thickness)
            else:
                # Если лицо не распознано в текущем кадре
                # Но! Проверим, было ли это лицо распознано недавно в том же месте
                # Это поможет избежать мигания между "Unknown" и именем
                
                # Проверим, есть ли пересечение с любыми недавно распознанными лицами
                # с приемлемым отклонением (например, 20 пикселей)
                tolerance = 20
                found_match = False
                
                for result in self.last_recognized_faces:
                    known_name, known_confidence, (known_x, known_y, known_w, known_h) = result
                    
                    # Проверяем пересечение областей с учетом допуска
                    if (abs(known_x - x) < tolerance and 
                        abs(known_y - y) < tolerance and 
                        abs(known_w - w) < tolerance and 
                        abs(known_h - h) < tolerance):
                            
                        # Найдено совпадение с недавно распознанным лицом
                        found_match = True
                        
                        # Конвертируем уверенность в проценты
                        confidence_pct = max(0, min(100, 100 - known_confidence))
                        
                        # Пропускаем, если уверенность слишком низкая
                        if confidence_pct < 10:
                            found_match = False
                            break
                            
                        # Пропускаем, если заблокированы на конкретном человеке
                        if self.is_locked and known_name != self.locked_name:
                            # Draw gray rectangle for ignored faces when locked
                            cv2.rectangle(display_image, (x, y), (x+w, y+h), (128, 128, 128), 2)
                            break
                            
                        # Draw green rectangle for recognized faces
                        cv2.rectangle(display_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        
                        # Color gradient based on confidence
                        if confidence_pct >= 90:
                            bg_color = (0, 255, 0)  # Green for high confidence
                            text_color = (0, 0, 0)  # Black text
                        elif confidence_pct >= 75:
                            bg_color = (0, 255, 255)  # Yellow for medium confidence
                            text_color = (0, 0, 0)  # Black text
                        else:
                            bg_color = (0, 165, 255)  # Orange for low confidence
                            text_color = (255, 255, 255)  # White text
                        
                        text = f"{known_name} ({confidence_pct:.1f}%)"
                        
                        # Calculate text size to position it properly
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        font_scale = 0.5
                        thickness = 2
                        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
                        
                        # Draw background rectangle for text
                        cv2.rectangle(display_image, 
                                    (x, y - text_height - 10), 
                                    (x + text_width, y - 5), 
                                    bg_color, 
                                    -1)  # -1 fills the rectangle
                        
                        # Add text
                        cv2.putText(display_image, text, 
                                    (x, y - 10), 
                                    font, 
                                    font_scale, 
                                    text_color,
                                    thickness)
                        break
                
                if not found_match:
                    # Если лицо действительно не распознано и не соответствует ни одному недавнему
                    
                    # If locked, don't display unknown faces
                    if self.is_locked:
                        # Draw gray rectangle for unknown faces when locked
                        cv2.rectangle(display_image, (x, y), (x+w, y+h), (128, 128, 128), 2)
                    else:
                        # Draw red rectangle for unrecognized faces
                        cv2.rectangle(display_image, (x, y), (x+w, y+h), (0, 0, 255), 2)
                        # Add "Unknown" text with background
                        text = "Unknown"
                        
                        # Calculate text size
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        font_scale = 0.5
                        thickness = 2
                        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
                        
                        # Draw background rectangle for text
                        cv2.rectangle(display_image, 
                                    (x, y - text_height - 10), 
                                    (x + text_width, y - 5), 
                                    (0, 0, 255), 
                                    -1)  # -1 fills the rectangle
                        
                        # Add text
                        cv2.putText(display_image, text, 
                                    (x, y - 10), 
                                    font, 
                                    font_scale, 
                                    (255, 255, 255),  # White text
                                    thickness)

        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB)
        
        # Resize image to fit the label
        height, width = image_rgb.shape[:2]
        label_size = self.image_label.size()
        scale = min(label_size.width() / width, label_size.height() / height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        image_rgb = cv2.resize(image_rgb, (new_width, new_height))
        
        # Convert to QImage and QPixmap
        h, w, ch = image_rgb.shape
        bytes_per_line = ch * w
        qt_image = QImage(image_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        
        # Scale pixmap to fit label while maintaining aspect ratio
        scaled_pixmap = pixmap.scaled(
            self.image_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        
        self.image_label.setPixmap(scaled_pixmap)

    def dialog_opened(self):
        self.active_dialogs += 1
        self.is_scanning = False

    def dialog_closed(self):
        self.active_dialogs -= 1
        if self.active_dialogs == 0:
            self.is_scanning = True

    def add_face(self):
        """Add a new face to the database"""
        name = self.name_entry.text().strip()
        if not name:
            QMessageBox.warning(self, "Warning", "Please enter a name")
            return
            
        # Check if face already exists
        if name in self.face_recognizer.known_face_names:
            reply = QMessageBox.question(self, 'Face Exists', 
                f'Face "{name}" already exists. Do you want to add more photos?',
                QMessageBox.Yes | QMessageBox.No)
            if reply == QMessageBox.No:
                return
        
        # Create person directory if it doesn't exist
        person_dir = Path("faces/faces") / name
        person_dir.mkdir(parents=True, exist_ok=True)
        
        # If using a loaded image instead of camera
        if self.current_image is not None and (not self.camera_thread or not self.camera_thread.isRunning()):
            # Check if we have detected faces in the current image
            if not hasattr(self, 'last_detected_faces') or len(self.last_detected_faces) == 0:
                # Try to detect faces in the current image
                self.last_detected_faces = self.face_recognizer.detect_faces(self.current_image)
                
            if not self.last_detected_faces or len(self.last_detected_faces) == 0:
                QMessageBox.warning(self, "Warning", "No faces detected in the current image")
                return
                
            if len(self.last_detected_faces) > 1:
                QMessageBox.warning(self, "Warning", "Multiple faces detected. Please use an image with a single face.")
                return
                
            # Extract and save the detected face
            x, y, w, h = self.last_detected_faces[0]
            face_img = self.current_image[y:y+h, x:x+w]
            
            # Create a task to add the face and wait for retraining
            async def add_face_and_wait_for_loaded_image():
                # Add the face and wait for retraining to complete
                success = await self.face_recognizer.add_face_and_wait(face_img, name)
                if success:
                    # Update UI after retraining is complete
                    self.status_label.setText(f'Face "{name}" added successfully and model retrained!')
                    self.name_entry.clear()
                    self.update_people_combo()
                else:
                    QMessageBox.warning(self, "Warning", f"Failed to add face for {name}")
            
            # Start the task
            create_task(add_face_and_wait_for_loaded_image())
            
            # Show temporary message
            self.status_label.setText(f'Adding face for "{name}" and retraining model...')
            return
            
        # Make sure camera is started for live registration
        camera_was_off = False
        if not self.camera_thread or not self.camera_thread.isRunning():
            self.start_camera()
            camera_was_off = True
            # Give the camera a moment to initialize
            QMessageBox.information(self, "Camera Started", 
                "Camera has been started for face registration. Position your face in the camera view.")
        
        # Signal to the user that we're entering registration mode
        self.dialog_opened()  # Pause normal scanning
        self.status_label.setText(f"Starting face registration for {name}...")
        
        # Set flag to disable main camera frame updates
        self.in_face_registration = True
        
        # Clear the main camera display
        blank_image = np.zeros((480, 640, 3), dtype=np.uint8)
        blank_image[:] = (45, 45, 45)  # Dark gray background
        
        # Add text to blank image
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = "Face Registration in Progress"
        text_size = cv2.getTextSize(text, font, 1, 2)[0]
        text_x = (blank_image.shape[1] - text_size[0]) // 2
        text_y = (blank_image.shape[0] + text_size[1]) // 2
        cv2.putText(blank_image, text, (text_x, text_y), font, 1, (255, 255, 255), 2)
        
        # Convert to RGB and display
        rgb_image = cv2.cvtColor(blank_image, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        self.image_label.setPixmap(pixmap)
            
        # Create registration dialog
        dialog = FaceRegistrationDialog(self)
        dialog.set_name(name)
        
        # Ensure camera thread exists and is running before connecting
        if self.camera_thread and self.camera_thread.isRunning():
            # Connect camera frame to dialog
            self.camera_thread.frame_ready.connect(dialog.set_image)
            
            if dialog.exec() == QDialog.DialogCode.Accepted:
                # Update metadata and UI
                self.update_metadata()
                self.status_label.setText(f'Face "{name}" added successfully!')
                self.name_entry.clear()  # Clear the name field after successful registration
            else:
                # Clean up if cancelled
                # Check if the directory was created and is empty
                if person_dir.exists() and not any(person_dir.iterdir()):
                    person_dir.rmdir()
                self.status_label.setText("Face registration cancelled")
            
            # Disconnect camera frame
            try:
                self.camera_thread.frame_ready.disconnect(dialog.set_image)
            except TypeError:
                # Handle case where connection doesn't exist
                pass
        else:
            QMessageBox.critical(self, "Error", "Camera could not be started or is not available")
            # Reset the flag
            self.in_face_registration = False
            return
        
        # Reset the flag to enable main camera frame again
        self.in_face_registration = False
        
        # Resume normal scanning
        self.dialog_closed()
        
        # If camera was off before, turn it off again
        if camera_was_off:
            self.stop_camera()

    def delete_face(self):
        name = self.name_entry.text().strip()
        if not name:
            QMessageBox.warning(self, "Warning", "Please enter a name to delete")
            return
            
        if self.face_recognizer.delete_face(name):
            self.status_label.setText(f"Face deleted successfully for {name}")
            
            # Update people in combo box
            self.update_people_combo()
            
            # If we had this person locked, unlock
            if self.is_locked and self.locked_name == name:
                self.lock_button.setChecked(False)
                self.toggle_lock()
                
            self.name_entry.clear()
        else:
            QMessageBox.critical(self, "Error", "Face not found")

    def update_metadata(self):
        """Update the face recognition system with new metadata"""
        # Start loading database in background
        self.loading_task = self.face_recognizer.start_loading_database()
        
        # Save metadata to file by scanning faces directory
        metadata = []
        faces_dir = Path("faces/faces")
        if faces_dir.exists():
            for person_dir in faces_dir.iterdir():
                if person_dir.is_dir():
                    person_name = person_dir.name
                    person_photos = []
                    # Add all photos from person's directory
                    for photo in person_dir.glob("*.jpg"):
                        person_photos.append(photo.name)
                    if person_photos:
                        metadata.append({
                            "name": person_name,
                            "photos": person_photos
                        })
        
        # Save metadata to file
        metadata_path = Path("faces/metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        
        # Update the combo box immediately from metadata file
        self.update_people_combo()
        
        # Also schedule another update after the database is fully loaded
        async def update_combo_after_loading():
            await self.loading_task
            self.update_people_combo()
            
        create_task(update_combo_after_loading())

    def check_for_faces(self):
        """Automatically check for faces in the current image (simplified)"""
        if not self.is_scanning or self.current_image is None or self.active_dialogs > 0:
            return
            
        # Face detection is already handled in display_image, no need for duplicate processing here

    def closeEvent(self, event):
        """Handle application close event by cleaning up resources"""
        print("Application closing, cleaning up resources...")
        
        # Stop the camera
        self.stop_camera()
        
        # Stop all timers
        self.recognition_timer.stop()
        if self.update_timer and self.update_timer.isActive():
            self.update_timer.stop()
        
        # Clean up face recognizer
        if hasattr(self, 'face_recognizer'):
            self.face_recognizer.cleanup()
            
        event.accept()
        print("Application cleanup completed")

    def update_people_combo(self):
        """Update the list of people in the combo box directly from metadata.json"""
        self.people_combo.clear()
        
        # Read names directly from metadata.json
        metadata_path = Path("faces/metadata.json")
        if metadata_path.exists():
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                # Extract names from metadata
                names = [person["name"] for person in metadata if "name" in person]
                
                # Add all names to the combo box
                for name in sorted(names):
                    self.people_combo.addItem(name)
                
                # Enable or disable lock button based on available names
                # Check if lock_button exists before accessing it
                if hasattr(self, 'lock_button'):
                    self.lock_button.setEnabled(len(names) > 0)
            except (json.JSONDecodeError, IOError) as e:
                print(f"Error reading metadata.json: {e}")
                self.status_label.setText("Error reading faces database. Please restart the application.")
        else:
            # If metadata file doesn't exist, disable lock button
            if hasattr(self, 'lock_button'):
                self.lock_button.setEnabled(False)

    def toggle_lock(self):
        """Toggle lock state for the selected person"""
        self.is_locked = self.lock_button.isChecked()
        
        if self.is_locked:
            # Get the selected person's name
            if self.people_combo.count() == 0:
                QMessageBox.warning(self, "Warning", "No known faces available for locking")
                self.lock_button.setChecked(False)
                self.is_locked = False
                return
                
            self.locked_name = self.people_combo.currentText()
            self.lock_button.setText(f"Unlock Recognition (Locked to {self.locked_name})")
            
            # Display a message
            self.status_label.setText(f"Recognition locked to {self.locked_name}")
            
            # Find and display a matching face if available
            face_found = False
            for result in self.last_recognized_faces:
                name, confidence, (x, y, w, h) = result
                if name == self.locked_name and self.current_image is not None:
                    self.locked_face = self.current_image[y:y+h, x:x+w]
                    self.locked_confidence = confidence
                    self.display_locked_face()
                    face_found = True
                    break
                    
            # If no matching face was found in the current frame, just display the lock info
            if not face_found:
                self.display_locked_info()
        else:
            self.lock_button.setText("Lock Recognition")
            self.status_label.setText("Recognition unlocked")
            # Clear the locked face area
            for i in reversed(range(self.recognized_faces_layout.count())):
                self.recognized_faces_layout.itemAt(i).widget().setParent(None)
                
    def display_locked_face(self):
        """Display the locked face in the recognized faces panel"""
        # Clear previous recognized faces
        for i in reversed(range(self.recognized_faces_layout.count())): 
            self.recognized_faces_layout.itemAt(i).widget().setParent(None)
        
        # Create face frame
        face_frame = ModernFrame()
        face_layout = QVBoxLayout(face_frame)
        face_layout.setSpacing(5)
        
        # Convert face image to QPixmap
        face_rgb = cv2.cvtColor(self.locked_face, cv2.COLOR_BGR2RGB)
        h, w, ch = face_rgb.shape
        bytes_per_line = ch * w
        qt_image = QImage(face_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        
        # Create and set face label
        face_label = ModernLabel()
        face_label.setPixmap(pixmap.scaled(150, 150, Qt.AspectRatioMode.KeepAspectRatio))
        face_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # Calculate confidence percentage (higher is better)
        confidence_pct = max(0, min(100, 100 - self.locked_confidence))
        
        # Create name and confidence label with styling based on confidence
        confidence_html = f"<span style='color:"
        if confidence_pct >= 90:
            confidence_html += "#00FF00'>Excellent Match"  # Green
        elif confidence_pct >= 75:
            confidence_html += "#FFFF00'>Good Match"  # Yellow
        else:
            confidence_html += "#FFA500'>Low Confidence Match"  # Orange
        confidence_html += f" ({confidence_pct:.1f}%)</span>"
        
        info_label = ModernLabel(f"<html><body><h3>{self.locked_name}</h3>{confidence_html}<br><b>(Locked)</b></body></html>")
        info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        info_label.setStyleSheet("""
            QLabel {
                color: white;
                font-size: 14px;
                padding: 5px;
                background-color: rgba(66, 66, 66, 0.5);
                border-radius: 5px;
            }
        """)
        
        face_layout.addWidget(face_label)
        face_layout.addWidget(info_label)
        
        self.recognized_faces_layout.addWidget(face_frame)

    def display_locked_info(self):
        """Display just the locked person info without a face image"""
        # Clear previous recognized faces
        for i in reversed(range(self.recognized_faces_layout.count())): 
            self.recognized_faces_layout.itemAt(i).widget().setParent(None)
        
        # Create info frame
        info_frame = ModernFrame()
        info_layout = QVBoxLayout(info_frame)
        info_layout.setSpacing(5)
        
        # Create info label
        info_label = ModernLabel(f"<html><body><h3>{self.locked_name}</h3><br><b>(Locked)</b></body></html>")
        info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        info_label.setStyleSheet("""
            QLabel {
                color: white;
                font-size: 14px;
                padding: 10px;
                background-color: rgba(66, 66, 66, 0.5);
                border-radius: 5px;
            }
        """)
        
        info_layout.addWidget(info_label)
        self.recognized_faces_layout.addWidget(info_frame)

def main():
    """Main application entry point with asyncio-Qt integration"""
    app = QApplication(sys.argv)
    
    # Create a custom event loop that integrates with Qt
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    # Store all tasks that need to be canceled on exit
    pending_tasks = set()
    
    # Override the create_task function to track tasks
    old_create_task = asyncio.create_task
    def patched_create_task(coro):
        task = old_create_task(coro)
        # Add task to tracking set
        pending_tasks.add(task)
        # Remove task from set when it's done
        task.add_done_callback(pending_tasks.discard)
        return task
    
    # Replace the built-in create_task with our tracked version
    asyncio.create_task = patched_create_task
    
    # Create and show the main window
    window = FaceRecognitionGUI()
    window.show()
    
    # Set up a periodic callback to process asyncio events
    def process_asyncio_events():
        loop.call_soon(loop.stop)
        loop.run_forever()
    
    # Create timer to process asyncio events
    asyncio_timer = QTimer()
    asyncio_timer.timeout.connect(process_asyncio_events)
    asyncio_timer.start(10)  # 10ms interval for processing asyncio events
    
    # Start the Qt event loop
    try:
        sys.exit(app.exec())
    finally:
        # Cancel all pending tasks before closing loop
        print(f"Cancelling {len(pending_tasks)} pending tasks...")
        for task in pending_tasks:
            task.cancel()
        
        # Run the event loop one last time to let cancelled tasks clean up
        if pending_tasks:
            loop.run_until_complete(asyncio.gather(*pending_tasks, return_exceptions=True))
        
        # Restore original create_task function
        asyncio.create_task = old_create_task
        
        # Clean up the loop when the application exits
        loop.close()
        print("Event loop closed successfully")

if __name__ == '__main__':
    main()