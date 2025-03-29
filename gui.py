import sys
import cv2
import numpy as np
import os
from face_recognition import FaceRecognition
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QPushButton, QLabel, QLineEdit, 
                            QFileDialog, QMessageBox, QFrame, QScrollArea, QDialog)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QImage, QPixmap, QFont, QIcon, QPalette, QColor
import time

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
        self.setMinimumSize(1000, 800)
        self.setStyleSheet("""
            QDialog {
                background-color: #212121;
            }
        """)
        
        # Create layout
        layout = QVBoxLayout(self)
        layout.setSpacing(20)
        
        # Title
        title = ModernLabel("Face Registration")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("""
            QLabel {
                color: white;
                font-size: 24px;
                font-weight: bold;
            }
        """)
        layout.addWidget(title)
        
        # Instructions
        self.instructions = ModernLabel()
        self.instructions.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.instructions.setWordWrap(True)
        self.instructions.setStyleSheet("""
            QLabel {
                color: white;
                font-size: 16px;
                padding: 20px;
                background-color: rgba(66, 66, 66, 0.5);
                border-radius: 10px;
            }
        """)
        layout.addWidget(self.instructions)
        
        # Create horizontal layout for camera and preview
        preview_layout = QHBoxLayout()
        
        # Camera view
        self.camera_label = ModernLabel()
        self.camera_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.camera_label.setMinimumSize(400, 300)
        self.camera_label.setStyleSheet("""
            QLabel {
                background-color: #212121;
                border-radius: 10px;
            }
        """)
        preview_layout.addWidget(self.camera_label)
        
        # Preview of taken photo
        self.preview_label = ModernLabel()
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setMinimumSize(400, 300)
        self.preview_label.setStyleSheet("""
            QLabel {
                background-color: #212121;
                border-radius: 10px;
            }
        """)
        preview_layout.addWidget(self.preview_label)
        
        layout.addLayout(preview_layout)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        self.capture_button = ModernButton("Capture")
        self.capture_button.clicked.connect(self.capture_photo)
        self.accept_button = ModernButton("Accept")
        self.accept_button.clicked.connect(self.accept_photo)
        self.accept_button.setEnabled(False)
        self.redo_button = ModernButton("Redo")
        self.redo_button.clicked.connect(self.redo_photo)
        self.redo_button.setEnabled(False)
        self.cancel_button = ModernButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        
        button_layout.addWidget(self.capture_button)
        button_layout.addWidget(self.redo_button)
        button_layout.addWidget(self.accept_button)
        button_layout.addWidget(self.cancel_button)
        
        layout.addLayout(button_layout)
        
        # Progress
        self.progress_label = ModernLabel()
        self.progress_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.progress_label)
        
        # Initialize variables
        self.current_step = 0
        self.steps = [
            ("front", "Please look directly at the camera"),
            ("left", "Please turn your head slightly to the left"),
            ("right", "Please turn your head slightly to the right"),
            ("up", "Please look slightly upward"),
            ("down", "Please look slightly downward"),
            ("front2", "Please look directly at the camera again")
        ]
        self.current_image = None
        self.name = ""
        self.taken_photos = {}
        
        self.update_ui()

    def set_name(self, name: str):
        self.name = name
        self.update_ui()

    def update_ui(self):
        if self.current_step < len(self.steps):
            angle, instruction = self.steps[self.current_step]
            self.instructions.setText(instruction)
            self.progress_label.setText(f"Step {self.current_step + 1} of {len(self.steps)}")
            
            # Show preview of taken photo if exists
            if angle in self.taken_photos:
                self.display_image(self.taken_photos[angle], self.preview_label)
            else:
                self.preview_label.clear()
        else:
            self.accept()

    def set_image(self, image):
        self.current_image = image
        self.display_image(image, self.camera_label)

    def capture_photo(self):
        if self.current_image is not None:
            self.display_image(self.current_image, self.preview_label)
            self.accept_button.setEnabled(True)
            self.redo_button.setEnabled(True)
            self.capture_button.setEnabled(False)

    def accept_photo(self):
        if self.current_step < len(self.steps):
            angle, _ = self.steps[self.current_step]
            self.taken_photos[angle] = self.current_image.copy()
            self.parent().face_recognizer.add_face(self.current_image, self.name, angle)
            self.current_step += 1
            self.current_image = None
            self.accept_button.setEnabled(False)
            self.redo_button.setEnabled(False)
            self.capture_button.setEnabled(True)
            self.update_ui()

    def redo_photo(self):
        self.current_image = None
        self.preview_label.clear()
        self.accept_button.setEnabled(False)
        self.redo_button.setEnabled(False)
        self.capture_button.setEnabled(True)

    def display_image(self, image, label):
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize image to fit the label
        height, width = image_rgb.shape[:2]
        label_size = label.size()
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
            label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        
        label.setPixmap(scaled_pixmap)

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
        
        # Info text
        info_text = ModernLabel(f"Name: {name}\nConfidence: {100 - confidence:.1f}%")
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
        
        # Initialize face recognition
        self.face_recognizer = FaceRecognition("faces/")
        
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
        self.add_button = ModernButton("Add Face")
        self.add_button.clicked.connect(self.add_face)
        self.recognize_button = ModernButton("Recognize Face")
        self.recognize_button.clicked.connect(self.recognize_face)
        self.delete_button = ModernButton("Delete Face")
        self.delete_button.clicked.connect(self.delete_face)
        
        # Add lock button
        self.lock_button = ModernButton("Lock Recognition")
        self.lock_button.setCheckable(True)
        self.lock_button.clicked.connect(self.toggle_lock)
        self.lock_button.setEnabled(False)
        
        middle_layout.addWidget(self.add_button)
        middle_layout.addWidget(self.recognize_button)
        middle_layout.addWidget(self.delete_button)
        middle_layout.addWidget(self.lock_button)
        
        # Right panel (Recognized Faces)
        right_panel = ModernFrame()
        right_layout = QVBoxLayout(right_panel)
        
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
        self.current_image = None
        self.camera_thread = None
        
        # Add recognition timer
        self.recognition_timer = QTimer()
        self.recognition_timer.timeout.connect(self.check_for_faces)
        self.recognition_timer.start(3000)  # Check every 3000ms
        
        # Initialize recognition variables
        self.last_recognition_time = 0
        self.face_detected_time = 0
        self.last_face_detection = 0
        self.is_scanning = True
        self.active_dialogs = 0  # Track number of open dialogs
        
        # Initialize lock variables
        self.is_locked = False
        self.locked_face = None
        self.locked_name = None
        self.locked_confidence = None
        
        # Set dark theme
        self.setStyleSheet("""
            QMainWindow {
                background-color: #212121;
            }
        """)

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
                self.display_image(self.current_image)
                self.status_label.setText("Image loaded successfully")
            else:
                QMessageBox.critical(self, "Error", "Failed to load image")

    def display_image(self, image):
        # Create a copy of the image for drawing
        display_image = image.copy()
        self.detection_counter += 1
        
        # Draw face rectangles if not locked
        if self.detection_counter > 30:
            self.detection_counter = 0
            faces = self.face_recognizer.detect_faces(image)
            results = self.face_recognizer.recognize_face(image)
            recognized_faces = [(x, y, w, h) for _, _, (x, y, w, h) in results]
            
            for (x, y, w, h) in faces:
                # Check if face is recognized
                if (x, y, w, h) in recognized_faces:
                    # Draw green rectangle for recognized faces
                    cv2.rectangle(display_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
                else:
                    # Draw red rectangle for unrecognized faces
                    cv2.rectangle(display_image, (x, y), (x+w, y+h), (0, 0, 255), 2)
        
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
        if self.current_image is None:
            QMessageBox.warning(self, "Warning", "Please load or capture an image first")
            return
            
        name = self.name_entry.text().strip()
        if not name:
            QMessageBox.warning(self, "Warning", "Please enter a name")
            return
            
        # Check if face is already registered
        registered_angles = self.face_recognizer.get_registered_angles(name)
        if registered_angles:
            reply = QMessageBox.question(
                self,
                "Face Already Registered",
                f"Face '{name}' is already registered with {len(registered_angles)} photos. Do you want to add more photos?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.No:
                return
        
        # Start registration dialog
        dialog = FaceRegistrationDialog(self)
        dialog.set_name(name)
        
        # Connect camera frame to dialog
        if self.camera_thread and self.camera_thread.isRunning():
            self.camera_thread.frame_ready.connect(dialog.set_image)
            self.dialog_opened()
            dialog.exec()
            self.dialog_closed()
            self.camera_thread.frame_ready.disconnect(dialog.set_image)
        else:
            self.dialog_opened()
            dialog.set_image(self.current_image)
            dialog.exec()
            self.dialog_closed()
        
        self.name_entry.clear()

    def recognize_face(self):
        if self.current_image is None:
            QMessageBox.warning(self, "Warning", "Please load or capture an image first")
            return
            
        results = self.face_recognizer.recognize_face(self.current_image)
        
        if not results:
            QMessageBox.warning(self, "Warning", "No faces detected or confidence too low")
            return
            
        # Get the best match
        best_result = min(results, key=lambda x: x[1])
        name, confidence, (x, y, w, h) = best_result
        
        if confidence <= 75:  # Lower confidence means better match
            # Extract face region
            face_img = self.current_image[y:y+h, x:x+w]
            # Show popup
            dialog = RecognitionPopupDialog(name, confidence, face_img, self)
            self.dialog_opened()
            dialog.exec()
            self.dialog_closed()
            
            # Enable lock button
            self.lock_button.setEnabled(True)
            # Store locked face data
            self.locked_face = face_img
            self.locked_name = name
            self.locked_confidence = confidence

    def delete_face(self):
        name = self.name_entry.text().strip()
        if not name:
            QMessageBox.warning(self, "Warning", "Please enter a name to delete")
            return
            
        if self.face_recognizer.delete_face(name):
            self.status_label.setText(f"Face deleted successfully for {name}")
            self.name_entry.clear()
        else:
            QMessageBox.critical(self, "Error", "Face not found")

    def toggle_lock(self):
        self.is_locked = self.lock_button.isChecked()
        if not self.is_locked:
            self.locked_face = None
            self.locked_name = None
            self.locked_confidence = None
            self.lock_button.setEnabled(False)

    def display_locked_face(self):
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
        
        # Create name and confidence label
        info_label = ModernLabel(f"{self.locked_name}\nConfidence: {100 - self.locked_confidence:.1f}%\n(Locked)")
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

    def check_for_faces(self):
        if not self.is_scanning or self.current_image is None or self.active_dialogs > 0:
            return
            
        current_time = time.time()
        faces = self.face_recognizer.detect_faces(self.current_image)
        
        if len(faces) > 0:
            if self.last_face_detection == 0:
                self.last_face_detection = current_time
                self.face_detected_time = current_time
            elif current_time - self.face_detected_time >= 1.0:  # Face detected for 1 second
                # Try to recognize the face
                results = self.face_recognizer.recognize_face(self.current_image)
                if results:
                    # Get the result with highest confidence
                    best_result = min(results, key=lambda x: x[1])
                    name, confidence, (x, y, w, h) = best_result
                    if confidence <= 75:  # Lower confidence means better match
                        # Enable lock button
                        self.lock_button.setEnabled(True)
                        # Store locked face data
                        self.locked_face = self.current_image[y:y+h, x:x+w]
                        self.locked_name = name
                        self.locked_confidence = confidence
        else:
            self.last_face_detection = 0
            self.face_detected_time = 0

    def closeEvent(self, event):
        self.stop_camera()
        self.recognition_timer.stop()
        event.accept()

def main():
    app = QApplication(sys.argv)
    window = FaceRecognitionGUI()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()