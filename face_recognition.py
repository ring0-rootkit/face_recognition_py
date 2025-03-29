import cv2
import numpy as np
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from typing import List, Tuple, Dict, Optional
import json
import time

class GhostNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(GhostNet, self).__init__()
        # GhostNet architecture implementation
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        
        # Ghost blocks
        self.ghost_blocks = nn.Sequential(
            self._make_ghost_block(16, 16, 1),
            self._make_ghost_block(16, 24, 2),
            self._make_ghost_block(24, 24, 1),
            self._make_ghost_block(24, 40, 2),
            self._make_ghost_block(40, 40, 1),
            self._make_ghost_block(40, 80, 2),
            self._make_ghost_block(80, 80, 1),
            self._make_ghost_block(80, 80, 1),
            self._make_ghost_block(80, 112, 2),
            self._make_ghost_block(112, 112, 1),
            self._make_ghost_block(112, 160, 2),
            self._make_ghost_block(160, 160, 1),
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(160, num_classes)
        
    def _make_ghost_block(self, in_channels, out_channels, stride):
        return nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, stride, 1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.ghost_blocks(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

class GhostFaceNets:
    def __init__(self, database_path: str):
        self.database_path = database_path
        self.face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = GhostNet(num_classes=512)  # 512-dimensional face embedding
        self.model.to(self.device)
        self.model.eval()
        
        # Load pre-trained weights if available
        weights_path = os.path.join(database_path, 'ghostface_weights.pth')
        if os.path.exists(weights_path):
            self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
        
        self.transform = transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Create database directory if it doesn't exist
        os.makedirs(database_path, exist_ok=True)
        
        # Load metadata
        self.metadata_file = os.path.join(database_path, 'metadata.json')
        self.metadata = self._load_metadata()
        
    def _load_metadata(self) -> Dict:
        if os.path.exists(self.metadata_file):
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {}
        
    def _save_metadata(self):
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f)
            
    def _get_face_embedding(self, face_img: np.ndarray) -> np.ndarray:
        # Convert BGR to RGB
        face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        face_pil = Image.fromarray(face_rgb)
        
        # Apply transforms
        face_tensor = self.transform(face_pil).unsqueeze(0).to(self.device)
        
        # Get embedding
        with torch.no_grad():
            embedding = self.model(face_tensor)
            embedding = embedding.cpu().numpy()
            
        return embedding[0]
        
    def detect_faces(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces in the image
        Returns: List of tuples (x, y, w, h) representing face locations
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector.detectMultiScale(gray, 1.1, 4)
        return faces.tolist()
        
    def add_face(self, image: np.ndarray, name: str, angle: str) -> bool:
        """
        Add a face to the database
        Returns: True if successful, False otherwise
        """
        faces = self.detect_faces(image)
        if len(faces) == 0:
            return False
            
        # Get the largest face
        face = max(faces, key=lambda x: x[2] * x[3])
        x, y, w, h = face
        face_img = image[y:y+h, x:x+w]
        
        # Get face embedding
        embedding = self._get_face_embedding(face_img)
        
        # Save face image
        face_dir = os.path.join(self.database_path, name)
        os.makedirs(face_dir, exist_ok=True)
        face_path = os.path.join(face_dir, f"{angle}.jpg")
        cv2.imwrite(face_path, face_img)
        
        # Save embedding
        embedding_path = os.path.join(face_dir, f"{angle}_embedding.npy")
        np.save(embedding_path, embedding)
        
        # Update metadata
        if name not in self.metadata:
            self.metadata[name] = {}
        self.metadata[name][angle] = {
            'image_path': face_path,
            'embedding_path': embedding_path
        }
        self._save_metadata()
        
        return True
        
    def recognize_face(self, image: np.ndarray) -> List[Tuple[str, float, Tuple[int, int, int, int]]]:
        """
        Recognize faces in the image
        Returns: List of tuples (name, confidence, face_location) or empty list if no faces recognized
        """
        faces = self.detect_faces(image)
        if len(faces) == 0:
            return []
            
        results = []
        for (x, y, w, h) in faces:
            face_img = image[y:y+h, x:x+w]
            face_embedding = self._get_face_embedding(face_img)
            
            best_match = None
            best_confidence = float('inf')
            
            # Compare with all registered faces
            for name, angles in self.metadata.items():
                for angle, data in angles.items():
                    stored_embedding = np.load(data['embedding_path'])
                    # Calculate cosine distance (lower is better)
                    distance = 1 - np.dot(face_embedding, stored_embedding) / (
                        np.linalg.norm(face_embedding) * np.linalg.norm(stored_embedding)
                    )
                    
                    if distance < best_confidence:
                        best_confidence = distance
                        best_match = name
            
            if best_match is not None:
                results.append((best_match, best_confidence, (x, y, w, h)))
                
        return results
        
    def delete_face(self, name: str) -> bool:
        """
        Delete a face from the database
        Returns: True if successful, False otherwise
        """
        if name not in self.metadata:
            return False
            
        # Delete face images and embeddings
        for angle, data in self.metadata[name].items():
            if os.path.exists(data['image_path']):
                os.remove(data['image_path'])
            if os.path.exists(data['embedding_path']):
                os.remove(data['embedding_path'])
                
        # Delete face directory
        face_dir = os.path.join(self.database_path, name)
        if os.path.exists(face_dir):
            os.rmdir(face_dir)
            
        # Update metadata
        del self.metadata[name]
        self._save_metadata()
        
        return True
        
    def get_registered_angles(self, name: str) -> List[str]:
        """
        Get list of registered angles for a face
        Returns: List of angle names or empty list if face not found
        """
        return list(self.metadata.get(name, {}).keys()) 