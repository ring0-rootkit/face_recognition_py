import os
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras
from keras_cv_attention_models import ghostnet
from keras_cv_attention_models.ghostnet import GhostNet
import cv2
from tqdm import tqdm
import pickle
import argparse

def load_lfw_dataset(lfw_path):
    """Load LFW dataset from binary file"""
    with open(lfw_path, 'rb') as f:
        data = pickle.load(f)
    return data

def preprocess_image(image):
    """Preprocess image for training"""
    # Resize to 112x112
    image = cv2.resize(image, (112, 112))
    # Normalize to [-1, 1]
    image = (image.astype(np.float32) - 127.5) / 128.0
    return image

def create_training_data(lfw_data):
    """Create training data from LFW dataset"""
    X = []
    y = []
    
    for person_name, images in lfw_data.items():
        for image in images:
            # Preprocess image
            processed_image = preprocess_image(image)
            X.append(processed_image)
            y.append(person_name)
    
    return np.array(X), np.array(y)

def create_model(num_classes):
    """Create GhostFaceNet model"""
    # Create base GhostNet model
    base_model = GhostNet(
        input_shape=(112, 112, 3),
        num_classes=num_classes,
        pretrained=None
    )
    
    # Add ArcFace layer
    x = base_model.output
    x = keras.layers.Dense(512, activation='relu')(x)
    x = keras.layers.Dense(num_classes, activation='softmax')(x)
    
    model = keras.Model(inputs=base_model.input, outputs=x)
    return model

def train_model(model, X_train, y_train, epochs=100, batch_size=32):
    """Train the model"""
    # Convert labels to one-hot encoding
    y_train = keras.utils.to_categorical(y_train, num_classes=len(set(y_train)))
    
    # Compile model
    optimizer = tfa.optimizers.AdamW(learning_rate=0.001, weight_decay=5e-5)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Create callbacks
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            'ghostface_weights.h5',
            save_best_only=True,
            monitor='val_accuracy'
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True
        )
    ]
    
    # Train model
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        callbacks=callbacks
    )
    
    return history

def main():
    parser = argparse.ArgumentParser(description='Train GhostFaceNet on LFW dataset')
    parser.add_argument('--lfw_path', type=str, required=True, help='Path to LFW dataset binary file')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    args = parser.parse_args()
    
    # Load dataset
    print("Loading LFW dataset...")
    lfw_data = load_lfw_dataset(args.lfw_path)
    
    # Create training data
    print("Preparing training data...")
    X_train, y_train = create_training_data(lfw_data)
    
    # Create and train model
    print("Creating model...")
    model = create_model(num_classes=len(set(y_train)))
    
    print("Starting training...")
    history = train_model(
        model,
        X_train,
        y_train,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
    
    print("Training completed!")

if __name__ == "__main__":
    main() 