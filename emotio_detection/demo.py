#!/usr/bin/env python3
"""
Demo script for the emotion detection project.
This script demonstrates the basic functionality without requiring a trained model.
"""

import cv2
import numpy as np
import os
import sys
from threading import Timer
import time

# Add the parent directory to the path so we can import the modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import models from the package
try:
    from .dnn import CNN
    from .mlmodel import SVM, RF, NN
except ImportError:
    # Fallback for direct script execution
    from dnn import CNN
    from mlmodel import SVM, RF, NN

def demo_face_detection():
    """Demo face detection using OpenCV"""
    print("=== Face Detection Demo ===")
    
    # Load the face detection model
    face_model_path = os.path.join(os.getcwd(), "..", "face_detector")
    prototxtPath = os.path.sep.join([face_model_path, "deploy.prototxt"])
    weightsPath = os.path.sep.join([face_model_path, "res10_300x300_ssd_iter_140000.caffemodel"])
    
    print(f"Loading face detector model from {face_model_path}")
    faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
    print("Face detector model loaded successfully!")
    
    # Try to open camera
    print("Attempting to open camera...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera. Please make sure a camera is connected.")
        return
    
    print("Camera opened successfully!")
    print("Press 'q' to quit the demo")
    print("This demo shows face detection without emotion recognition")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from camera")
            break
            
        # Convert to grayscale for better processing
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
        faceNet.setInput(blob)
        detections = faceNet.forward()
        
        # Draw bounding boxes around detected faces
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:  # Confidence threshold
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                
                # Draw rectangle around face
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                cv2.putText(frame, f"Face: {confidence:.2f}", (startX, startY - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
        
        # Display frame
        cv2.imshow("Face Detection Demo", frame)
        
        # Break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("Demo completed!")

def demo_model_creation():
    """Demo creating different ML models"""
    print("\n=== Model Creation Demo ===")
    
    # Create sample data for demonstration
    print("Creating sample data...")
    X_train = np.random.rand(100, 10)  # 100 samples, 10 features
    y_train = np.random.randint(0, 3, 100)  # 3 classes
    
    print("Creating different machine learning models:")
    
    # SVM Model
    print("1. Creating SVM model...")
    svm_model = SVM()
    print(f"   Model name: {svm_model.name}")
    
    # Random Forest Model
    print("2. Creating Random Forest model...")
    rf_model = RF()
    print(f"   Model name: {rf_model.name}")
    
    # Neural Network Model
    print("3. Creating Neural Network model...")
    nn_model = NN()
    print(f"   Model name: {nn_model.name}")
    
    # CNN Model (for image data)
    print("4. Creating CNN model...")
    try:
        cnn_model = CNN(input_shape=(224, 224, 1), num_classes=3)
        print(f"   Model name: {cnn_model.name}")
        print(f"   Input shape: {cnn_model.input_shape}")
    except Exception as e:
        print(f"   CNN model creation failed: {e}")
    
    print("Model creation demo completed!")

def demo_dataset_creation():
    """Demo dataset creation process"""
    print("\n=== Dataset Creation Demo ===")
    print("This project can create datasets for training emotion detection models.")
    print("The dataset creation process involves:")
    print("1. Capturing facial expressions using the camera")
    print("2. Saving images to labeled folders (happy, sad, neutral, angry)")
    print("3. Using keyboard keys to capture different expressions:")
    print("   - 'h' for happy")
    print("   - 's' for sad") 
    print("   - 'n' for neutral")
    print("   - 'a' for angry")
    print("   - 'q' to quit")
    print("\nTo run the dataset creation script:")
    print("python create_dataset.py")

def demo_training_process():
    """Demo training process"""
    print("\n=== Training Process Demo ===")
    print("The training process involves:")
    print("1. Loading the dataset from the 'dataset' folder")
    print("2. Preprocessing images (resize to 224x224, normalize)")
    print("3. Splitting data into training and testing sets")
    print("4. Training a MobileNetV2 model with custom head")
    print("5. Saving the trained model to 'model/emotion_model.h5'")
    print("\nTo run the training script:")
    print("python train_emotion_detector.py")

def main():
    """Main demo function"""
    print("Emotion Detection Project Demo")
    print("=" * 40)
    
    while True:
        print("\nSelect a demo option:")
        print("1. Face Detection Demo")
        print("2. Model Creation Demo") 
        print("3. Dataset Creation Info")
        print("4. Training Process Info")
        print("5. Exit")
        
        choice = input("Enter your choice (1-5): ").strip()
        
        if choice == '1':
            demo_face_detection()
        elif choice == '2':
            demo_model_creation()
        elif choice == '3':
            demo_dataset_creation()
        elif choice == '4':
            demo_training_process()
        elif choice == '5':
            print("Exiting demo...")
            break
        else:
            print("Invalid choice. Please enter a number between 1-5.")

if __name__ == "__main__":
    main()