#!/usr/bin/env python3
"""
Simple demo script for the emotion detection project.
This script shows information about the project and how to run it.
"""

import os
import sys

def show_project_info():
    """Show information about the emotion detection project"""
    print("=" * 50)
    print("EMOTION DETECTION PROJECT")
    print("=" * 50)
    print()
    print("This project detects facial expressions (emotions) using:")
    print("- OpenCV for face detection")
    print("- TensorFlow/Keras for emotion recognition")
    print("- Deep learning models (CNN, MobileNetV2)")
    print()
    print("Supported emotions: Happy, Sad, Neutral, Angry")
    print()

def show_setup_info():
    """Show setup information"""
    print("SETUP INFORMATION")
    print("-" * 20)
    print("Required directories:")
    print("- model/ (for trained emotion models)")
    print("- face_detector/ (for face detection models)")
    print("- dataset/ (for training data)")
    print()
    print("Required files in face_detector/:")
    print("- deploy.prototxt")
    print("- res10_300x300_ssd_iter_140000.caffemodel")
    print()
    print("Dependencies installed:")
    print("- tensorflow")
    print("- opencv-python")
    print("- imutils")
    print("- numpy")
    print("- scikit-learn")
    print("- etc.")
    print()

def show_how_to_run():
    """Show how to run the different scripts"""
    print("HOW TO RUN THE PROJECT")
    print("-" * 25)
    print()
    print("1. CREATE DATASET:")
    print("   python create_dataset.py")
    print("   - Captures facial expressions using your camera")
    print("   - Saves images to dataset folders")
    print("   - Use keys: h(Happy), s(Sad), n(Neutral), a(Angry), q(Quit)")
    print()
    print("2. TRAIN MODEL:")
    print("   python train_emotion_detector.py")
    print("   - Trains emotion detection model on your dataset")
    print("   - Saves trained model to model/emotion_model.h5")
    print("   - Shows training progress and accuracy")
    print()
    print("3. DETECT EMOTIONS:")
    print("   python detect_emotion_video.py")
    print("   - Runs real-time emotion detection")
    print("   - Shows video with emotion labels")
    print("   - Requires trained model (emotion_model.h5)")
    print()
    print("4. RUN THIS DEMO:")
    print("   python project_info.py")
    print("   - Shows this information")
    print()

def show_current_status():
    """Show current project status"""
    print("CURRENT PROJECT STATUS")
    print("-" * 25)
    
    # Check directories
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    dirs = ["model", "face_detector", "dataset"]
    for dir_name in dirs:
        dir_path = os.path.join(base_path, dir_name)
        if os.path.exists(dir_path):
            files = os.listdir(dir_path)
            print(f"✓ {dir_name}/ exists ({len(files)} files)")
        else:
            print(f"✗ {dir_name}/ missing")
    
    # Check face detector files
    face_detector_path = os.path.join(base_path, "face_detector")
    required_files = ["deploy.prototxt", "res10_300x300_ssd_iter_140000.caffemodel"]
    
    print("\nFace detector files:")
    for file_name in required_files:
        file_path = os.path.join(face_detector_path, file_name)
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            print(f"✓ {file_name} ({size} bytes)")
        else:
            print(f"✗ {file_name} missing")
    
    # Check model file
    model_path = os.path.join(base_path, "model", "emotion_model.h5")
    print(f"\nEmotion model:")
    if os.path.exists(model_path):
        size = os.path.getsize(model_path)
        print(f"✓ emotion_model.h5 ({size} bytes)")
    else:
        print("✗ emotion_model.h5 (not trained yet)")
    
    print()

def main():
    """Main function"""
    show_project_info()
    show_setup_info()
    show_current_status()
    show_how_to_run()
    
    print("Ready to start working with the emotion detection project!")
    print("Run one of the scripts above to get started.")

if __name__ == "__main__":
    main()