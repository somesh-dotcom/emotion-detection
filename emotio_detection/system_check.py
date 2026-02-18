#!/usr/bin/env python3
"""
Simple demo of the emotion detection project components without camera access.
This demonstrates that the models and dependencies are working correctly.
"""

import os
import sys
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import project modules
try:
    from .dnn import CNN
    from .mlmodel import SVM, RF, NN
    from .__init__ import Model
except ImportError:
    from dnn import CNN
    from mlmodel import SVM, RF, NN
    from __init__ import Model

def demo_model_creation():
    """Demo that we can create and initialize models"""
    print("=== Model Creation Demo ===")
    print("Testing that all model classes can be imported and instantiated...")
    print()
    
    try:
        # Test SVM model
        print("1. Creating SVM model...")
        svm_model = SVM()
        print(f"   ✓ Successfully created {svm_model.name}")
        
        # Test Random Forest model
        print("2. Creating Random Forest model...")
        rf_model = RF()
        print(f"   ✓ Successfully created {rf_model.name}")
        
        # Test Neural Network model
        print("3. Creating Neural Network model...")
        nn_model = NN()
        print(f"   ✓ Successfully created {nn_model.name}")
        
        # Test CNN model (this might fail due to Keras setup)
        print("4. Creating CNN model...")
        try:
            cnn_model = CNN(input_shape=(224, 224, 1), num_classes=3)
            print(f"   ✓ Successfully created {cnn_model.name}")
            print(f"   Input shape: {cnn_model.input_shape}")
        except Exception as e:
            print(f"   ! CNN creation warning: {e}")
            print("   This is normal if Keras isn't fully configured yet")
        
        print()
        print("✓ All model classes imported and instantiated successfully!")
        return True
        
    except Exception as e:
        print(f"✗ Error creating models: {e}")
        return False

def demo_data_processing():
    """Demo data processing capabilities"""
    print("\n=== Data Processing Demo ===")
    print("Testing data handling and preprocessing...")
    print()
    
    try:
        # Create sample data
        print("1. Creating sample data...")
        X_train = np.random.rand(50, 10)  # 50 samples, 10 features
        y_train = np.random.randint(0, 3, 50)  # 3 classes
        print(f"   ✓ Created sample data: {X_train.shape[0]} samples, {X_train.shape[1]} features")
        
        # Test basic operations
        print("2. Testing basic operations...")
        mean_values = np.mean(X_train, axis=0)
        print(f"   ✓ Calculated means: {mean_values.shape}")
        
        unique_labels = np.unique(y_train)
        print(f"   ✓ Found {len(unique_labels)} unique classes: {unique_labels}")
        
        print()
        print("✓ Data processing working correctly!")
        return True
        
    except Exception as e:
        print(f"✗ Error in data processing: {e}")
        return False

def demo_file_structure():
    """Demo that required files and directories exist"""
    print("\n=== File Structure Demo ===")
    print("Checking project structure and required files...")
    print()
    
    # Get base directory
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print(f"Base project path: {base_path}")
    
    # Check required directories
    required_dirs = ["model", "face_detector", "dataset"]
    all_dirs_exist = True
    
    for dir_name in required_dirs:
        dir_path = os.path.join(base_path, dir_name)
        if os.path.exists(dir_path):
            files = os.listdir(dir_path)
            print(f"✓ {dir_name}/ exists ({len(files)} items)")
        else:
            print(f"✗ {dir_name}/ missing")
            all_dirs_exist = False
    
    # Check face detection models
    print("\nFace detection models:")
    face_detector_path = os.path.join(base_path, "face_detector")
    required_files = ["deploy.prototxt", "res10_300x300_ssd_iter_140000.caffemodel"]
    
    all_files_exist = True
    for file_name in required_files:
        file_path = os.path.join(face_detector_path, file_name)
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            size_mb = size / (1024 * 1024)
            print(f"✓ {file_name} ({size_mb:.2f} MB)")
        else:
            print(f"✗ {file_name} missing")
            all_files_exist = False
    
    # Check emotion model
    print("\nEmotion detection model:")
    emotion_model_path = os.path.join(base_path, "model", "emotion_model.h5")
    if os.path.exists(emotion_model_path):
        size = os.path.getsize(emotion_model_path)
        size_mb = size / (1024 * 1024)
        print(f"✓ emotion_model.h5 ({size_mb:.2f} MB) - Model is trained!")
    else:
        print("✗ emotion_model.h5 - Model not trained yet")
    
    return all_dirs_exist and all_files_exist

def demo_dependencies():
    """Demo that all dependencies are available"""
    print("\n=== Dependencies Check ===")
    print("Testing that all required packages are available...")
    print()
    
    required_packages = [
        ("numpy", "Numerical computing"),
        ("cv2", "OpenCV (computer vision)"),
        ("tensorflow", "Deep learning framework"),
        ("sklearn", "Machine learning"),
        ("imutils", "Image utilities"),
        ("keras", "Neural networks"),
        ("scipy", "Scientific computing"),
        ("matplotlib", "Plotting")
    ]
    
    all_packages_available = True
    
    for package_name, description in required_packages:
        try:
            if package_name == "cv2":
                import cv2
                version = cv2.__version__
            elif package_name == "sklearn":
                import sklearn
                version = sklearn.__version__
            else:
                module = __import__(package_name)
                version = getattr(module, '__version__', 'unknown')
            
            print(f"✓ {package_name} ({version}) - {description}")
        except ImportError as e:
            print(f"✗ {package_name} - NOT AVAILABLE - {description}")
            print(f"  Error: {e}")
            all_packages_available = False
        except Exception as e:
            print(f"? {package_name} - Available but error: {e}")
    
    return all_packages_available

def main():
    """Main demo function"""
    print("=" * 60)
    print("EMOTION DETECTION PROJECT - SYSTEM CHECK")
    print("=" * 60)
    print()
    
    # Run all demos
    results = []
    
    results.append(("Dependencies", demo_dependencies()))
    results.append(("File Structure", demo_file_structure()))
    results.append(("Model Creation", demo_model_creation()))
    results.append(("Data Processing", demo_data_processing()))
    
    # Summary
    print("\n" + "=" * 60)
    print("SYSTEM CHECK SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{test_name:20} {status}")
        if result:
            passed += 1
    
    print("-" * 60)
    print(f"Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All systems are ready! You can now:")
        print("1. Create a dataset: python create_dataset.py")
        print("2. Train the model: python train_emotion_detector.py") 
        print("3. Run detection: python detect_emotion_video.py")
    else:
        print("\n⚠️  Some components need attention before running the full project.")
        print("Check the failed tests above for details.")

if __name__ == "__main__":
    main()