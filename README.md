# Emotion Detection Project

This project detects facial expressions (emotions) in real-time using computer vision and deep learning techniques.

## Features

- Real-time face detection using OpenCV
- Emotion recognition for 4 emotions: Happy, Sad, Neutral, Angry
- Deep learning models (CNN, MobileNetV2)
- Interactive dataset creation tool
- Model training and evaluation

## Requirements

- Python 3.7+
- Camera (webcam)
- The following Python packages:
  - tensorflow>=2.12.0
  - opencv-python==4.8.0.74
  - imutils==0.5.4
  - numpy==1.24.3
  - scipy==1.10.1
  - speechpy==2.4
  - scikit-learn==1.3.0
  - matplotlib==3.7.2

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. The required directories will be created automatically:
   - `model/` - for trained emotion models
   - `face_detector/` - for face detection models
   - `dataset/` - for training data

3. Face detection models are automatically downloaded during setup

## How to Use

### 1. View Project Information
```bash
python project_info.py
```
Shows project status and usage instructions.

### 2. Create Dataset
```bash
python create_dataset.py
```
- Captures facial expressions using your camera
- Saves images to labeled folders in the dataset directory
- Controls:
  - `h` - Capture happy expression
  - `s` - Capture sad expression
  - `n` - Capture neutral expression
  - `a` - Capture angry expression
  - `q` - Quit

### 3. Train Model
```bash
python train_emotion_detector.py
```
- Trains emotion detection model on your dataset
- Uses MobileNetV2 with custom head
- Saves trained model to `model/emotion_model.h5`
- Shows training progress, accuracy, and loss graphs

### 4. Detect Emotions
```bash
python detect_emotion_video.py
```
- Runs real-time emotion detection
- Shows video feed with emotion labels
- Displays bounding boxes around detected faces
- Colors indicate emotions:
  - Green: Happy
  - White: Neutral
  - Blue: Sad

## Project Structure

```
emotion-detection/
├── emotio_detection/
│   ├── __init__.py              # Base model classes
│   ├── create_dataset.py        # Dataset creation tool
│   ├── detect_emotion_video.py  # Real-time emotion detection
│   ├── dnn.py                   # Deep neural network models
│   ├── mlmodel.py               # Traditional ML models
│   ├── train_emotion_detector.py # Model training script
│   ├── utilities.py             # Utility functions
│   ├── demo.py                  # Interactive demo
│   └── project_info.py          # Project information
├── model/                       # Trained models (created during training)
├── face_detector/               # Face detection models
├── dataset/                     # Training data (created during dataset creation)
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## How It Works

1. **Face Detection**: Uses OpenCV's DNN module with a pre-trained SSD model to detect faces
2. **Preprocessing**: Detected faces are resized to 224x224 pixels and normalized
3. **Emotion Recognition**: A MobileNetV2 model with a custom classification head predicts emotions
4. **Real-time Processing**: The system processes video frames in real-time and displays results

## Model Architecture

The emotion detection model uses:
- **Base**: MobileNetV2 (pre-trained on ImageNet)
- **Custom Head**: 
  - AveragePooling2D
  - Flatten
  - Dense(128) with ReLU
  - Dropout(0.5)
  - Dense(3) with Softmax (for 3 emotions)

## Training Process

1. Dataset images are preprocessed and split into training/validation sets
2. Data augmentation is applied (zoom, shifts, shear)
3. Transfer learning is used (MobileNetV2 base is frozen)
4. Only the custom head is trained
5. Model is evaluated and saved

## Troubleshooting

### Common Issues

1. **Camera not detected**: Make sure your camera is connected and not in use by another application
2. **Model not found**: Run `train_emotion_detector.py` first to create the model
3. **Import errors**: Make sure all dependencies are installed correctly
4. **Poor detection accuracy**: Try creating a larger, more diverse dataset

### Performance Tips

- Ensure good lighting when creating your dataset
- Capture expressions from different angles and distances
- Include diverse subjects in your training data
- Train for more epochs if needed (modify EPOCHS in train_emotion_detector.py)

## License

This project is for educational purposes. Feel free to modify and extend it for your own use.