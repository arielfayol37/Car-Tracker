# Car Tracker

A robust multi-object tracking system for detecting and tracking vehicles in roundabout video footage using a combination of Convolutional Neural Networks (CNN) and Recurrent Neural Networks (RNN).

## Overview

This project combines state-of-the-art object detection (YOLOv8) with custom RNN-based tracking to follow vehicles through successive video frames, even during occlusions or motion blur. The system predicts vehicle positions using an RNN and matches identities using a CNN for robust multi-object tracking.

### Key Features

- **Real-time vehicle detection** using YOLOv8
- **Temporal tracking** with LSTM-based trajectory prediction
- **Visual re-identification** using Siamese CNN
- **Robust handling** of occlusions and motion blur
- **Multi-object tracking** with consistent ID assignment

## Architecture

### Object Detector (CNN)
- **YOLOv8**: Real-time object detection providing bounding box coordinates (x, y, width, height) and class confidences
- Pre-trained on COCO dataset for vehicle detection
- Serves as the "eyes" of the system, spotting vehicles in each frame

### Re-identification CNN
- **Siamese Network**: Compares two car images and outputs similarity scores
- **Architecture**: Twin branches of convolutional layers (3 Conv2D layers with max-pooling)
- **Output**: Similarity score (0-1) indicating whether images depict the same vehicle
- **Purpose**: Ensures consistent ID assignment even when vehicles come close or swap positions

### Trajectory RNN (LSTM)
- **Core Component**: LSTM network modeling temporal dynamics of vehicle movement
- **Input**: Sequence of past bounding box coordinates for each car
- **Output**: Predicted future positions (next 8 frames)
- **Architecture**: 
  - Single LSTM layer with 64 hidden units
  - Two Dense layers: Dense(64, ReLU) + Dense(32)
  - Context window: 20 frames (CONTEXT_SIZE = 20)
  - Prediction window: 8 frames (PREDICT_SIZE = 8)

## Data Processing Pipeline

### Frame Processing
1. **Read frame** from input video
2. **Apply YOLOv8** for vehicle detection
3. **Filter detections** by confidence threshold (30%)
4. **Format detections** as [x, y, w, h] coordinates
5. **Send to tracker** for processing

### Tracking Algorithm
The tracker maintains internal data structures for all active tracks:

- **cars_info**: Dictionary storing track information
- **coords**: List of recent coordinates for each car
- **predicted**: Array of future predicted coordinates
- **time_elapsed**: Counter for track retirement
- **time_update_pred**: Counter for RNN prediction updates

#### Update Process
1. **Matching**: For each detection, compute IoU overlap and CNN similarity scores
2. **Association**: Assign detection to best matching track or create new track
3. **RNN Prediction**: Update predicted positions every 8 frames
4. **Visualization**: Draw tracking annotations on output frame
5. **Output**: Save annotated video with tracking results

## Installation

### Prerequisites
- Python 3.9+
- Required libraries: ultralytics, opencv-python, numpy, tensorflow, keras

### Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install ultralytics==8.0.20 opencv-python numpy tensorflow==2.9 keras==2.9
```

## Usage

### Running Vehicle Tracking

1. **Prepare video**: Place your video file in the solution folder
2. **Configure paths**: Edit `car_tracking.py` and update the `filepaths` list
   ```python
   filepaths = ["your_video_name"]  # without extension
   ```
3. **Run tracking**:
   ```bash
   python car_tracking.py
   ```

The script will:
- Load YOLOv8 model (downloads automatically on first run)
- Process video frame by frame
- Display progress every 10% of frames
- Save output video with tracking annotations (e.g., `roundabout_1_with_rnn_v1.mp4`)

### Model Training (Optional)

If you need to retrain the models:

#### CNN Re-identification Training
```bash
# Generate training data
python cnn_generate_data.py

# Train Siamese CNN
python best_cnn_training.py
```

#### RNN Trajectory Training
```bash
# Generate trajectory data
python rnn_generate_data.py

# Train LSTM model
python best_rnn_training.py
```

After training, place the weight files in `solution/ai_weights/`:
- `cnn_reidentification_weights.h5`
- `car_trajectory_best_validation_final_weights.h5`

## File Structure

```
solution/
├── car_tracking.py              # Main tracking script
├── my_tracker.py               # Tracker implementation
├── ai_weights/                 # Pre-trained model weights
│   ├── cnn_reidentification_weights.h5
│   └── car_trajectory_best_validation_final_weights.h5
├── best_cnn_training.py        # CNN training script
├── best_rnn_training.py        # RNN training script
├── cnn_generate_data.py        # CNN data generation
└── rnn_generate_data.py        # RNN data generation
```

## Troubleshooting

### YOLO Model Issues
- Ensure `ultralytics` package is properly installed
- Check model name in code (default: `yolov8s.pt`)
- Verify internet connection for first-time model download

### Re-identification CNN Issues
- If you don't want to use re-identification, simplify the tracker to rely on motion only
- Skip loading `cnn_reidentification_weights.h5`
- Adjust matching to use IoU and predicted location only

### Performance Optimization
- Adjust confidence thresholds for detection filtering
- Modify prediction frequency by changing `PREDICT_SIZE`
- Tune CNN similarity thresholds for re-identification

## Technical Details

### Model Specifications
- **YOLOv8**: Pre-trained on COCO dataset
- **Siamese CNN**: 3 Conv2D layers with max-pooling
- **LSTM**: 64 hidden units, 20-frame context, 8-frame prediction
- **Loss Function**: Mean-squared error for trajectory prediction

### Performance Metrics
- Real-time processing capability
- Robust tracking through occlusions
- Consistent ID assignment
- Accurate trajectory prediction

## Contributing

This project demonstrates the effectiveness of combining CNN-based detection with RNN-based prediction for robust multi-object tracking. The architecture can be extended to other tracking applications by adapting the model architectures and training procedures.
