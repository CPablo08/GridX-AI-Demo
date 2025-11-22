# GridX AI Demo - Jetson Orin Nano Object Detection

A real-time object detection demonstration application for NVIDIA Jetson Orin Nano Super, featuring YOLOv8 AI inference with live camera feed, performance metrics, and comprehensive statistics.

## Features

- **Real-time Object Detection**: YOLOv8-powered detection with bounding boxes and confidence scores
- **Live Camera Feed**: USB webcam support with smooth video display
- **Fullscreen Interface**: Immersive fullscreen display optimized for demonstrations
- **Performance Metrics**: 
  - Real-time FPS counter
  - Inference time tracking
  - GPU utilization monitoring
- **Statistics Dashboard**:
  - Total detections
  - Object counts per class
  - Most common object tracking
  - Detection history
- **High-Tech UI**: Modern terminal-style interface with code-like fonts and green-on-black color scheme
- **Color-Coded Detection**: Each object class gets a unique, consistent color for easy identification

## Requirements

### Hardware
- NVIDIA Jetson Orin Nano Super
- USB webcam
- Display (HDMI or DisplayPort)

### Software
- JetPack 5.x or later (includes PyTorch, CUDA, TensorRT)
- Python 3.8+
- USB webcam drivers

## Installation

### 1. Clone or Download the Project

```bash
cd ~
# If using git:
# git clone <repository-url> "GridX AI Demo App"
# Or extract the project files to ~/GridX\ AI\ Demo\ App/
```

### 2. Install Python Dependencies

```bash
cd "GridX AI Demo App"
pip3 install -r requirements.txt
```

**Note**: On Jetson, some packages may need to be installed via `apt` or may already be included in JetPack:
- PyQt5: `sudo apt install python3-pyqt5` (or use pip)
- OpenCV: Usually pre-installed, but can use `pip install opencv-python`
- PyTorch: Pre-installed with JetPack
- Ultralytics: Will auto-download YOLOv8 models on first run

### 3. Verify Camera Access

```bash
# List available cameras
ls /dev/video*

# Test camera with v4l2
v4l2-ctl --list-devices
```

The application will use `/dev/video0` by default. If your camera is on a different device, you can modify `camera_index` in `main.py`.

## Usage

### Running the Application

```bash
cd "GridX AI Demo App"
python3 main.py
```

The application will:
1. **Automatically optimize** Jetson system settings (power mode, GPU clocks)
2. **Automatically export to TensorRT** on first run (takes a few minutes, then cached)
3. Initialize the camera
4. Load the YOLOv8 model with TensorRT optimization
5. Start detection in a separate thread
6. **Automatically launch in fullscreen** for the demo experience

### Keyboard Controls

- **ESC**: Exit the application
- **F11**: Toggle fullscreen/windowed mode

### First Run

On the first run:
1. **YOLOv8 model download**: Automatically downloads (approximately 6MB for YOLOv8n)
2. **TensorRT export**: Automatically exports to TensorRT engine format (takes 2-5 minutes, but only happens once)
3. **Jetson optimization**: Automatically sets power mode and GPU clocks for maximum performance
4. **Fullscreen launch**: Automatically launches in fullscreen mode

**Note**: The TensorRT export only happens once. Subsequent runs will load the pre-exported engine instantly.

## Configuration

### Camera Settings

Edit `main.py` to change camera settings:

```python
self.camera = Camera(camera_index=0, width=1280, height=720)
```

- `camera_index`: Camera device number (0, 1, 2, etc.)
- `width`, `height`: Desired resolution

### Detection Settings

Edit `main.py` to adjust detection parameters:

```python
self.detector = Detector(model_name='yolov8n.pt', confidence_threshold=0.25)
```

- `model_name`: 
  - `'yolov8n.pt'` - Nano (fastest, lower accuracy)
  - `'yolov8s.pt'` - Small (balanced)
  - `'yolov8m.pt'` - Medium (slower, higher accuracy)
  - `'yolov8l.pt'` - Large (much slower)
  - `'yolov8x.pt'` - Extra Large (slowest, highest accuracy)

- `confidence_threshold`: Minimum confidence (0.0 to 1.0) for detections

## TensorRT Optimization (Automatic)

The application automatically exports and uses TensorRT on Jetson for maximum performance:

- **First Run**: Automatically exports YOLOv8 model to TensorRT engine format (takes a few minutes)
- **Subsequent Runs**: Automatically loads the pre-exported TensorRT engine for fast startup
- **Performance**: 2-3x faster inference compared to standard PyTorch
- **Location**: TensorRT engine files (`.engine`) are saved alongside the model files

The TensorRT optimization happens automatically - no manual steps required!

## Performance Tips

1. **Automatic Optimizations**: The app automatically:
   - Sets Jetson to MAXN power mode (maximum performance)
   - Sets GPU clocks to maximum (`jetson_clocks`)
   - Uses TensorRT for 2-3x faster inference
   - Uses FP16 precision for better performance
   - Optimizes camera buffer for low latency

2. **Use YOLOv8n**: The nano model provides the best balance of speed and accuracy for real-time demos

3. **Fullscreen Mode**: Automatically launches in fullscreen for best demo experience

4. **Close Other Apps**: Free up GPU memory by closing other applications

5. **First Run**: TensorRT export takes a few minutes but only happens once

## Troubleshooting

### Camera Not Found

```bash
# Check camera devices
ls -la /dev/video*

# Test with OpenCV
python3 -c "import cv2; cap = cv2.VideoCapture(0); print('Camera OK' if cap.isOpened() else 'Camera FAILED')"
```

### Low FPS

- Reduce camera resolution in `main.py`
- Use YOLOv8n instead of larger models
- Check GPU utilization (should be displayed in stats panel)
- Ensure no other processes are using the GPU

### Model Download Issues

If automatic model download fails, manually download:

```bash
# Create models directory
mkdir -p models

# Download YOLOv8n (example)
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt -O models/yolov8n.pt
```

### PyQt5 Installation Issues

On some Jetson systems, PyQt5 may need to be installed via apt:

```bash
sudo apt update
sudo apt install python3-pyqt5 python3-pyqt5.qtwidgets
```

## Project Structure

```
GridX AI Demo App/
├── main.py                 # Main application entry point
├── app/
│   ├── __init__.py
│   ├── camera.py          # Camera capture module
│   ├── detector.py        # YOLOv8 detection engine
│   ├── gui.py             # PyQt5 GUI components
│   └── utils.py           # Utility functions (FPS, stats, etc.)
├── models/                 # YOLO model storage (auto-created)
├── requirements.txt       # Python dependencies
└── README.md              # This file
```

## Competition Demo Tips

1. **Pre-test Everything**: Test camera, lighting, and positioning before the demo
2. **Interesting Objects**: Have various objects ready (phones, bottles, cups, books, etc.)
3. **Lighting**: Ensure good lighting for better detection accuracy
4. **Performance**: Use YOLOv8n for smooth real-time performance
5. **Backup Plan**: Have a pre-recorded video ready in case of camera issues
6. **Explain Features**: Point out the statistics panel, FPS counter, and color coding

## License

This project is provided as-is for demonstration purposes.

## Credits

- **YOLOv8**: Ultralytics (https://ultralytics.com)
- **PyQt5**: The Qt Company
- **OpenCV**: OpenCV Foundation
- **NVIDIA Jetson**: NVIDIA Corporation

## Support

For Jetson-specific issues, refer to:
- NVIDIA Jetson Developer Forums
- JetsonHacks tutorials
- Ultralytics YOLOv8 documentation

---

**GridX AI Demo** - Showcasing the power of edge AI on NVIDIA Jetson Orin Nano Super

