"""YOLOv8 object detection module with TensorRT optimization support."""

import os
import time
from typing import List, Dict, Optional
import numpy as np
from ultralytics import YOLO
import torch


class Detector:
    """YOLOv8 object detector with TensorRT optimization for Jetson."""
    
    def __init__(self, model_name: str = 'yolov8n.pt', confidence_threshold: float = 0.25):
        """Initialize detector.
        
        Args:
            model_name: YOLOv8 model name (yolov8n.pt, yolov8s.pt, etc.)
            confidence_threshold: Minimum confidence for detections
        """
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.model: Optional[YOLO] = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._load_model()
    
    def _load_model(self):
        """Load YOLOv8 model with TensorRT optimization for Jetson."""
        try:
            print(f"Loading YOLOv8 model: {self.model_name}")
            
            # Check for existing TensorRT engine file
            model_base = os.path.splitext(self.model_name)[0]
            engine_path = f"{model_base}.engine"
            
            if self.device == 'cuda':
                print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
                
                # Try to load TensorRT engine if it exists
                if os.path.exists(engine_path):
                    print(f"Loading TensorRT engine from {engine_path}...")
                    try:
                        self.model = YOLO(engine_path)
                        print("TensorRT engine loaded successfully!")
                    except Exception as e:
                        print(f"Failed to load TensorRT engine: {e}")
                        print("Falling back to PyTorch model...")
                        self.model = YOLO(self.model_name)
                else:
                    # Export to TensorRT for Jetson optimization
                    print("TensorRT engine not found. Exporting to TensorRT...")
                    print("This may take a few minutes on first run...")
                    try:
                        self.model = YOLO(self.model_name)
                        # Export to TensorRT engine format
                        self.model.export(
                            format='engine',
                            device=0,
                            half=True,  # Use FP16 for better performance on Jetson
                            simplify=True
                        )
                        print(f"TensorRT engine exported successfully to {engine_path}")
                        # Reload the engine
                        self.model = YOLO(engine_path)
                        print("TensorRT engine loaded and ready!")
                    except Exception as e:
                        print(f"TensorRT export failed: {e}")
                        print("Continuing with standard PyTorch inference...")
                        self.model = YOLO(self.model_name)
            else:
                # CPU mode - no TensorRT
                self.model = YOLO(self.model_name)
                print("Running in CPU mode (no TensorRT)")
            
            print("Model loaded successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to load YOLOv8 model: {e}")
    
    def detect(self, frame: np.ndarray) -> List[Dict]:
        """Run object detection on a frame.
        
        Args:
            frame: Input frame as numpy array (RGB format)
            
        Returns:
            List of detection dictionaries with keys:
            - 'bbox': [x1, y1, x2, y2] bounding box coordinates
            - 'class': class name (string)
            - 'confidence': confidence score (float)
        """
        if self.model is None:
            return []
        
        try:
            # Run inference with Jetson optimizations
            inference_kwargs = {
                'conf': self.confidence_threshold,
                'verbose': False,
                'imgsz': 640,  # Standard YOLO input size for best performance
                'device': self.device
            }
            
            # Use FP16 on GPU for Jetson (TensorRT engines are already FP16)
            if self.device == 'cuda':
                inference_kwargs['half'] = True
            
            results = self.model(frame, **inference_kwargs)
            
            detections = []
            if len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes
                
                for i in range(len(boxes)):
                    # Get box coordinates
                    box = boxes.xyxy[i].cpu().numpy()
                    x1, y1, x2, y2 = box.astype(int)
                    
                    # Get class and confidence
                    cls_id = int(boxes.cls[i].cpu().numpy())
                    confidence = float(boxes.conf[i].cpu().numpy())
                    class_name = self.model.names[cls_id]
                    
                    detections.append({
                        'bbox': [x1, y1, x2, y2],
                        'class': class_name,
                        'confidence': confidence
                    })
            
            return detections
            
        except Exception as e:
            print(f"Detection error: {e}")
            return []
    
    def detect_with_timing(self, frame: np.ndarray) -> tuple:
        """Run detection and return results with timing information.
        
        Args:
            frame: Input frame as numpy array (RGB format)
            
        Returns:
            Tuple of (detections, inference_time) where inference_time is in seconds
        """
        start_time = time.time()
        detections = self.detect(frame)
        inference_time = time.time() - start_time
        return detections, inference_time
    
    def set_confidence_threshold(self, threshold: float):
        """Update confidence threshold.
        
        Args:
            threshold: New confidence threshold (0.0 to 1.0)
        """
        self.confidence_threshold = max(0.0, min(1.0, threshold))
    
    def get_model_info(self) -> Dict:
        """Get information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
        return {
            'model_name': self.model_name,
            'device': self.device,
            'confidence_threshold': self.confidence_threshold,
            'cuda_available': torch.cuda.is_available(),
            'cuda_device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        }

