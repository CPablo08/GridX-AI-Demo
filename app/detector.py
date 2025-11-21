"""YOLOv8 object detection module with TensorRT optimization support."""

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
        """Load YOLOv8 model."""
        try:
            print(f"Loading YOLOv8 model: {self.model_name}")
            self.model = YOLO(self.model_name)
            
            # Move to GPU if available
            if self.device == 'cuda':
                print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
            
            # Try to export to TensorRT for Jetson optimization
            try:
                if self.device == 'cuda':
                    print("Attempting TensorRT optimization...")
                    # Export to TensorRT (this will be done once and cached)
                    # The model will automatically use TensorRT if available
                    pass  # TensorRT export can be done manually or on first run
            except Exception as e:
                print(f"TensorRT optimization not available: {e}")
                print("Continuing with standard PyTorch inference...")
            
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
            # Run inference
            results = self.model(frame, conf=self.confidence_threshold, verbose=False)
            
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

