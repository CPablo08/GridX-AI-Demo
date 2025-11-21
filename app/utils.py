"""Utility functions for FPS calculation, statistics tracking, and performance metrics."""

import time
from collections import defaultdict, deque
from typing import Dict, List, Optional
import threading


class FPSCounter:
    """Calculate and track frames per second."""
    
    def __init__(self, window_size: int = 30):
        """Initialize FPS counter.
        
        Args:
            window_size: Number of frames to average over
        """
        self.window_size = window_size
        self.frame_times = deque(maxlen=window_size)
        self.last_time = time.time()
        self.current_fps = 0.0
        self.lock = threading.Lock()
    
    def update(self):
        """Update FPS calculation with current frame."""
        current_time = time.time()
        with self.lock:
            if self.last_time > 0:
                frame_time = current_time - self.last_time
                self.frame_times.append(frame_time)
                if len(self.frame_times) > 0:
                    avg_frame_time = sum(self.frame_times) / len(self.frame_times)
                    self.current_fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0.0
            self.last_time = current_time
    
    def get_fps(self) -> float:
        """Get current FPS value."""
        with self.lock:
            return self.current_fps
    
    def reset(self):
        """Reset FPS counter."""
        with self.lock:
            self.frame_times.clear()
            self.last_time = time.time()
            self.current_fps = 0.0


class StatisticsTracker:
    """Track object detection statistics."""
    
    def __init__(self, history_size: int = 100):
        """Initialize statistics tracker.
        
        Args:
            history_size: Number of recent detections to keep in history
        """
        self.history_size = history_size
        self.class_counts = defaultdict(int)
        self.detection_history = deque(maxlen=history_size)
        self.total_detections = 0
        self.lock = threading.Lock()
    
    def update(self, detections: List[Dict]):
        """Update statistics with new detections.
        
        Args:
            detections: List of detection dictionaries with 'class' and 'confidence' keys
        """
        with self.lock:
            self.total_detections += len(detections)
            for detection in detections:
                class_name = detection.get('class', 'unknown')
                self.class_counts[class_name] += 1
                self.detection_history.append({
                    'class': class_name,
                    'confidence': detection.get('confidence', 0.0),
                    'timestamp': time.time()
                })
    
    def get_class_counts(self) -> Dict[str, int]:
        """Get current count for each object class."""
        with self.lock:
            return dict(self.class_counts)
    
    def get_total_detections(self) -> int:
        """Get total number of detections."""
        with self.lock:
            return self.total_detections
    
    def get_most_common(self) -> Optional[tuple]:
        """Get most common object class and its count.
        
        Returns:
            Tuple of (class_name, count) or None if no detections
        """
        with self.lock:
            if not self.class_counts:
                return None
            most_common = max(self.class_counts.items(), key=lambda x: x[1])
            return most_common
    
    def reset(self):
        """Reset all statistics."""
        with self.lock:
            self.class_counts.clear()
            self.detection_history.clear()
            self.total_detections = 0


class PerformanceMonitor:
    """Monitor inference performance metrics."""
    
    def __init__(self, window_size: int = 30):
        """Initialize performance monitor.
        
        Args:
            window_size: Number of recent measurements to average over
        """
        self.window_size = window_size
        self.inference_times = deque(maxlen=window_size)
        self.lock = threading.Lock()
        self.gpu_utilization = 0.0
        self._gpu_available = False
        self._init_gpu_monitoring()
    
    def _init_gpu_monitoring(self):
        """Initialize GPU monitoring if available."""
        try:
            import pynvml
            pynvml.nvmlInit()
            self._gpu_available = True
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        except Exception:
            self._gpu_available = False
            self._handle = None
    
    def update_inference_time(self, inference_time: float):
        """Update inference time measurement.
        
        Args:
            inference_time: Time taken for inference in seconds
        """
        with self.lock:
            self.inference_times.append(inference_time)
    
    def get_avg_inference_time(self) -> float:
        """Get average inference time in milliseconds."""
        with self.lock:
            if not self.inference_times:
                return 0.0
            avg_time = sum(self.inference_times) / len(self.inference_times)
            return avg_time * 1000  # Convert to milliseconds
    
    def get_gpu_utilization(self) -> float:
        """Get current GPU utilization percentage."""
        if not self._gpu_available or self._handle is None:
            return 0.0
        try:
            import pynvml
            with self.lock:
                util = pynvml.nvmlDeviceGetUtilizationRates(self._handle)
                self.gpu_utilization = util.gpu
                return self.gpu_utilization
        except Exception:
            return 0.0
    
    def is_gpu_available(self) -> bool:
        """Check if GPU monitoring is available."""
        return self._gpu_available


def get_class_color(class_name: str) -> tuple:
    """Get a consistent color for an object class.
    
    Args:
        class_name: Name of the object class
        
    Returns:
        RGB tuple (R, G, B) with values 0-255
    """
    # Generate a consistent color based on class name hash
    hash_value = hash(class_name)
    r = (hash_value & 0xFF0000) >> 16
    g = (hash_value & 0x00FF00) >> 8
    b = hash_value & 0x0000FF
    
    # Ensure colors are bright enough for visibility
    r = max(100, min(255, r))
    g = max(100, min(255, g))
    b = max(100, min(255, b))
    
    return (r, g, b)

