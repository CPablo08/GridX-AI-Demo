"""Camera capture module for USB webcam."""

import cv2
import threading
import time
from typing import Optional, Callable
import numpy as np


class Camera:
    """USB webcam capture with threaded frame reading."""
    
    def __init__(self, camera_index: int = 0, width: int = 1280, height: int = 720):
        """Initialize camera.
        
        Args:
            camera_index: Camera device index (default 0)
            width: Desired frame width
            height: Desired frame height
        """
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.cap: Optional[cv2.VideoCapture] = None
        self.running = False
        self.frame: Optional[np.ndarray] = None
        self.lock = threading.Lock()
        self.thread: Optional[threading.Thread] = None
        self.frame_callback: Optional[Callable] = None
        self.error_callback: Optional[Callable] = None
        self.last_frame_time = 0
    
    def start(self):
        """Start camera capture thread."""
        if self.running:
            return
        
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera {self.camera_index}")
        
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Get actual resolution
        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.width = actual_width
        self.height = actual_height
        
        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
    
    def stop(self):
        """Stop camera capture."""
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=2.0)
        if self.cap is not None:
            self.cap.release()
            self.cap = None
    
    def _capture_loop(self):
        """Main capture loop running in separate thread."""
        consecutive_errors = 0
        max_errors = 5
        
        while self.running:
            try:
                if self.cap is None or not self.cap.isOpened():
                    if self.error_callback:
                        self.error_callback("Camera not opened")
                    time.sleep(0.1)
                    continue
                
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    consecutive_errors += 1
                    if consecutive_errors >= max_errors:
                        if self.error_callback:
                            self.error_callback("Failed to read frame from camera")
                        time.sleep(0.1)
                    continue
                
                consecutive_errors = 0
                
                # Convert BGR to RGB for display
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                with self.lock:
                    self.frame = frame_rgb
                    self.last_frame_time = time.time()
                
                # Call frame callback if set
                if self.frame_callback:
                    self.frame_callback(frame_rgb)
                    
            except Exception as e:
                if self.error_callback:
                    self.error_callback(f"Camera error: {str(e)}")
                time.sleep(0.1)
    
    def get_frame(self) -> Optional[np.ndarray]:
        """Get the latest frame.
        
        Returns:
            Latest frame as numpy array (RGB format) or None
        """
        with self.lock:
            return self.frame.copy() if self.frame is not None else None
    
    def is_running(self) -> bool:
        """Check if camera is running."""
        return self.running
    
    def set_frame_callback(self, callback: Callable):
        """Set callback function for new frames.
        
        Args:
            callback: Function that takes a frame (numpy array) as argument
        """
        self.frame_callback = callback
    
    def set_error_callback(self, callback: Callable):
        """Set callback function for errors.
        
        Args:
            callback: Function that takes an error message (string) as argument
        """
        self.error_callback = callback

