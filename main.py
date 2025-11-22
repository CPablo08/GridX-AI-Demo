"""Main application entry point for GridX AI Demo."""

import sys
import os
import time
import threading
import subprocess
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QThread, pyqtSignal, QTimer
import numpy as np

from app.gui import MainWindow
from app.camera import Camera
from app.detector import Detector
from app.utils import FPSCounter, StatisticsTracker, PerformanceMonitor


def optimize_for_jetson():
    """Optimize system settings for Jetson Orin Nano."""
    print("Optimizing for Jetson Orin Nano...")
    
    try:
        # Set power mode to MAXN (maximum performance)
        # This requires sudo, so we'll try but won't fail if it doesn't work
        try:
            result = subprocess.run(
                ['sudo', 'nvpmodel', '-m', '0'],
                capture_output=True,
                timeout=5
            )
            if result.returncode == 0:
                print("✓ Power mode set to MAXN (maximum performance)")
        except (subprocess.TimeoutExpired, FileNotFoundError, PermissionError):
            print("Note: Could not set power mode (may require sudo or nvpmodel not available)")
        
        # Set GPU clock to maximum
        try:
            result = subprocess.run(
                ['sudo', 'jetson_clocks'],
                capture_output=True,
                timeout=10
            )
            if result.returncode == 0:
                print("✓ Jetson clocks set to maximum")
        except (subprocess.TimeoutExpired, FileNotFoundError, PermissionError):
            print("Note: Could not set jetson_clocks (may require sudo)")
        
        # Set CUDA device to use Tensor Cores if available
        os.environ['CUDA_LAUNCH_BLOCKING'] = '0'  # Non-blocking for better performance
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
        
        print("Jetson optimization complete!")
        
    except Exception as e:
        print(f"Warning: Some optimizations may not have been applied: {e}")
        print("Continuing anyway...")


class DetectionThread(QThread):
    """Thread for running object detection on frames."""
    
    frame_processed = pyqtSignal(np.ndarray, list, float)  # frame, detections, inference_time
    error_occurred = pyqtSignal(str)
    
    def __init__(self, detector: Detector):
        """Initialize detection thread.
        
        Args:
            detector: Detector instance
        """
        super().__init__()
        self.detector = detector
        self.current_frame: np.ndarray = None
        self.frame_lock = threading.Lock()
        self.running = False
    
    def set_frame(self, frame: np.ndarray):
        """Set frame for processing.
        
        Args:
            frame: Frame to process
        """
        with self.frame_lock:
            self.current_frame = frame.copy() if frame is not None else None
    
    def run(self):
        """Main detection loop."""
        self.running = True
        while self.running:
            frame = None
            with self.frame_lock:
                if self.current_frame is not None:
                    frame = self.current_frame.copy()
                    self.current_frame = None
            
            if frame is not None:
                try:
                    detections, inference_time = self.detector.detect_with_timing(frame)
                    self.frame_processed.emit(frame, detections, inference_time)
                except Exception as e:
                    self.error_occurred.emit(f"Detection error: {str(e)}")
            else:
                time.sleep(0.01)  # Small sleep if no frame to process
    
    def stop(self):
        """Stop detection thread."""
        self.running = False
        self.wait()


class GridXAIDemoApp:
    """Main application class."""
    
    def __init__(self):
        """Initialize application."""
        # Optimize for Jetson before initializing
        optimize_for_jetson()
        
        self.app = QApplication(sys.argv)
        self.window = MainWindow()
        
        # Initialize components with Jetson-optimized settings
        # Use 720p for better performance on Jetson
        self.camera = Camera(camera_index=0, width=1280, height=720)
        self.detector = Detector(model_name='yolov8n.pt', confidence_threshold=0.25)
        
        # Statistics and performance tracking
        self.fps_counter = FPSCounter()
        self.stats_tracker = StatisticsTracker()
        self.perf_monitor = PerformanceMonitor()
        
        # Detection thread
        self.detection_thread = DetectionThread(self.detector)
        self.detection_thread.frame_processed.connect(self._on_detection_complete)
        self.detection_thread.error_occurred.connect(self._on_error)
        
        # Current state
        self.current_frame: np.ndarray = None
        self.current_detections: list = []
        self.last_inference_time = 0.0
        
        # Setup camera callbacks
        self.camera.set_frame_callback(self._on_camera_frame)
        self.camera.set_error_callback(self._on_error)
        
        # Setup update timer
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self._update_display)
        self.update_timer.start(33)  # ~30 FPS for display updates
    
    def _on_camera_frame(self, frame: np.ndarray):
        """Handle new frame from camera.
        
        Args:
            frame: New frame from camera
        """
        self.current_frame = frame
        self.fps_counter.update()
        
        # Send frame to detection thread
        if self.detection_thread.isRunning():
            self.detection_thread.set_frame(frame)
    
    def _on_detection_complete(self, frame: np.ndarray, detections: list, inference_time: float):
        """Handle completed detection.
        
        Args:
            frame: Processed frame
            detections: Detection results
            inference_time: Time taken for inference
        """
        self.current_frame = frame
        self.current_detections = detections
        self.last_inference_time = inference_time
        
        # Update statistics
        self.stats_tracker.update(detections)
        self.perf_monitor.update_inference_time(inference_time)
    
    def _on_error(self, error_message: str):
        """Handle error.
        
        Args:
            error_message: Error message
        """
        print(f"Error: {error_message}")
        # Could show error dialog here if needed
    
    def _update_display(self):
        """Update display with current frame and statistics."""
        if self.current_frame is not None:
            self.window.update_frame(self.current_frame, self.current_detections)
        
        # Update statistics
        fps = self.fps_counter.get_fps()
        total = self.stats_tracker.get_total_detections()
        most_common = self.stats_tracker.get_most_common()
        inference_time_ms = self.perf_monitor.get_avg_inference_time()
        gpu_util = self.perf_monitor.get_gpu_utilization()
        gpu_available = self.perf_monitor.is_gpu_available()
        class_counts = self.stats_tracker.get_class_counts()
        
        self.window.update_statistics(
            fps, total, most_common, inference_time_ms,
            gpu_util, gpu_available, class_counts
        )
    
    def run(self):
        """Start and run the application."""
        try:
            # Start camera
            print("Starting camera...")
            self.camera.start()
            
            # Start detection thread
            print("Starting detection thread...")
            self.detection_thread.start()
            
            # Show window
            self.window.show()
            
            # Run application
            exit_code = self.app.exec()
            
            return exit_code
            
        except Exception as e:
            print(f"Application error: {e}")
            return 1
        finally:
            # Cleanup
            print("Shutting down...")
            self.detection_thread.stop()
            self.camera.stop()
    
    def cleanup(self):
        """Cleanup resources."""
        self.detection_thread.stop()
        self.camera.stop()


def main():
    """Main entry point."""
    app = GridXAIDemoApp()
    try:
        exit_code = app.run()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        exit_code = 0
    finally:
        app.cleanup()
    
    sys.exit(exit_code)


if __name__ == "__main__":
    main()

