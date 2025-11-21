"""PyQt6 GUI components for the GridX AI Demo application."""

import os
import random
from typing import Optional, List, Dict
import numpy as np
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QRect
from PyQt6.QtGui import QPixmap, QImage, QFont, QPainter, QColor, QPen
from app.utils import get_class_color


class VideoWidget(QWidget):
    """Widget for displaying video feed with object detection overlays."""
    
    def __init__(self, parent=None):
        """Initialize video widget."""
        super().__init__(parent)
        self.frame: Optional[np.ndarray] = None
        self.detections: List[Dict] = []
        self.message: Optional[str] = None
        self.message_timer = 0
        self.setMinimumSize(640, 480)
        self.setStyleSheet("background-color: #0d1117;")
    
    def set_frame(self, frame: np.ndarray):
        """Set the current frame to display.
        
        Args:
            frame: Frame as numpy array (RGB format)
        """
        self.frame = frame
        self.update()
    
    def set_detections(self, detections: List[Dict]):
        """Set detections to overlay on the frame.
        
        Args:
            detections: List of detection dictionaries
        """
        self.detections = detections
        self.update()
    
    def capture_photo(self):
        """Capture current frame with detections and return it."""
        if self.frame is not None:
            # Show random message
            messages = [
                "Cool pic!",
                "Nice!",
                "Awesome!",
                "Great shot!",
                "Perfect!",
                "Excellent!",
                "Amazing!",
                "Fantastic!",
                "Well done!",
                "Incredible!"
            ]
            self.message = random.choice(messages)
            self.message_timer = 120  # Show for ~2 seconds at 60fps
            self.update()
            # Return captured frame and detections
            return (self.frame.copy(), self.detections.copy())
        return None
    
    def paintEvent(self, event):
        """Paint the frame and detection overlays."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        if self.frame is None:
            painter.fillRect(self.rect(), QColor(0x0d, 0x11, 0x17))
            painter.setPen(QColor(0xc9, 0xd1, 0xd9))
            system_font = QFont()
            system_font.setPixelSize(14)
            painter.setFont(system_font)
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Waiting for camera...")
            return
        
        # Draw live frame
        self._draw_frame_with_detections(painter, self.frame, self.detections)
        
        # Draw message if showing
        if self.message and self.message_timer > 0:
            self._draw_message(painter)
            self.message_timer -= 1
            if self.message_timer <= 0:
                self.message = None
    
    def _draw_frame_with_detections(self, painter: QPainter, frame: np.ndarray, detections: List[Dict]):
        """Draw frame with detection overlays."""
        # Convert numpy array to QImage
        height, width, channel = frame.shape
        bytes_per_line = 3 * width
        q_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
        
        # Scale image to fit widget while maintaining aspect ratio
        scaled_pixmap = QPixmap.fromImage(q_image).scaled(
            self.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
        )
        
        # Center the image
        x = (self.width() - scaled_pixmap.width()) // 2
        y = (self.height() - scaled_pixmap.height()) // 2
        painter.drawPixmap(x, y, scaled_pixmap)
        
        # Draw detections if we have scale information
        if detections and scaled_pixmap.width() > 0:
            scale_x = scaled_pixmap.width() / width
            scale_y = scaled_pixmap.height() / height
            
            for detection in detections:
                bbox = detection['bbox']
                class_name = detection['class']
                confidence = detection['confidence']
                
                # Scale bounding box coordinates
                x1 = int(x + bbox[0] * scale_x)
                y1 = int(y + bbox[1] * scale_y)
                x2 = int(x + bbox[2] * scale_x)
                y2 = int(y + bbox[3] * scale_y)
                
                # Use primary accent color for bounding boxes
                q_color = QColor(0x58, 0xa6, 0xff)  # Primary accent blue
                
                # Draw bounding box
                pen = QPen(q_color, 2)
                painter.setPen(pen)
                painter.drawRect(x1, y1, x2 - x1, y2 - y1)
                
                # Draw label background
                label_text = f"{class_name} {confidence:.1%}"
                system_font = QFont()
                system_font.setPixelSize(12)
                system_font.setWeight(QFont.Weight.DemiBold)
                painter.setFont(system_font)
                fm = painter.fontMetrics()
                text_width = fm.horizontalAdvance(label_text)
                text_height = fm.height()
                
                label_rect_x = x1
                label_rect_y = y1 - text_height - 4
                if label_rect_y < 0:
                    label_rect_y = y1
                
                # Draw semi-transparent background for label with border radius effect
                label_bg = QColor(0x0d, 0x11, 0x17, 240)
                painter.fillRect(
                    label_rect_x, label_rect_y, text_width + 12, text_height + 6,
                    label_bg
                )
                
                # Draw border around label
                painter.setPen(QPen(QColor(0x30, 0x36, 0x3d), 1))
                painter.drawRect(label_rect_x, label_rect_y, text_width + 12, text_height + 6)
                
                # Draw label text
                painter.setPen(QColor(0xc9, 0xd1, 0xd9))  # Primary text color
                painter.drawText(
                    label_rect_x + 6, label_rect_y + text_height - 2,
                    label_text
                )
    
    def _draw_message(self, painter: QPainter):
        """Draw congratulatory message overlay."""
        if not self.message:
            return
        
        # Draw semi-transparent background
        painter.fillRect(self.rect(), QColor(0x0d, 0x11, 0x17, 200))
        
        # Draw message text
        message_font = QFont()
        message_font.setPixelSize(48)
        message_font.setWeight(QFont.Weight.Bold)
        painter.setFont(message_font)
        painter.setPen(QColor(0x58, 0xa6, 0xff))  # Blue accent color
        
        fm = painter.fontMetrics()
        text_width = fm.horizontalAdvance(self.message)
        text_height = fm.height()
        
        # Center the message
        x = (self.width() - text_width) // 2
        y = (self.height() - text_height) // 2
        
        # Draw text with shadow effect
        painter.setPen(QColor(0x0d, 0x11, 0x17))
        painter.drawText(x + 2, y + 2, self.message)
        painter.setPen(QColor(0x58, 0xa6, 0xff))
        painter.drawText(x, y, self.message)


class PhotoSlideshowWidget(QWidget):
    """Widget for displaying three photos at a time in a slideshow."""
    
    def __init__(self, parent=None):
        """Initialize slideshow widget."""
        super().__init__(parent)
        self.captured_photos: List[tuple] = []  # List of (frame, detections) tuples
        self.slideshow_index = 0
        self.setMinimumSize(640, 200)
        self.setStyleSheet("background-color: #0d1117;")
    
    def add_photo(self, photo_data: tuple):
        """Add a captured photo to the slideshow.
        
        Args:
            photo_data: Tuple of (frame, detections)
        """
        if photo_data:
            self.captured_photos.append(photo_data)
            self.update()
    
    def next_slide(self):
        """Move to next set of three photos."""
        if len(self.captured_photos) > 3:
            self.slideshow_index = (self.slideshow_index + 3) % len(self.captured_photos)
            self.update()
    
    def get_visible_photos(self) -> List[tuple]:
        """Get the three photos currently visible."""
        if len(self.captured_photos) == 0:
            return []
        
        visible = []
        for i in range(3):
            idx = (self.slideshow_index + i) % len(self.captured_photos)
            visible.append(self.captured_photos[idx])
        return visible
    
    def paintEvent(self, event):
        """Paint the slideshow frames."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        if len(self.captured_photos) == 0:
            # Show placeholder when no photos
            painter.fillRect(self.rect(), QColor(0x0d, 0x11, 0x17))
            painter.setPen(QColor(0x8b, 0x94, 0x9e))
            system_font = QFont()
            system_font.setPixelSize(14)
            painter.setFont(system_font)
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No photos captured yet")
            return
        
        # Draw three photos side by side
        visible_photos = self.get_visible_photos()
        photo_width = self.width() // 3
        photo_height = self.height()
        
        for i, (frame, detections) in enumerate(visible_photos):
            x = i * photo_width
            photo_rect = QRect(x, 0, photo_width, photo_height)
            self._draw_frame_with_detections(painter, frame, detections, photo_rect)
    
    def _draw_frame_with_detections(self, painter: QPainter, frame: np.ndarray, detections: List[Dict], target_rect: QRect = None):
        """Draw frame with detection overlays."""
        # Convert numpy array to QImage
        height, width, channel = frame.shape
        bytes_per_line = 3 * width
        q_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
        
        # Determine target size
        if target_rect:
            target_size = target_rect.size()
            target_x = target_rect.x()
            target_y = target_rect.y()
        else:
            target_size = self.size()
            target_x = 0
            target_y = 0
        
        # Scale image to fit target while maintaining aspect ratio
        scaled_pixmap = QPixmap.fromImage(q_image).scaled(
            target_size, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
        )
        
        # Center the image in target area
        x = target_x + (target_size.width() - scaled_pixmap.width()) // 2
        y = target_y + (target_size.height() - scaled_pixmap.height()) // 2
        painter.drawPixmap(x, y, scaled_pixmap)
        
        # Draw detections if we have scale information
        if detections and scaled_pixmap.width() > 0:
            scale_x = scaled_pixmap.width() / width
            scale_y = scaled_pixmap.height() / height
            
            for detection in detections:
                bbox = detection['bbox']
                class_name = detection['class']
                confidence = detection['confidence']
                
                # Scale bounding box coordinates
                x1 = int(x + bbox[0] * scale_x)
                y1 = int(y + bbox[1] * scale_y)
                x2 = int(x + bbox[2] * scale_x)
                y2 = int(y + bbox[3] * scale_y)
                
                # Use primary accent color for bounding boxes
                q_color = QColor(0x58, 0xa6, 0xff)  # Primary accent blue
                
                # Draw bounding box (thinner for smaller photos)
                pen = QPen(q_color, 1)
                painter.setPen(pen)
                painter.drawRect(x1, y1, x2 - x1, y2 - y1)
                
                # Draw label background (smaller font for compact display)
                label_text = f"{class_name} {confidence:.1%}"
                system_font = QFont()
                system_font.setPixelSize(8)
                system_font.setWeight(QFont.Weight.DemiBold)
                painter.setFont(system_font)
                fm = painter.fontMetrics()
                text_width = fm.horizontalAdvance(label_text)
                text_height = fm.height()
                
                label_rect_x = x1
                label_rect_y = y1 - text_height - 2
                if label_rect_y < 0:
                    label_rect_y = y1
                
                # Draw semi-transparent background for label
                label_bg = QColor(0x0d, 0x11, 0x17, 240)
                painter.fillRect(
                    label_rect_x, label_rect_y, text_width + 6, text_height + 2,
                    label_bg
                )
                
                # Draw border around label
                painter.setPen(QPen(QColor(0x30, 0x36, 0x3d), 1))
                painter.drawRect(label_rect_x, label_rect_y, text_width + 6, text_height + 2)
                
                # Draw label text
                painter.setPen(QColor(0xc9, 0xd1, 0xd9))
                painter.drawText(
                    label_rect_x + 3, label_rect_y + text_height - 1,
                    label_text
                )


class StatisticsPanel(QWidget):
    """Panel displaying detection statistics and performance metrics."""
    
    def __init__(self, parent=None):
        """Initialize statistics panel."""
        super().__init__(parent)
        self.setStyleSheet("""
            QWidget {
                background-color: #161b22;
                color: #c9d1d9;
                border-radius: 8px;
            }
            QLabel {
                background-color: transparent;
                color: #c9d1d9;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Oxygen', 'Ubuntu', 'Cantarell', 'Fira Sans', 'Droid Sans', 'Helvetica Neue', sans-serif;
            }
        """)
        self._init_ui()
    
    def _init_ui(self):
        """Initialize UI components."""
        layout = QVBoxLayout()
        layout.setSpacing(12)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Title
        title = QLabel("Statistics")
        title_font = QFont()
        title_font.setPixelSize(16)
        title_font.setWeight(QFont.Weight.DemiBold)
        title_font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, -0.5)
        title.setFont(title_font)
        title.setStyleSheet("color: #c9d1d9;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        
        # Stats labels
        self.fps_label = QLabel("FPS: 0.0")
        stats_font = QFont()
        stats_font.setPixelSize(14)
        self.fps_label.setFont(stats_font)
        self.fps_label.setStyleSheet("color: #c9d1d9;")
        layout.addWidget(self.fps_label)
        
        self.total_label = QLabel("Total Detections: 0")
        self.total_label.setFont(stats_font)
        self.total_label.setStyleSheet("color: #c9d1d9;")
        layout.addWidget(self.total_label)
        
        self.most_common_label = QLabel("Most Common: None")
        self.most_common_label.setFont(stats_font)
        self.most_common_label.setStyleSheet("color: #c9d1d9;")
        layout.addWidget(self.most_common_label)
        
        self.inference_label = QLabel("Inference: 0.0 ms")
        self.inference_label.setFont(stats_font)
        self.inference_label.setStyleSheet("color: #c9d1d9;")
        layout.addWidget(self.inference_label)
        
        self.gpu_label = QLabel("GPU: N/A")
        self.gpu_label.setFont(stats_font)
        self.gpu_label.setStyleSheet("color: #8b949e;")
        layout.addWidget(self.gpu_label)
        
        # Object counts section
        layout.addWidget(QLabel(""))  # Spacer
        counts_title = QLabel("Object Counts")
        counts_title_font = QFont()
        counts_title_font.setPixelSize(14)
        counts_title_font.setWeight(QFont.Weight.DemiBold)
        counts_title_font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, -0.5)
        counts_title.setFont(counts_title_font)
        counts_title.setStyleSheet("color: #c9d1d9;")
        layout.addWidget(counts_title)
        
        self.counts_label = QLabel("No detections yet")
        self.counts_label.setFont(stats_font)
        self.counts_label.setStyleSheet("color: #8b949e;")
        self.counts_label.setWordWrap(True)
        layout.addWidget(self.counts_label)
        
        layout.addStretch()
        self.setLayout(layout)
    
    def update_fps(self, fps: float):
        """Update FPS display."""
        self.fps_label.setText(f"FPS: {fps:.1f}")
    
    def update_total(self, total: int):
        """Update total detections."""
        self.total_label.setText(f"Total Detections: {total}")
    
    def update_most_common(self, class_name: Optional[str], count: Optional[int]):
        """Update most common object."""
        if class_name and count is not None:
            self.most_common_label.setText(f"Most Common: {class_name} ({count})")
        else:
            self.most_common_label.setText("Most Common: None")
    
    def update_inference_time(self, time_ms: float):
        """Update inference time."""
        self.inference_label.setText(f"Inference: {time_ms:.1f} ms")
    
    def update_gpu_utilization(self, gpu_percent: float, available: bool):
        """Update GPU utilization."""
        if available:
            self.gpu_label.setText(f"GPU: {gpu_percent:.1f}%")
        else:
            self.gpu_label.setText("GPU: N/A")
    
    def update_class_counts(self, counts: Dict[str, int]):
        """Update object class counts."""
        if not counts:
            self.counts_label.setText("No detections yet")
            return
        
        # Sort by count (descending) and show top 10
        sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:10]
        counts_text = "\n".join([f"{name}: {count}" for name, count in sorted_counts])
        self.counts_label.setText(counts_text)


class MainWindow(QMainWindow):
    """Main application window."""
    
    def __init__(self):
        """Initialize main window."""
        super().__init__()
        self.setWindowTitle("GridX AI Demo")
        self._init_ui()
        self._setup_timers()
    
    def _init_ui(self):
        """Initialize UI components."""
        # Set fullscreen
        self.showFullScreen()
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QVBoxLayout()
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        # Title container with logo and text side by side (Header)
        title_container = QWidget()
        title_container.setStyleSheet("""
            QWidget {
                background-color: #161b22;
                border-bottom: 1px solid #30363d;
                padding: 0px;
            }
        """)
        title_container.setFixedHeight(120)
        title_container_layout = QHBoxLayout()
        title_container_layout.setContentsMargins(20, 20, 20, 20)
        title_container_layout.setSpacing(20)
        
        # Add stretch to center content
        title_container_layout.addStretch()
        
        # Load and display logo (centered)
        logo_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "GridX Logo.png")
        if os.path.exists(logo_path):
            logo_label = QLabel()
            logo_pixmap = QPixmap(logo_path)
            # Scale logo to be much bigger (max height ~150px, maintain aspect ratio)
            scaled_pixmap = logo_pixmap.scaledToHeight(150, Qt.TransformationMode.SmoothTransformation)
            logo_label.setPixmap(scaled_pixmap)
            logo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            logo_label.setStyleSheet("border: none; background-color: transparent;")
            title_container_layout.addWidget(logo_label)
        
        # Add stretch to center content
        title_container_layout.addStretch()
        title_container.setLayout(title_container_layout)
        main_layout.addWidget(title_container)
        
        # Content layout (video + slideshow + stats)
        content_layout = QHBoxLayout()
        content_layout.setSpacing(10)
        content_layout.setContentsMargins(10, 10, 10, 10)
        
        # Left side: Live feed and slideshow stacked vertically
        left_panel = QWidget()
        left_panel_layout = QVBoxLayout()
        left_panel_layout.setSpacing(10)
        left_panel_layout.setContentsMargins(0, 0, 0, 0)
        
        # Video widget (live feed)
        self.video_widget = VideoWidget()
        left_panel_layout.addWidget(self.video_widget, 2)
        
        # Photo slideshow widget - three photos at a time
        slideshow_label = QLabel("Captured Photos")
        slideshow_label.setStyleSheet("color: #c9d1d9; font-size: 14px; font-weight: 600;")
        left_panel_layout.addWidget(slideshow_label)
        
        self.slideshow_widget = PhotoSlideshowWidget()
        self.slideshow_widget.setFixedHeight(180)
        left_panel_layout.addWidget(self.slideshow_widget)
        
        left_panel.setLayout(left_panel_layout)
        content_layout.addWidget(left_panel, 3)
        
        # Right side panel (stats + controls) - Sidebar
        right_panel = QWidget()
        right_panel.setStyleSheet("""
            QWidget {
                background-color: #161b22;
                border-left: 1px solid #30363d;
            }
        """)
        right_panel_layout = QVBoxLayout()
        right_panel_layout.setSpacing(10)
        right_panel_layout.setContentsMargins(15, 15, 15, 15)
        
        # Capture button
        self.capture_button = QPushButton("📸 Capture Photo")
        self.capture_button.setStyleSheet("""
            QPushButton {
                background-color: #58a6ff;
                color: #0d1117;
                border: none;
                border-radius: 6px;
                padding: 12px 20px;
                font-size: 14px;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #6eb5ff;
            }
            QPushButton:pressed {
                background-color: #4a95e6;
            }
        """)
        self.capture_button.clicked.connect(self._on_capture_clicked)
        right_panel_layout.addWidget(self.capture_button)
        
        # Statistics panel
        self.stats_panel = StatisticsPanel()
        right_panel_layout.addWidget(self.stats_panel, 1)
        
        right_panel.setLayout(right_panel_layout)
        right_panel.setFixedWidth(300)
        content_layout.addWidget(right_panel, 1)
        
        main_layout.addLayout(content_layout)
        central_widget.setLayout(main_layout)
        
        # Set overall style
        self.setStyleSheet("""
            QMainWindow {
                background-color: #0d1117;
            }
        """)
    
    def _setup_timers(self):
        """Setup update timers."""
        # Timer for UI updates (30 FPS)
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self._update_ui)
        self.update_timer.start(33)  # ~30 FPS
        
        # Title animation timer - disabled since we only have logo now
        # self.title_animation_timer = QTimer()
        # self.title_animation_timer.timeout.connect(self._animate_title)
        # self.title_animation_timer.start(2000)
        # self.title_glow_state = 0
        
        # Slideshow timer - continuously cycle through photos
        self.slideshow_timer = QTimer()
        self.slideshow_timer.timeout.connect(self._slideshow_next)
        self.slideshow_interval = 3000  # 3 seconds per photo
        self.slideshow_timer.start(self.slideshow_interval)
    
    def _update_ui(self):
        """Update UI elements (called by timer)."""
        # This will be connected to the main application's update loop
        pass
    
    def _on_capture_clicked(self):
        """Handle capture button click."""
        photo_data = self.video_widget.capture_photo()
        if photo_data:
            # Add photo to slideshow
            self.slideshow_widget.add_photo(photo_data)
            # Update button text to show count
            count = len(self.slideshow_widget.captured_photos)
            self.capture_button.setText(f"📸 Captured ({count})")
    
    def _slideshow_next(self):
        """Move to next slide in slideshow."""
        if len(self.slideshow_widget.captured_photos) > 0:
            self.slideshow_widget.next_slide()
    
    def _animate_title(self):
        """Animate title with subtle effect."""
        # No animation needed for logo only
        self.title_glow_state = (self.title_glow_state + 1) % 3
        # Title label was removed, so nothing to animate
        pass
    
    def update_frame(self, frame: np.ndarray, detections: List[Dict]):
        """Update video frame and detections.
        
        Args:
            frame: Current frame
            detections: Current detections
        """
        self.video_widget.set_frame(frame)
        self.video_widget.set_detections(detections)
    
    def update_statistics(self, fps: float, total: int, most_common: Optional[tuple],
                         inference_time: float, gpu_util: float, gpu_available: bool,
                         class_counts: Dict[str, int]):
        """Update statistics panel.
        
        Args:
            fps: Current FPS
            total: Total detections
            most_common: Tuple of (class_name, count) or None
            inference_time: Inference time in milliseconds
            gpu_util: GPU utilization percentage
            gpu_available: Whether GPU monitoring is available
            class_counts: Dictionary of class counts
        """
        self.stats_panel.update_fps(fps)
        self.stats_panel.update_total(total)
        if most_common:
            self.stats_panel.update_most_common(most_common[0], most_common[1])
        else:
            self.stats_panel.update_most_common(None, None)
        self.stats_panel.update_inference_time(inference_time)
        self.stats_panel.update_gpu_utilization(gpu_util, gpu_available)
        self.stats_panel.update_class_counts(class_counts)
    
    def keyPressEvent(self, event):
        """Handle keyboard events."""
        if event.key() == Qt.Key.Key_Escape:
            self.close()
        elif event.key() == Qt.Key.Key_F11:
            if self.isFullScreen():
                self.showNormal()
            else:
                self.showFullScreen()
        super().keyPressEvent(event)

