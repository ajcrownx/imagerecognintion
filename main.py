import sys
import queue
from PyQt5 import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PIL import Image
import cv2
from textToSpeech import Text2Speech
from transformers import pipeline
import torch

class TTSWorker(QThread):
    finished = pyqtSignal()
    
    def __init__(self):
        super().__init__()
        self.tts = Text2Speech()
        self.text_queue = queue.Queue()
        self.running = True
        
    def setApiKey(self, key):
        self.tts.set_api_key(key)
        
    def add_text(self, text):
        self.text_queue.put(text)
        if not self.isRunning():
            self.start()
            
    def run(self):
        while self.running:
            try:
                text = self.text_queue.get(timeout=0.5)
                self.tts.streamAudio(self.tts.GenerateAudioStream(text))
                self.finished.emit()
            except queue.Empty:
                continue
            
    def stop(self):
        self.running = False
        self.wait()


class InferenceWorker(QThread):
    result_ready = pyqtSignal(str)
    processing = pyqtSignal(bool)
    
    def __init__(self, model_id="bczhou/tiny-llava-v1-hf"):
        super().__init__()
        self.model_id = model_id
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.pipe = None
        self.queue = queue.Queue()
        self.max_new_tokens = 200
        
    def initialize_model(self):
        if self.pipe is None:
            self.pipe = pipeline("image-to-text", model="./tiny-llava-v1-hf", device=self.device)
        
    def process(self, image, prompt):
        self.queue.put((image, prompt))
        if not self.isRunning():
            self.start()
            
    def run(self):
        self.initialize_model()
        while True:
            image, prompt = self.queue.get()
            self.processing.emit(True)
            
            try:
                outputs = self.pipe(image, prompt=prompt, generate_kwargs={"max_new_tokens": self.max_new_tokens})
                response = outputs[0]["generated_text"]
                self.result_ready.emit(response)
            except Exception as e:
                self.result_ready.emit(f"Error: {str(e)}")
            finally:
                self.processing.emit(False)
                self.queue.task_done()


class CameraCapture(QThread):
    frame_ready = pyqtSignal(object)
    
    def __init__(self, camera_id=0):
        super().__init__()
        self.camera_id = camera_id
        self.running = True
        
    def run(self):
        capture = cv2.VideoCapture(self.camera_id)
        while self.running:
            ret, frame = capture.read()
            if ret:
                cv2_im = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_im = Image.fromarray(cv2_im)
                self.frame_ready.emit((frame, pil_im))
            
            # Small sleep to prevent high CPU usage
            self.msleep(30)
            
        capture.release()
            
    def stop(self):
        self.running = False
        self.wait()


class CameraFeedWidget(QLabel):
    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignCenter)
        self.setObjectName("cameraFeed")
        self.setMinimumSize(480, 360)
        
        self.current_frame = None
        self.current_pil_image = None
        
        # Start camera in a separate thread
        self.camera = CameraCapture()
        self.camera.frame_ready.connect(self.update_frame)
        self.camera.start()
        
        # Timer for UI updates
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_display)
        self.timer.start(33)  # ~30 fps
        
    def update_frame(self, data):
        self.current_frame, self.current_pil_image = data
        
    def update_display(self):
        if self.current_frame is not None:
            rgb_frame = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)
            height, width, channel = rgb_frame.shape
            bytes_per_line = rgb_frame.strides[0]
            qt_image = QImage(rgb_frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image)
            self.setPixmap(pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.FastTransformation))
            
    def get_current_image(self):
        return self.current_pil_image
            
    def closeEvent(self, event):
        self.camera.stop()
        event.accept()


class LoadingIndicator(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setObjectName("loadingIndicator")
        self.dots = 0
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_dots)
        self.hide()
        
    def start_animation(self):
        self.show()
        self.timer.start(300)
        
    def stop_animation(self):
        self.timer.stop()
        self.hide()
        
    def update_dots(self):
        self.dots = (self.dots + 1) % 4
        self.setText(f"Processing{'.' * self.dots}")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("AI Vision Assistant")
        self.setGeometry(100, 100, 1000, 700)
        
        # Set application style
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1e1e2e;
            }
            QWidget {
                background-color: #1e1e2e;
                color: #cdd6f4;
                font-size: 12px;
            }
            QLabel#cameraFeed {
                background-color: #181825;
                border: 1px solid #313244;
                border-radius: 8px;
            }
            QLineEdit {
                background-color: #313244;
                border: 1px solid #45475a;
                border-radius: 4px;
                padding: 8px;
                color: #cdd6f4;
                font-size: 13px;
            }
            QTextEdit {
                background-color: #313244;
                border: 1px solid #45475a;
                border-radius: 4px;
                padding: 10px;
                color: #cdd6f4;
                font-size: 13px;
                line-height: 1.5;
            }
            QPushButton {
                background-color: #89b4fa;
                color: #11111b;
                border: none;
                border-radius: 4px;
                padding: 10px 20px;
                font-weight: bold;
                font-size: 13px;
            }
            QPushButton:hover {
                background-color: #b4befe;
            }
            QPushButton:pressed {
                background-color: #74c7ec;
            }
            QPushButton:disabled {
                background-color: #45475a;
                color: #6c7086;
            }
            QSplitter::handle {
                background-color: #313244;
            }
            QLabel#loadingIndicator {
                background-color: #313244;
                color: #f5e0dc;
                border-radius: 4px;
                padding: 8px;
                font-size: 12px;
                font-weight: bold;
            }
        """)
        
        # Create central widget and main layout
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(15)
        
        # Header
        header_layout = QHBoxLayout()
        title_label = QLabel("AI Vision Assistant")
        title_label.setMaximumHeight(50)
        title_label.setFont(QFont("Arial", 18, QFont.Bold))
        header_layout.addWidget(title_label)
        header_layout.addStretch()
        main_layout.addLayout(header_layout)
        
        # Main content splitter (camera and output)
        content_splitter = QSplitter(Qt.Horizontal)
        content_splitter.setChildrenCollapsible(False)
        
        # Left panel (camera)
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        
        camera_label = QLabel("Camera Feed")
        camera_label.setMaximumHeight(50)
        camera_label.setFont(QFont("Arial", 12))
        left_layout.addWidget(camera_label)
        
        self.camera_feed = CameraFeedWidget()
        left_layout.addWidget(self.camera_feed)
        
        self.loading_indicator = LoadingIndicator()
        left_layout.addWidget(self.loading_indicator)
        
        content_splitter.addWidget(left_panel)
        
        # Right panel (text input/output)
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(10, 0, 0, 0)
        
        # Input section
        input_label = QLabel("Your Prompt")
        input_label.setFont(QFont("Arial", 12))
        right_layout.addWidget(input_label)
        
        self.text_input = QLineEdit()
        self.text_input.setPlaceholderText("Ask something about the image...")
        self.text_input.setMinimumHeight(40)
        right_layout.addWidget(self.text_input)
        
        # Output section
        output_label = QLabel("Assistant Response")
        output_label.setFont(QFont("Arial", 12))
        right_layout.addWidget(output_label)
        
        self.text_output = QTextEdit()
        self.text_output.setPlaceholderText("Response will appear here")
        self.text_output.setReadOnly(True)
        right_layout.addWidget(self.text_output)
        
        # Button section
        button_layout = QHBoxLayout()
        
        self.generate_button = QPushButton("Generate Response")
        self.generate_button.setMinimumHeight(45)
        self.generate_button.setIcon(QIcon.fromTheme("system-search"))
        self.generate_button.clicked.connect(self.generate_response)
        button_layout.addWidget(self.generate_button)
        
        right_layout.addLayout(button_layout)
        content_splitter.addWidget(right_panel)
        
        # Set splitter sizes
        content_splitter.setSizes([400, 600])
        main_layout.addWidget(content_splitter)
        
        # Set up workers
        self.inference_worker = InferenceWorker()
        self.inference_worker.result_ready.connect(self.update_output)
        self.inference_worker.processing.connect(self.set_processing_state)
        
        self.tts_worker = TTSWorker()
        
        # Set up keyboard shortcuts
        self.text_input.returnPressed.connect(self.generate_response)
        
    def set_processing_state(self, is_processing):
        if is_processing:
            self.generate_button.setEnabled(False)
            self.loading_indicator.start_animation()
        else:
            self.generate_button.setEnabled(True)
            self.loading_indicator.stop_animation()
    
    def generate_response(self):
        user_input = self.text_input.text().strip()
        if not user_input:
            return
            
        if user_input.lower() == "exit":
            self.close()
            return
            
        image = self.camera_feed.get_current_image()
        if image is None:
            self.text_output.setText("Error: No camera image available.")
            return
            
        prompt = f"USER: <image>\n{user_input}\nASSISTANT:"
        self.inference_worker.process(image, prompt)
        
    def update_output(self, response):
        self.text_output.setText(response)
        self.tts_worker.add_text(response)
        
    def closeEvent(self, event):
        self.camera_feed.camera.stop()
        self.tts_worker.stop()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())