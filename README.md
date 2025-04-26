# AI Vision Assistant

## Overview

AI Vision Assistant is a modern desktop application that combines real-time camera feed with advanced AI vision capabilities. This application allows users to ask questions about what their camera sees and receive intelligent responses in both text and speech formats.

## Features

- **Real-time Camera Analysis**: Process live camera feed using computer vision
- **Natural Language Interaction**: Ask questions about what the camera sees in plain language
- **Text-to-Speech Responses**: Hear responses through integrated text-to-speech
- **Modern UI**: Clean, responsive interface with dark mode design
- **Non-blocking Architecture**: Multi-threaded design prevents UI freezing during processing

## Requirements

- Python 3.8+
- PyQt5
- PyTorch
- Transformers
- OpenCV
- PIL (Python Imaging Library)
- CUDA-compatible GPU (recommended for better performance)

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/ai-vision-assistant.git
cd ai-vision-assistant
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the tiny-llava model:
```bash
git clone https://huggingface.co/bczhou/tiny-llava-v1-hf
```

4. Run the application:
```bash
python vision_assistant.py
```

## Usage

1. Launch the application
2. Ensure your camera can see the subject you want to ask about
3. Type your question in the prompt field (e.g., "What objects do you see?")
4. Click "Generate Response" or press Enter
5. View the text response and listen to the spoken answer

## Architecture

The application uses a multi-threaded architecture to ensure responsive UI:
- Camera capture runs in a dedicated thread
- AI inference processing occurs in a separate worker thread
- Text-to-speech generation happens in the background
- PyQt5 signals and slots handle communication between threads

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Tiny-LLaVA Model](https://huggingface.co/bczhou/tiny-llava-v1-hf) for image understanding capabilities
- [PyQt5](https://www.riverbankcomputing.com/software/pyqt/) for the UI framework
- [Transformers](https://huggingface.co/docs/transformers/index) by Hugging Face