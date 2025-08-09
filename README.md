# Real-Time Face and Hand Detection System

## Overview
This Python application detects faces and hands in real-time using:
- OpenCV's Haar Cascade for face detection
- MediaPipe for hand detection and landmark visualization

## Features
- 👥 Face detection with bounding boxes
- ✋ Hand detection with 21 landmarks per hand
- � Real-time processing (15-30 FPS)
- 🖥️ Clean visualization interface

## Installation
1. Install Python 3.7+
2. Install dependencies:
```bash
pip install opencv-python mediapipe
Usage
Run the detection script:

bash
python detect.py
Controls:

Press 'q' to quit

Blue rectangles = Detected faces

Green rectangles = Detected hands
