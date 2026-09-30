---
title: Lunar Crater Detection
emoji: 🌖
colorFrom: purple
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# Lunar Crater Detection System

A high-performance lunar image processing and crater detection system built with FastAPI, PyTorch, and OpenCV.

## Features

- **PDS4 Ingestion**: Native parser for lunar dataset products (LROC NAC images, RAW buffers, XML metadata).
- **Photometric Correction**: Lunar surface illumination normalization and reflectance standardisation.
- **Feature Matching & Registration**: Robust keypoint correspondence using SuperPoint and LoFTR for invariant surface alignment.
- **Crater Detection**: YOLO-based crater recognition and invariance training pipeline.
- **Interactive Web Interface & API**: FastAPI backend with rich visualization for multi-temporal lunar image comparison.

## Project Structure

```
lunar/
├── app/                  # FastAPI web application & pipeline logic
│   ├── config.py         # Pipeline configuration
│   ├── main.py           # FastAPI endpoints & UI server
│   ├── models/           # Neural network models (YOLO, SuperPoint, LoFTR)
│   ├── pipeline/         # Image processing, registration, and crater detection
│   └── static/           # Web dashboard assets
├── demo_data/            # Sample LROC NAC lunar imagery & generator
├── tests/                # Pipeline, training, and registration test suites
├── weights/              # Pretrained model weights
├── lunar_craters.yaml    # Dataset configuration for YOLO
├── requirements.txt      # Python dependencies
├── train_all.py          # Unified training launcher
├── train_crater_invariance.py # Illumination invariance training
└── train_yolo_craters.py # YOLO crater detection training
```

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/mukesh76650/lunar.git
cd lunar

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Web Application

```bash
# Using Python
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# Or via the batch script (Windows)
start_app_gpu.bat
```

Open `http://localhost:8000` in your browser to access the dashboard.

### 3. Training & Validation

```bash
# Run training on GPU
python train_all.py

# Run test suite
pytest tests/
```
