# Crowd Density Estimator

A computer vision project that estimates how many people are present in a CCTV-style image or video frame and generates a crowd density heatmap.

## Overview

This project aims to build a crowd density estimation system using Python, OpenCV, and deep learning.  
The system will take an image or video as input, estimate the number of people in the frame, and produce a density heatmap showing crowded regions.

This project is being developed as a team project.

## Objectives

- Estimate the number of people in an image or video frame
- Generate a density heatmap
- Handle dense crowds and partial occlusion
- Build a simple demo interface for testing
- Explore real-time inference if possible

## Tech Stack

- Python
- OpenCV
- PyTorch
- NumPy
- Flask
- HTML/CSS

## Project Structure

```text
CrowdDensityEstimator/
├── .gitignore
├── README.md
├── requirements.txt
├── main.py
├── data/
├── notebooks/
├── src/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── density_map.py
│   ├── infer.py
│   ├── train.py
│   └── utils.py
├── models/
│   └── checkpoints/
├── outputs/
│   ├── heatmaps/
│   └── predictions/
├── app/
│   ├── app.py
│   ├── templates/
│   │   └── index.html
│   └── static/
│       └── uploads/
└── notebooks/
```

## Current Status

The repository structure has been created and the basic project setup is in progress.

Planned steps:
1. Set up project folders and dependencies
2. Prepare sample crowd images
3. Build preprocessing and density map generation scripts
4. Add a basic inference pipeline
5. Build a simple web interface
6. Test and improve model performance

## Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/crowd-density-estimator.git
cd crowd-density-estimator
```

Create and activate a virtual environment:

### Windows
```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / Mac
```bash
python3 -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

At the current stage, the project is being prepared.

Later, the project will support:
- image input
- video input
- people count estimation
- density heatmap generation
- simple web demo

## Future Work

- Implement crowd density map regression
- Train or integrate a crowd counting model
- Improve handling of occlusion and scale variation
- Add real-time video inference
- Deploy a simple demo interface

## Notes

This is an team project focused on crowd density estimation using computer vision techniques.

## License

This project is for educational purposes (FOR AI-ML CLUB (TAM) - VIT, VELLORE)