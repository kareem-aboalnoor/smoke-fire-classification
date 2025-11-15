# Smoke & Fire Classification using Convolutional Neural Network (CNN)

## Description
A Convolutional Neural Network model that classifies video frames into Normal, Smoke, and Fire stages for early fire detection.

This project aims to help detect fire incidents at early stages by monitoring video streams and classifying visible smoke before flames appear.

---

## Features
- Classifies each video frame as Normal, Smoke, or Fire
- Works on pre-recorded videos or webcam streams
- Real-time frame processing using OpenCV
- High accuracy after training on labeled fire/smoke datasets

---

## Dataset

The dataset used in this project was obtained from Roboflow Universe:

[Classification of Fire and Smoke Dataset (Roboflow)](https://universe.roboflow.com/classificationfire/classification-of-fire-and-smoke/dataset/1)

It contains labeled images classified into three categories:
- Normal / No Fire
- Smoke
- Fire

This dataset was used for training and testing the Convolutional Neural Network (CNN) model to distinguish between different fire stages from video frames.

Dataset credits © Roboflow Universe – "Classification of Fire and Smoke" Project.

---

## Dataset Import (Optional)
If you want to load this dataset directly in your Python project, you can use the Roboflow API:

```python
from roboflow import Roboflow

rf = Roboflow(api_key="YOUR_API_KEY")
project = rf.workspace("classificationfire").project("classification-of-fire-and-smoke")
dataset = project.version(1).download("folder")
