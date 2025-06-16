# Basic Machine Learning Demos

This repository provides a collection of small interactive machine learning demos centred around real‑time audio processing and computer vision.  The code lives in the `basicmachinelearning/` folder and includes utilities for capturing data, training simple models and running real‑time inference.

## Contents

| File | Description |
| --- | --- |
| `training.py` | Interactive tool to collect vowel samples from a microphone, visualise the spectrogram and train a k‑nearest neighbours classifier on the first two LPC formants.  Saves the model (`knn_model.pkl`), the captured formants (`formant_samples.npy`, `formant_labels.npy`) and an optional baseline noise spectrum. |
| `inference.py` | Runs real‑time vowel inference using the saved k‑NN model and optional `noise_spectrum.npy`.  Displays a live spectrogram and the predicted vowel in formant space. |
| `audioprocessing.py` | Stand‑alone demo that can collect vowel samples, estimate formants, perform noise reduction and display the results in real time.  Predictions from the trained classifier are shown along with the spectrogram. |
| `facegesturerecognition` | Script that recognises a reference face and tracks hand gestures using OpenCV, `face_recognition` and MediaPipe.  Gestures can be memorised on‑the‑fly and recognised in subsequent frames. |
| `takepicture.py` | Simple helper to capture an image from the webcam (used to create the reference face image). |
| `*.npy`, `knn_model.pkl` | Example data files produced by `training.py`.  They store captured formant samples, labels and the baseline noise spectrum. |

## Usage

1. **Training vowel classifier**
   ```bash
   python training.py
   ```
   Follow the prompts to record vowels (`a/e/i/o/u`) and optional background noise.  A k‑NN model will be saved to `knn_model.pkl`.

2. **Real‑time inference**
   ```bash
   python inference.py
   ```
   Loads `knn_model.pkl` (and optionally `noise_spectrum.npy`) and displays the live spectrogram with predicted vowels.

3. **Face and gesture recognition**
   Capture a reference face using `takepicture.py` and then run the gesture demo:
   ```bash
   python facegesturerecognition --image caleb.jpg
   ```
   Press `g` to label a detected hand pose and `q` to quit.

## Requirements

The demos rely on a number of third‑party packages including:

- `numpy`, `scipy`, `sounddevice`, `librosa`
- `matplotlib` (configured to use `TkAgg` backend)
- `scikit‑learn`
- `opencv-python`, `mediapipe`, `face_recognition` (and `dlib` for some platforms)

Install them via `pip` before running the scripts.

---

These scripts are intended for experimentation and demonstration purposes.  They provide simple examples of capturing data, training lightweight models and performing inference in real time using Python.
