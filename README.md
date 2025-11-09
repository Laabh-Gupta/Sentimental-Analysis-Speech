# Speech Emotion Recognition (SER) – Deep Learning Upgrade (2025)

This repository contains a fully upgraded **Speech Emotion Recognition (SER)** pipeline using **PyTorch**, supporting multiple neural architectures, advanced augmentation, and full evaluation workflows.

---

## 🚀 Features

### ✅ Multi-Model Architecture
The system implements and compares three deep-learning architectures:

| Model | Input Type | Description |
|-------|------------|-------------|
| **CNN2D** | Log-Mel Spectrogram | Learns time–frequency emotional cues |
| **CRNN** | Log-Mel Spectrogram | Convolution + BiLSTM for temporal modeling |
| **CNN1D** | Raw Waveform | Direct feature extraction from waveform |

---

## ✅ Speaker-Independent Dataset Split

To ensure strong generalization, the dataset is split by **actor identity**:

- **Train:** Actors 1–16  
- **Validation:** Actors 17–20  
- **Test:** Actors 21–24  

Dataset used: **RAVDESS Emotional Speech Audio**.

---

## ✅ Advanced Feature Engineering

### Log-Mel Spectrograms
- 64–80 mel bins  
- Dynamic range compression  
- Mean–variance normalization  

### MFCC Features
- 40-dimensional coefficients  
- Cepstral representations of human speech  

### Raw Waveform
- Normalized 16 KHz audio  
- Processed via a 1D convolutional feature extractor  

### SpecAugment + Waveform Augmentation
- Time masking  
- Frequency masking  
- Gain jitter  
- Gaussian noise injection  
- Pitch shifting  

These augmentations significantly improve robustness for small datasets like RAVDESS.

---

## ✅ Enhanced Training System

- **Mixed Precision Training (torch.amp)**  
- **AdamW Optimizer with Weight Decay**  
- **OneCycleLR scheduler** for smoother convergence  
- **Early stopping** based on validation F1  
- **Class weighting + label smoothing**  
- **Batch-level progress display**  

---

## ✅ Evaluation Suite

The notebook automatically generates:

- Accuracy  
- Weighted F1 Score  
- Confusion Matrix  
- Training Curves (Loss + F1)  
- UMAP Embeddings (optional)  
- **Model Comparison Bar Graph**  

---

## ✅ Model Architectures

### CNN2D
```
Conv → BN → ReLU → MaxPool × 4
AdaptiveAvgPool2d
Fully Connected Classifier
```

### CRNN
```
CNN Feature Extractor
Bi-directional LSTM
Temporal pooling
Fully connected output layer
```

### CNN1D
```
1D Convolutional blocks
AdaptiveAvgPool1d
Fully connected output layer
```

---

## 📊 Example Results (RAVDESS)

| Model | Accuracy | F1 Score |
|-------|----------|---------|
| **CNN2D (Mel)** | ~60–65% | ~0.60 |
| **CRNN (Mel)** | ~65–70% | ~0.67 |
| **CNN1D (Wave)** | ~50–55% | ~0.50 |

Results vary depending on augmentation intensity and audio length.

---

## 🛠 Installation

```bash
pip install torch torchaudio librosa matplotlib scikit-learn umap-learn
```

---

## ▶️ Usage

Run the main notebook:

```
SER_torch_baselines.ipynb
```

Example training:

```python
cnn2d = CNN2D(in_ch=1, num_classes=NUM_CLASSES)
cnn2d, history, results = fit_model(
    cnn2d,
    (train_loader_mel, val_loader_mel, test_loader_mel)
)
```

Example evaluation:

```python
plot_curves(history, "CNN2D Training")
plot_conf_mat(results["y_true"], results["y_pred"], CLASSES)
```

---

## 📁 Project Structure

```
Sentimental-Analysis-Speech/
│
├── archive/                   # Dataset (RAVDESS)
├── SER_torch_baselines.ipynb  # Main Notebook
├── SER_torch_baselines.py     # Script version
├── best_cnn2d_mel.pt          # Saved model (auto-generated)
├── README.md
└── plots/                     # Saved visualizations
```

---

## 🔮 Future Improvements

- Add **Wav2Vec2 / HuBERT** pretrained embeddings  
- Add **PANNs CNN14** architecture  
- ONNX / TorchScript export  
- Build a **Streamlit real-time SER app**  

---

## 👤 Author
**Laabh Gupta**  
GitHub: https://github.com/Laabh-Gupta
