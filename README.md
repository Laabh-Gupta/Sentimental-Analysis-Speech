# Speech Sentiment and Emotion Analysis

This project develops a deep learning model to classify emotions from speech audio files. Using a Convolutional Neural Network (CNN) combined with a Long Short-Term Memory (LSTM) network, the model is trained on the RAVDESS dataset to recognize emotions like happy, sad, and angry from acoustic features.

## 📖 Table of Contents
* [About The Project](#about-the-project)
* [Dataset](#dataset)
* [Feature Extraction](#feature-extraction)
* [Model & Results](#model--results)
* [Technologies Used](#technologies-used)
* [Installation](#installation)
* [Usage](#usage)
* [License](#license)
* [Contact](#contact)

---

## ℹ️ About The Project

The primary goal of this project is to build a robust classifier for emotion recognition from audio. This has applications in customer service analysis, mental health monitoring, and interactive AI systems. The project involves preprocessing audio, extracting key acoustic features, and training a hybrid CNN-LSTM model for classification.

The core analysis is in `Speech_Sentimental_Analysis.ipynb`.

---

## 📊 Dataset

This project uses the **RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)**, a high-quality, validated dataset designed specifically for emotional research.

* **Concept**: The dataset provides a standardized set of voice recordings where actors express specific emotions. By training a model on the acoustic features of these recordings, it can learn to recognize the underlying emotion in other speech samples.
* **Content**: It features 24 professional actors (12 male, 12 female) vocalizing two lexically-matched statements. The recordings cover 8 distinct emotions: neutral, calm, happy, sad, angry, fearful, disgust, and surprise. Each emotion is produced at both normal and strong intensity levels.
* **Link**: [https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio](https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio)

---

## 🎵 Feature Extraction

To enable the model to learn from the audio signals, the following key acoustic features were extracted from each file using the Librosa library:

* **MFCC (Mel-Frequency Cepstral Coefficients)**: Captures the essential shape of the vocal tract, which is crucial for identifying phonetic characteristics.
* **Chroma Features**: Represents the 12 distinct pitch classes, which is useful for analyzing the harmonic and melodic content of speech.
* **Mel Spectrogram**: A visual representation of the spectrum of frequencies as they vary with time, converted to the mel scale to mimic human hearing.

---

## 🤖 Model & Results

A hybrid deep learning model combining a **Convolutional Neural Network (CNN)** and a **Long Short-Term Memory (LSTM)** network was designed to classify the extracted features.

* **Architecture**:
    1.  A **1D CNN** layer (`Conv1D`) acts as a feature detector, identifying local patterns in the audio features.
    2.  An **LSTM** layer processes the sequence of features from the CNN, capturing temporal dependencies in the speech signal.
    3.  A **Dense** output layer with a `softmax` activation function classifies the input into one of the 8 emotional categories.
* **Performance**: The model was trained for 50 epochs and compiled using the Adam optimizer and categorical cross-entropy loss. It achieved the following performance on the test set:

| Metric         | Score                |
| :------------- | :------------------- |
| **Test Accuracy** | **Approximately 76%**|

---

## 🛠️ Technologies Used

* **Python**
* **TensorFlow & Keras**: For building and training the CNN-LSTM deep learning model.
* **Librosa**: For audio processing and feature extraction.
* **Scikit-learn**: For data splitting (`train_test_split`) and encoding labels.
* **Pandas & NumPy**: For data manipulation and numerical operations.
* **Matplotlib & Seaborn**: For creating visualizations of the data and results.

---

## ⚙️ Installation

To get a local copy up and running, follow these steps.

1.  **Clone the repository:**
    ```sh
    git clone [https://github.com/Laabh-Gupta/Sentimental-Analysis-Speech.git](https://github.com/Laabh-Gupta/Sentimental-Analysis-Speech.git)
    ```
2.  **Navigate to the project directory:**
    ```sh
    cd Sentimental-Analysis-Speech
    ```
3.  **Install the required libraries:**
    ```sh
    pip install pandas numpy tensorflow scikit-learn librosa matplotlib seaborn jupyterlab
    ```

---

## 🚀 Usage

To explore the project, you can run the main Jupyter Notebook.

1.  Start Jupyter Lab:
    ```sh
    jupyter lab
    ```
2.  Open `Speech_Sentimental_Analysis.ipynb` to view the code, analysis, and results.

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

---

## 📫 Contact

Laabh Gupta - reachlaabhgupta@gmail.com - [https://www.linkedin.com/in/laabhgupta/](https://www.linkedin.com/in/laabhgupta/)

Project Link: [https://github.com/Laabh-Gupta/Sentimental-Analysis-Speech](https://github.com/Laabh-Gupta/Sentimental-Analysis-Speech)