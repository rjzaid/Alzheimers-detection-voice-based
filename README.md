# Alzheimer's Disease Detection Using Voice Data

This project explores the use of **speech/voice data** to predict cognitive decline associated with Alzheimer's disease, using **MMSE (Mini-Mental State Examination) scores** as the target measure. The pipeline fine-tunes a pretrained audio transformer model, extracts deep audio embeddings, and trains a regression model to predict MMSE scores directly from voice recordings.

##  Overview

Alzheimer's disease and related cognitive impairments are often reflected in subtle changes in a person's speech patterns — pauses, fluency, articulation, and acoustic features. This project investigates whether deep audio representations can be used to estimate a patient's MMSE score (a standard clinical measure of cognitive function) directly from voice recordings, without manual feature engineering.

##  Approach

1. **Audio Feature Extraction** — Uses [`facebook/data2vec-audio-base`](https://huggingface.co/facebook/data2vec-audio-base), a self-supervised speech representation model from Meta AI, as the backbone for extracting deep audio embeddings.
2. **Fine-Tuning** — The Data2Vec Audio model is partially fine-tuned (with selective layer unfreezing) on the voice dataset.
3. **CNN Embedding Layer** — A CNN layer is added on top of the audio embeddings to learn task-specific representations.
4. **Embedding Extraction & Storage** — Embeddings for all audio samples are extracted and saved (`.npy` format) for downstream training.
5. **MMSE Score Matching** — Extracted embeddings are matched against MMSE labels from a clinical CSV dataset.
6. **Regression Model** — A neural network regressor is trained to predict MMSE scores from the audio embeddings, evaluated using MAE and RMSE.

##  Tech Stack

- **Python **
- **PyTorch** — model training and fine-tuning
- **Hugging Face Transformers** — `Data2VecAudioModel`, `AutoFeatureExtractor`
- **Librosa** — audio processing
- **scikit-learn** — evaluation metrics (accuracy, precision, recall, F1, MAE, RMSE)
- **Pandas / NumPy** — data handling
- **Matplotlib** — training/validation visualization

##  Project Structure

##  Dataset

This project uses a private/clinical voice dataset (audio recordings + MMSE labels) sourced via Kaggle. The dataset is **not included** in this repository due to size and privacy considerations.

- `mmse label.csv` — MMSE clinical scores per subject
- Audio embeddings extracted from raw voice recordings

## ⚙️ How to Run

1. Clone this repository:
```bash
   git clone https://github.com/rjzaid/Alzheimers-detection-voice-based.git
   cd Alzheimers-detection-voice-based
```
2. Install dependencies:
```bash
   pip install transformers==4.28.1 torch librosa pandas numpy scikit-learn matplotlib
```
3. Open the notebook:
```bash
   jupyter notebook research-work.ipynb
```
4. Update dataset paths (`mmse label.csv`, audio embeddings folder) to match your local/Kaggle environment.

##  Evaluation Metrics

The regression model is evaluated using:
- **MAE** (Mean Absolute Error)
- **RMSE** (Root Mean Squared Error)

Classification variants are evaluated using:
- **Accuracy, Precision, Recall, F1-score**

##  Future Work

- Expand dataset size for better generalization
- Compare against additional pretrained audio models (Wav2Vec2, HuBERT)
- Incorporate linguistic/text-based features alongside acoustic features
- Deploy as a lightweight screening tool

##  License

This project is for research and educational purposes.



*Built as part of ongoing research into non-invasive, voice-based screening methods for Alzheimer's disease.*
