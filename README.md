# Sign Language Translation using Deep Learning

This project implements a deep learning–based sign language recognition system designed to classify hand gesture images into corresponding sign labels using computer vision and transfer learning techniques.

The system is built using a pretrained convolutional neural network backbone combined with a custom gated residual adaptation module to enhance feature learning while maintaining training stability and efficiency.

---

## Problem Statement

Sign language serves as the primary communication medium for many hearing- and speech-impaired individuals. However, effective interaction with non-signers remains a challenge due to the absence of automated translation systems.

This project aims to bridge this communication gap by developing an intelligent gesture recognition model capable of translating sign language gestures into textual representations.

---

## Dataset

The model is trained using a publicly available **Kaggle sign language dataset** containing labeled hand gesture images across multiple sign vocabulary classes.

Due to dataset size limitations, raw datasets are not included in this repository.

Dataset preparation involves:

- Image resizing and normalization  
- Region-of-Interest (ROI) extraction for isolating hand regions  
- Synthetic gesture generation to improve data diversity and class balance  

All preprocessing steps are implemented using custom Python scripts included in the project.

---

## Model Architecture

The proposed model, **SignWordNet**, is designed using a transfer learning approach with architectural enhancements for improved representation learning.

### Architecture Overview

**Backbone Network**  
A pretrained **ResNet-18** model initialized with ImageNet weights is used for deep feature extraction. The final fully connected layer of the backbone is removed to obtain high-level feature embeddings.

**Gated Residual Adapter**  
A custom adapter module is introduced after the backbone. This module applies channel-wise gating combined with residual connections to selectively enhance informative features while preserving original representations.

**Classification Head**  
The refined feature vectors are passed through fully connected layers with ReLU activation and dropout regularization to predict the final sign class.

This architecture allows efficient fine-tuning while reducing the risk of overfitting on limited gesture datasets.

---

## Key Features

- Transfer learning using pretrained ResNet-18  
- Custom gated residual feature adaptation  
- ROI-based gesture preprocessing  
- Synthetic data generation for dataset augmentation  
- Real-time gesture inference support  
- Model explainability using Grad-CAM visualization  

---

## Technologies Used

- Python  
- PyTorch  
- TorchVision  
- OpenCV  
- NumPy  
- Deep Learning (CNN)  
- Transfer Learning  

---

## Project Structure

```text
src/
  signwordnet_model.py
  train_signwordnet_synthetic.py
  evaluate_signwordnet.py
  infer_single_image.py
  gradcam_signwordnet.py

realtime/
  realtime_signwordnet_pretty.py
  app.py

---

## Training

To train the model, run:
  python src/train_signwordnet_synthetic.py

Training includes:

- Cross-entropy loss optimization  
- Adam optimizer  
- Validation-based checkpoint selection  
- Regularization using dropout  

---

## Inference

Single image prediction:
  python src/infer_single_image.py


---

## Model Explainability

Grad-CAM is used to visualize important image regions influencing model predictions. This ensures that the network focuses on relevant hand gesture regions rather than background noise.

---

## Future Enhancements

- Sequence-based sign translation using LSTM or Transformer models  
- Video-based continuous sign recognition  
- Speech synthesis integration  
- Deployment as a web or mobile application  

---


Real-time webcam-based inference:
  python realtime/realtime_signwordnet_pretty.py
