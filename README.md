# Siamese Network for MNIST Digit Similarity using PyTorch

## Overview

This project implements a **Siamese Neural Network** using **PyTorch** to determine whether two handwritten digit images belong to the same class. Instead of classifying the digit, the model learns a feature embedding for each image and compares the embeddings using **Contrastive Loss**.

---

## Features

- Custom Siamese Dataset for MNIST
- Shared-weight Convolutional Neural Network (CNN)
- Contrastive Loss for similarity learning
- Euclidean distance-based similarity prediction
- Supports inference on custom handwritten digit images

---

## Technologies Used

- Python
- PyTorch
- Torchvision
- Pillow (PIL)
- MNIST Dataset

---

## Requirements

Install the required packages:

```bash
pip install torch torchvision pillow
```

---

## Dataset

The project uses the **MNIST Handwritten Digit Dataset**.

It is automatically downloaded using:

```python
mnist = datasets.MNIST(
    root="./data",
    train=True,
    download=True
)
```

---

## Data Preprocessing

The following transformations are applied to every image:

```python
transform = transforms.Compose([
    transforms.Resize(100),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
```

- Resize images to **100 × 100**
- Convert images to tensors
- Normalize pixel values to improve training

---

## Siamese Dataset

A custom dataset class creates image pairs during training.

For each image:

- **50% probability** → select another image from the **same class**
- **50% probability** → select an image from a **different class**

The dataset returns:

```python
(image1, image2, label)
```

where:

- `label = 1` → Same digit
- `label = 0` → Different digits

---

## Network Architecture

```
Input Image (1 × 100 × 100)
        │
Conv2D (1 → 16, Kernel=3)
        │
ReLU
        │
MaxPool (2×2)
        │
Conv2D (16 → 32, Kernel=3)
        │
ReLU
        │
MaxPool (2×2)
        │
Flatten
        │
Linear (16928 → 128)
        │
ReLU
        │
Linear (128 → 64)
        │
64-D Feature Embedding
```

Both images pass through the **same CNN**, meaning both branches share identical weights.

---

## Contrastive Loss

The model uses **Contrastive Loss** to learn image similarity.

Formula:

```
Loss = y × D² + (1 − y) × max(0, margin − D)²
```

where:

- `D` = Euclidean distance between embeddings
- `y = 1` → Similar images
- `y = 0` → Different images
- Margin = **1.0**

Objective:

- Bring embeddings of similar images closer.
- Push embeddings of different images farther apart.

---

## Training

Train the model by running:

```bash
python siamese_network.py
```

Current settings:

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam |
| Learning Rate | 0.001 |
| Batch Size | 64 |
| Epochs | 5 |
| Loss Function | Contrastive Loss |

Example output:

```
loss is 0.1423
loss is 0.0985
loss is 0.0728
loss is 0.0562
loss is 0.0417
```

---

## Predicting Image Similarity

After training, compare two handwritten digit images:

```python
predict_similarity(
    model,
    "digit1.png",
    "digit2.png",
    transform
)
```

Example output:

```
distance : 0.2146
prediction : same
```

or

```
distance : 1.7328
prediction : different
```

---

## How Prediction Works

1. Load two grayscale images.
2. Resize and normalize them.
3. Pass both images through the Siamese network.
4. Compute the Euclidean distance between the embeddings.
5. Compare the distance with a threshold (`0.5`).

Decision rule:

```
Distance < Threshold
        ↓
    Same Digit

Distance ≥ Threshold
        ↓
 Different Digit
```

---

## Save the Trained Model

```python
torch.save(model.state_dict(), "siamese_mnist.pth")
```

---

## Load the Saved Model

```python
model = SiameseCNN()
model.load_state_dict(torch.load("siamese_mnist.pth"))
model.eval()
```

---

## Project Structure

```
.
├── data/
│   └── MNIST
├── siamese_network.py
├── siamese_mnist.pth
└── README.md
```

---

## Possible Improvements

- Train for more epochs (10–20)
- Add GPU (CUDA) support
- Use Batch Normalization
- Add Dropout to reduce overfitting
- Use Hard Negative Mining
- Tune the similarity threshold
- Evaluate using accuracy, precision, recall, and ROC curve
- Visualize learned embeddings using t-SNE

---

## Learning Outcomes

This project demonstrates:

- Custom PyTorch Dataset creation
- Siamese Neural Networks
- Shared-weight architectures
- Contrastive Learning
- Contrastive Loss
- Euclidean Distance
- Feature Embedding Learning
- Image Similarity Detection
- Deep Learning with PyTorch

---

## Author

**Shahid Farhan KP**

B.Tech Computer Science and Engineering

Cochin University of Science and Technology (CUSAT)
