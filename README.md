# Deep Learning Foundations: VGG19 & Neural Style Transfer

This repository explores the core mechanics of Deep Learning by moving from traditional image classification to the artistic realm of Neural Style Transfer. It features a from-scratch implementation of the VGG19 architecture and its application in visual optimization.

---

## 🦁 Part 1: VGG19 Custom Training from Scratch

Instead of using pre-defined models, this project implements the full **VGG19** architecture according to the original scientific specifications. It is trained on the **Animals-10** dataset (~26,000 images across 10 classes).

### 🏗️ Architecture Design
The network consists of 5 distinct convolutional blocks followed by a fully connected classifier head:
- **Block 1 & 2**: Two Conv layers each, focusing on low-level features (edges, colors).
- **Block 3, 4, & 5**: Four Conv layers each, capturing high-level abstract concepts (shapes, animal parts).
- **Spatial Reduction**: Each block concludes with a `MaxPool2d` layer, shrinking dimensions (e.g., 224x224 → 112x112).
- **Classifier**: Three `nn.Linear` layers with 4096 neurons and Dropout for regularization.

### 🔬 Key Technical Concepts
- **He Initialization**: Used for convolutional layers to prevent vanishing gradients during early training.
- **Label Smoothing**: Implemented in `CrossEntropyLoss` to improve model generalization.
- **Feature Visualization**: A custom hook system to visualize the 64 internal feature maps of the first block, revealing how the AI "sees" initial edges.

---

## 🎨 Part 2: Neural Style Transfer (NST)

Neural Style Transfer flips the standard training logic. Instead of updating model weights, we **freeze the model** and **update the image pixels** directly to minimize a custom visual loss.

### 🧠 The Dual-Loss Mechanism
The project uses the pre-trained VGG19 as a "Visual Judge" to calculate two competing losses:

#### 1. Content Loss
- **Layer**: Extracted from `conv4_2` (deep layer).
- **Logic**: Uses Mean Squared Error (MSE) to ensure the target image maintains the structural "DNA" of the content photo (e.g., Messi or a pet).

#### 2. Style Loss
- **Layers**: Extracted from multiple levels (`conv1_1` through `conv5_1`).
- **The Gram Matrix**: A mathematical trick that captures feature correlations. It records the "vibe" (color palette and brushstroke style) while discarding spatial location.
- **Logic**: Compares the Gram Matrices of the style painting and the generated artwork.

### 🔄 The Optimization Process
1. **Input**: A content image and a style painting (e.g., Van Gogh).
2. **Initialization**: The target image starts as a clone of the content image.
3. **Loop**: For 1000+ iterations, the Adam optimizer shifts pixel values to simultaneously satisfy both the Content Judge and the Style Judge.
4. **Result**: A unique masterpiece that retains the objects of the photo but inherits the soul of the painting.

---

## 🛠️ Requirements
- Python 3.10+
- PyTorch & Torchvision
- Matplotlib, PIL, NumPy
- CUDA-enabled GPU (Highly Recommended)

## 🚀 Usage
1. **Classification**: Run `vgg19_from_scratch.ipynb` to train the animal classifier.
2. **Style Transfer**: Execute the NST script, providing paths for `--content` and `--style` images.
<img width="208" height="243" alt="mohan" src="https://github.com/user-attachments/assets/efb5b6a8-c310-4fc5-965d-f73ad8b9ec78" />
<img width="250" height="202" alt="va" src="https://github.com/user-attachments/assets/6952da21-1c27-48dd-aedc-c4113308aee0" />

<img width="373" height="418" alt="image" src="https://github.com/user-attachments/assets/49942659-5b88-47d8-8b8e-9e6da8e0f4a3" />


