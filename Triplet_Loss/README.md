# 🧠 Triplet Loss Neural Network for Image Embeddings

*A deep learning system for learning discriminative image features using triplet networks*

## 🌟 Key Features
- **Triplet Loss Implementation**: Custom loss function for learning relative similarity
- **Transfer Learning**: Utilizes EfficientNetV2 as base model
- **Image Augmentation**: Built-in brightness/contrast adjustments
- **Evaluation Metrics**: Triplet accuracy and confusion matrix analysis
- **GPU Optimization**: Memory-efficient training with TensorFlow

## 🛠 Technical Architecture
```python
# Model Structure
base_model = EfficientNetV2L(include_top=False, pooling="avg")  # Feature extractor
input_anchor = Input(shape=(IMG_SIZE, IMG_SIZE, 3))  # Triplet inputs
input_positive = Input(shape=(IMG_SIZE, IMG_SIZE, 3))
input_negative = Input(shape=(IMG_SIZE, IMG_SIZE, 3))




Metric      Training	Validation
Loss	      0.2145	  0.2310
Accuracy	  89.34%	  87.21%
