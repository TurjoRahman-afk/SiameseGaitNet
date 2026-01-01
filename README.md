<<<<<<< HEAD
# SiameseGaitNet
=======
# Siamese Spatio-Temporal Attention Transformer for Gait Recognition

A state-of-the-art deep learning model for person identification using gait recognition. This implementation features a sophisticated architecture combining Graph Convolutional Networks (GCN), spatial and temporal attention mechanisms, and transformer encoders for robust gait-based biometric identification.

## 🎯 Overview

Gait recognition aims to identify individuals based on their unique walking patterns captured as skeleton sequences. This implementation provides:


## 🏗️ Architecture Flow

```
Input Gait Sequence (T frames)
        ↓
Frame-level Feature Extractor (GCN)
   • Graph Convolutional Network
   • Models spatial relationships between joints
        ↓
Spatial Attention
   • Multi-head attention across joints
   • Learns discriminative body parts
        ↓
Temporal Attention
   • Multi-head attention across frames
   • Captures gait cycle patterns
        ↓
Transformer Encoder (6 layers)
   • Long-range temporal dependencies
   • Positional encoding
        ↓
Hierarchical Feature Aggregation
   • Multi-scale temporal pooling [1, 2, 4, 8]
   • Statistical aggregation (mean, max, std)
        ↓
Siamese Metric Learning
   • L2-normalized embeddings
   • Contrastive/Triplet loss
        ↓
Final Embedding (256-dim)
```

## 📦 Project Structure

```
gait_recognition/
├── gait_config.py           # Configuration and hyperparameters
├── gait_model.py            # Complete model architecture
│   ├── SpatialAttention         # Joint importance learning
│   ├── TemporalAttention        # Frame importance learning
│   ├── GCNFeatureExtractor      # Skeleton graph processing
│   ├── TransformerEncoder       # Long-range dependencies
│   ├── HierarchicalAggregation  # Multi-scale pooling
│   ├── GaitEncoder              # Complete encoder pipeline
│   ├── SiameseGaitTransformer   # Siamese wrapper
│   ├── ContrastiveLoss          # Pair-based loss
│   └── TripletLoss              # Triplet-based loss
├── gait_data_utils.py       # Data loading and augmentation
│   ├── GaitPreprocessor         # Normalization, alignment
│   ├── GaitAugmentation         # Data augmentation
│   └── GaitDataset              # PyTorch dataset
├── gait_train.py            # Training pipeline
├── gait_evaluate.py         # Evaluation and inference
└── README.md                # This file
```

## 🚀 Installation

### Requirements

```bash
# Core dependencies
torch >= 2.0.0
numpy >= 1.20.0
scikit-learn >= 0.24.0
matplotlib >= 3.3.0
tqdm >= 4.60.0
```

### Setup

```bash
# Clone or download the project
cd gait_recognition

# Install dependencies
pip install torch torchvision torchaudio
pip install numpy scikit-learn matplotlib tqdm

# Verify installation
python gait_model.py
```

## 📖 Usage

### 1. Quick Start - Test Model Architecture

```bash
# Test the model architecture
python gait_model.py
```

Expected output:
```
✓ Model created successfully!
  Parameters: 15,234,816
  Size: 58.13 MB
✓ Testing forward pass...
  Output embedding shape: (4, 256)
```

### 2. Train the Model

```bash
# Train with default configuration
python gait_train.py
```

Training features:

### 3. Evaluate and Run Examples

```bash
# Run evaluation examples
python gait_evaluate.py
```

This demonstrates:

## 🎛️ Configuration

Key hyperparameters in `gait_config.py`:

```python
# Data
NUM_FRAMES = 60              # Frames per gait sequence
NUM_JOINTS = 25              # Skeleton joints (Kinect V2)
JOINT_DIM = 3                # Coordinates (x, y, z)

# Architecture
FEATURE_DIM = 256            # Base feature dimension
SPATIAL_HEADS = 8            # Spatial attention heads
TEMPORAL_HEADS = 8           # Temporal attention heads
TRANSFORMER_LAYERS = 6       # Transformer depth
EMBEDDING_DIM = 256          # Final embedding size

# Training
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
NUM_EPOCHS = 200
LOSS_TYPE = 'triplet'        # 'contrastive', 'triplet', 'combined'
MARGIN = 0.5
```

## 📊 Model Details

### Architecture Components

#### 1. **GCN Feature Extractor**

#### 2. **Spatial Attention**

#### 3. **Temporal Attention**

#### 4. **Transformer Encoder**

#### 5. **Hierarchical Aggregation**

#### 6. **Loss Functions**

**Contrastive Loss**:
$$L = y \cdot d^2 + (1-y) \cdot \max(0, m - d)^2$$

**Triplet Loss**:
$$L = \max(0, d(a, p) - d(a, n) + m)$$

### Model Size

```
Total Parameters: ~15.2M
Model Size: ~58 MB
Inference Speed: ~50ms per sequence (GPU)
```

### Parameter Breakdown

| Component | Parameters |
|-----------|------------|
| GCN Extractor | ~200K |
| Spatial Attention | ~500K |
| Temporal Attention | ~500K |
| Transformer (6 layers) | ~12M |
| Hierarchical Aggregation | ~2M |
| **Total** | **~15.2M** |

## 🎨 Data Augmentation

The model supports extensive augmentation:


## 📈 Training

### Training Process

1. **Data Loading**: Synthetic gait sequences or custom dataset
2. **Preprocessing**: Normalization, alignment, interpolation
3. **Augmentation**: Applied during training
4. **Forward Pass**: Through complete architecture
5. **Loss Computation**: Contrastive/Triplet/Combined
6. **Backpropagation**: With gradient clipping
7. **Validation**: Every epoch
8. **Checkpointing**: Best model + periodic saves

### Training Output

```
Epoch 1/200
  Train Loss: 0.4523
  Val Loss: 0.3892
  Learning Rate: 0.000100
  ✓ Best model saved

Epoch 10/200
  Train Loss: 0.2145
  Val Loss: 0.1987
  Learning Rate: 0.000095
```

### Saved Files

```
checkpoints/gait_model/
├── best_model.pth              # Best validation loss
├── final_model.pth             # Final epoch
└── checkpoint_epoch_X.pth      # Periodic checkpoints

logs/gait/
└── training.log                # Training logs

results/gait/
├── training_history.png        # Loss curves
└── attention_visualization.png # Attention weights
```

## 🔬 Evaluation

### Metrics

1. **Verification (1:1 matching)**
   - Accuracy
   - Equal Error Rate (EER)
   - AUC-ROC

2. **Identification (1:N matching)**
   - Rank-1 accuracy
   - Rank-5 accuracy
   - Rank-10 accuracy

### Evaluation Protocols


## 💻 Code Examples

### Example 1: Basic Similarity

```python
from gait_evaluate import GaitInference

# Load model
inference = GaitInference('checkpoints/gait_model/best_model.pth')

# Compare two gait sequences
similarity, distance = inference.compute_similarity(seq1, seq2)

if distance < 0.5:
    print("Same person")
else:
    print("Different person")
```

### Example 2: Person Identification

```python
# Identify person from gallery
top_matches = inference.identify_person(
    query_sequence=query_seq,
    gallery_sequences=gallery_seqs,
    gallery_labels=gallery_labels,
    top_k=5
)

for rank, (person_id, similarity, distance) in enumerate(top_matches, 1):
    print(f"Rank {rank}: Person {person_id} (similarity: {similarity:.4f})")
```

### Example 3: Attention Visualization

```python
# Visualize what the model focuses on
inference.visualize_attention(
    gait_sequence=sample_seq,
    save_path='attention_maps.png'
)
```

## 🔧 Custom Dataset

To use your own gait dataset:

```python
# 1. Format your data as skeleton sequences
# Shape: (num_frames, num_joints, joint_dim)
your_sequences = load_your_data()

# 2. Create dataset
from gait_data_utils import GaitDataset

dataset = GaitDataset(
    data=your_sequences,
    labels=your_labels,
    config=config,
    mode='triplet',
    augment=True
)

# 3. Create dataloader
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 4. Train
trainer = GaitTrainer(config)
trainer.train()
```

## 📊 Expected Performance

On synthetic data (1000 sequences, 100 persons):

On real gait datasets (e.g., CASIA-B):

## 🎯 Applications


## 🛠️ Troubleshooting

### Out of Memory (OOM)

```python
# Reduce batch size
config.BATCH_SIZE = 16

# Reduce sequence length
config.NUM_FRAMES = 40

# Disable mixed precision
config.USE_AMP = False
```

### Slow Training

```python
# Enable mixed precision
config.USE_AMP = True

# Increase batch size (if memory allows)
config.BATCH_SIZE = 64

# Reduce transformer layers
config.TRANSFORMER_LAYERS = 4
```

### Poor Accuracy

```python
# Increase model capacity
config.TRANSFORMER_LAYERS = 8
config.FEATURE_DIM = 512

# More augmentation
config.TEMPORAL_CROP = True
config.SPEED_PERTURBATION = True

# Longer training
config.NUM_EPOCHS = 300
config.PATIENCE = 30
```

## 📚 Technical Details

### Computational Complexity


### Memory Requirements


## 🔬 Research Background

This architecture is inspired by:

1. **Spatial Attention**: Focus on discriminative body parts
2. **Temporal Attention**: Capture periodic gait patterns
3. **Transformers**: Long-range temporal dependencies
4. **Metric Learning**: Robust similarity embeddings
5. **GCN**: Skeleton structure modeling

Key innovations:

## 🤝 Contributing

To extend this project:

1. **Custom Skeleton Structure**: Modify adjacency matrix in `GCNFeatureExtractor`
2. **New Loss Functions**: Add to `gait_model.py`
3. **Advanced Augmentation**: Extend `GaitAugmentation` class
4. **Multi-modal**: Add RGB features alongside skeleton

## 📄 License

This project is provided for educational and research purposes.

## 🙏 Acknowledgments


## 📞 Support

For issues or questions:
1. Check configuration settings
2. Review training logs
3. Verify data format
4. Test on synthetic data first

## 🚀 Future Enhancements



**Built with ❤️ for robust gait-based person identification**

*Last updated: December 2025*
>>>>>>> 669aff2 (Initial commit: Siamese Spatio-Temporal Attention Transformer for Gait Recognition)
