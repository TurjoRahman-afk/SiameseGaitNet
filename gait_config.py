"""
Configuration for Siamese Spatio-Temporal Attention Transformer (Gait Recognition)
"""
import torch


class GaitConfig:
    """
    Hyperparameters for Gait Recognition Model
    """
    
    # ==================== Data Configuration ====================
    # Gait sequence parameters
    NUM_FRAMES = 60              # Number of frames in gait sequence (T)
    NUM_JOINTS = 25              # Number of skeleton joints (e.g., 25 for Kinect V2)
    JOINT_DIM = 3                # Coordinates per joint (x, y, z) or (x, y, confidence)
    INPUT_CHANNELS = 3           # Input feature channels
    
    # Data splits
    TRAIN_SPLIT = 0.7
    VAL_SPLIT = 0.15
    TEST_SPLIT = 0.15
    
    # ==================== Model Architecture ====================
    # Feature extraction
    FEATURE_DIM = 256            # Base feature dimension
    CNN_CHANNELS = [64, 128, 256]  # CNN feature extractor channels
    GCN_HIDDEN_DIM = 128         # GCN hidden dimension
    USE_GCN = True               # Use GCN (True) or CNN (False) for feature extraction
    
    # Spatial attention
    SPATIAL_HEADS = 8            # Multi-head attention heads for spatial
    SPATIAL_DIM = 256            # Spatial attention feature dimension
    SPATIAL_DROPOUT = 0.1
    
    # Temporal attention
    TEMPORAL_HEADS = 8           # Multi-head attention heads for temporal
    TEMPORAL_DIM = 256           # Temporal attention feature dimension
    TEMPORAL_DROPOUT = 0.1
    
    # Transformer encoder
    TRANSFORMER_LAYERS = 6       # Number of transformer encoder layers
    TRANSFORMER_HEADS = 8        # Multi-head attention heads
    TRANSFORMER_DIM = 512        # Transformer hidden dimension
    TRANSFORMER_FF_DIM = 2048    # Feed-forward dimension
    TRANSFORMER_DROPOUT = 0.1
    
    # Hierarchical aggregation
    AGGREGATION_LEVELS = [1, 2, 4, 8]  # Multi-scale temporal pooling
    EMBEDDING_DIM = 256          # Final embedding dimension
    
    # ==================== Training Configuration ====================
    # Optimization
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5
    NUM_EPOCHS = 200
    
    # Learning rate scheduling
    LR_SCHEDULER = 'cosine'      # Options: 'step', 'cosine', 'plateau'
    LR_STEP_SIZE = 30
    LR_GAMMA = 0.1
    LR_WARMUP_EPOCHS = 5
    
    # Loss function
    LOSS_TYPE = 'triplet'        # Options: 'contrastive', 'triplet', 'combined'
    MARGIN = 0.5                 # Margin for contrastive/triplet loss
    TRIPLET_MINING = 'hard'      # Options: 'random', 'hard', 'semi-hard'
    CONTRASTIVE_WEIGHT = 0.5     # Weight for contrastive loss (if combined)
    TRIPLET_WEIGHT = 0.5         # Weight for triplet loss (if combined)
    
    # Regularization
    DROPOUT = 0.3
    WEIGHT_DROPOUT = 0.1         # Dropout for attention weights
    LABEL_SMOOTHING = 0.1
    
    # ==================== Training Optimization ====================
    # Early stopping
    PATIENCE = 20                # Early stopping patience
    MIN_DELTA = 1e-4            # Minimum improvement for early stopping
    
    # Gradient clipping
    GRAD_CLIP = 1.0
    
    # Mixed precision training
    USE_AMP = True               # Automatic Mixed Precision
    
    # ==================== Data Augmentation ====================
    # Gait sequence augmentation
    TEMPORAL_CROP = True         # Random temporal cropping
    TEMPORAL_CROP_RANGE = (40, 60)  # Min/max frames for cropping
    
    SPATIAL_JITTER = True        # Random spatial noise
    JITTER_STD = 0.01           # Standard deviation for spatial jitter
    
    TEMPORAL_SHIFT = True        # Random temporal shift
    SHIFT_RANGE = 5              # Max frames to shift
    
    JOINT_DROPOUT = True         # Random joint occlusion
    JOINT_DROPOUT_PROB = 0.1     # Probability to drop each joint
    
    FLIP_HORIZONTAL = True       # Horizontal flip (left-right)
    FLIP_PROB = 0.5
    
    SPEED_PERTURBATION = True    # Change gait speed
    SPEED_RANGE = (0.8, 1.2)    # Speed multiplier range
    
    # ==================== Evaluation Configuration ====================
    # Similarity metrics
    DISTANCE_METRIC = 'euclidean'  # Options: 'euclidean', 'cosine'
    SIMILARITY_THRESHOLD = 0.5      # Threshold for same/different person
    
    # Evaluation protocols
    EVAL_PROTOCOLS = ['same_view', 'cross_view', 'cross_condition']
    TOP_K = [1, 5, 10]           # Top-K accuracy metrics
    
    # ==================== Hardware Configuration ====================
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    NUM_WORKERS = 4              # DataLoader workers
    PIN_MEMORY = True
    
    # ==================== Paths ====================
    DATA_DIR = './data/gait_sequences'
    MODEL_SAVE_PATH = './checkpoints/gait_model'
    LOG_DIR = './logs/gait'
    RESULTS_DIR = './results/gait'
    
    # ==================== Logging ====================
    LOG_INTERVAL = 10            # Print training stats every N batches
    SAVE_INTERVAL = 5            # Save checkpoint every N epochs
    TENSORBOARD = True           # Use tensorboard logging
    
    # ==================== Model Variants ====================
    # Architecture variants
    USE_POSITIONAL_ENCODING = True   # Add positional encoding to transformer
    USE_RESIDUAL_CONNECTIONS = True  # Residual connections in aggregation
    USE_BATCH_NORM = True            # Batch normalization layers
    USE_LAYER_NORM = True            # Layer normalization in transformer
    
    # Attention variants
    ATTENTION_TYPE = 'scaled_dot_product'  # Options: 'scaled_dot_product', 'additive'
    USE_RELATIVE_POSITION = True           # Relative position encoding
    
    @classmethod
    def print_config(cls):
        """Print all configuration parameters"""
        print("=" * 80)
        print("GAIT RECOGNITION MODEL CONFIGURATION")
        print("=" * 80)
        print("\n📊 Data Configuration:")
        print(f"  Frames per sequence: {cls.NUM_FRAMES}")
        print(f"  Joints per frame: {cls.NUM_JOINTS}")
        print(f"  Joint dimension: {cls.JOINT_DIM}")
        
        print("\n🏗️  Model Architecture:")
        print(f"  Feature extractor: {'GCN' if cls.USE_GCN else 'CNN'}")
        print(f"  Feature dimension: {cls.FEATURE_DIM}")
        print(f"  Spatial attention heads: {cls.SPATIAL_HEADS}")
        print(f"  Temporal attention heads: {cls.TEMPORAL_HEADS}")
        print(f"  Transformer layers: {cls.TRANSFORMER_LAYERS}")
        print(f"  Transformer dimension: {cls.TRANSFORMER_DIM}")
        print(f"  Embedding dimension: {cls.EMBEDDING_DIM}")
        
        print("\n🎯 Training Configuration:")
        print(f"  Batch size: {cls.BATCH_SIZE}")
        print(f"  Learning rate: {cls.LEARNING_RATE}")
        print(f"  Epochs: {cls.NUM_EPOCHS}")
        print(f"  Loss type: {cls.LOSS_TYPE}")
        print(f"  Margin: {cls.MARGIN}")
        
        print("\n💻 Hardware:")
        print(f"  Device: {cls.DEVICE}")
        print(f"  Mixed precision: {cls.USE_AMP}")
        print("=" * 80)
    
    @classmethod
    def get_model_params(cls):
        """Return dictionary of model parameters"""
        return {
            'num_frames': cls.NUM_FRAMES,
            'num_joints': cls.NUM_JOINTS,
            'joint_dim': cls.JOINT_DIM,
            'feature_dim': cls.FEATURE_DIM,
            'spatial_heads': cls.SPATIAL_HEADS,
            'temporal_heads': cls.TEMPORAL_HEADS,
            'transformer_layers': cls.TRANSFORMER_LAYERS,
            'transformer_dim': cls.TRANSFORMER_DIM,
            'embedding_dim': cls.EMBEDDING_DIM,
            'dropout': cls.DROPOUT,
        }


if __name__ == "__main__":
    # Print configuration
    GaitConfig.print_config()
    
    # Calculate approximate parameter count
    print("\n📈 Estimated Model Size:")
    
    # Rough estimation
    gcn_params = GaitConfig.NUM_JOINTS * GaitConfig.GCN_HIDDEN_DIM * GaitConfig.FEATURE_DIM
    spatial_attn = GaitConfig.SPATIAL_DIM * GaitConfig.SPATIAL_DIM * GaitConfig.SPATIAL_HEADS
    temporal_attn = GaitConfig.TEMPORAL_DIM * GaitConfig.TEMPORAL_DIM * GaitConfig.TEMPORAL_HEADS
    transformer = (
        GaitConfig.TRANSFORMER_DIM * GaitConfig.TRANSFORMER_DIM * 4 * 
        GaitConfig.TRANSFORMER_LAYERS * GaitConfig.TRANSFORMER_HEADS
    )
    
    total_params = gcn_params + spatial_attn + temporal_attn + transformer
    print(f"  Estimated parameters: ~{total_params / 1e6:.2f}M")
    print(f"  Estimated size: ~{total_params * 4 / 1024**2:.2f} MB")
