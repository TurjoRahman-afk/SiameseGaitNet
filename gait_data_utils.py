"""
Data utilities for Gait Recognition
Handles gait sequence loading, preprocessing, and augmentation
"""
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import random

# GAIT SEQUENCE PREPROCESSING

class GaitPreprocessor:
    """
    Preprocess gait sequences (skeleton data)
    Handles normalization, alignment, and cleaning
    
    Supports Kinect V2 (25 joints) skeleton format by default:
    - Joint 0: SpineBase (hip center)
    - Joint 4: ShoulderLeft
    - Joint 8: ShoulderRight
    """
    
    def __init__(self, num_joints=25, joint_dim=3, left_shoulder_idx=4, right_shoulder_idx=8):
        self.num_joints = num_joints
        self.joint_dim = joint_dim
        self.left_shoulder_idx = left_shoulder_idx  # Kinect V2: 4
        self.right_shoulder_idx = right_shoulder_idx  # Kinect V2: 8
        
    def normalize_skeleton(self, skeleton):
        """
        Normalize skeleton by centering and scaling
        Args:
            skeleton: (num_frames, num_joints, joint_dim) - Raw skeleton sequence
        Returns:
            normalized: (num_frames, num_joints, joint_dim) - Normalized skeleton
        """
        # Center at spine/hip joint (assuming joint 0 is spine base)
        spine_position = skeleton[:, 0:1, :]  # (num_frames, 1, joint_dim)
        centered = skeleton - spine_position
        
        # Scale by maximum bone length
        max_length = np.max(np.linalg.norm(centered, axis=2))
        if max_length > 0:
            centered = centered / max_length
        """
        Before: Hand at (20, 50, 10), max_length = 100
        After: Hand at (0.2, 0.5, 0.1)
        """
        return centered
    
    def align_skeleton(self, skeleton):
        """
        Align skeleton to canonical view (frontal view)
        Uses shoulder line to determine orientation and rotates skeleton
        Args:
            skeleton: (num_frames, num_joints, joint_dim)
        Returns:
            aligned: (num_frames, num_joints, joint_dim)
        
        Note: Uses Kinect V2 format by default:
        - Joint 4: ShoulderLeft
        - Joint 8: ShoulderRight
        Customize indices in __init__ for other skeleton formats
        """
        # Extract shoulder joints using configured indices
        left_shoulder = skeleton[:, self.left_shoulder_idx, :]   # (num_frames, 3)
        right_shoulder = skeleton[:, self.right_shoulder_idx, :]  # (num_frames, 3)
        
        # Calculate shoulder vector (left to right)
        shoulder_vector = right_shoulder - left_shoulder  # (num_frames, 3)
        
        # Calculate rotation angle in XY plane (horizontal)
        # atan2(y, x) gives angle to align shoulder line horizontally
        angle = np.arctan2(shoulder_vector[:, 1], shoulder_vector[:, 0])  # (num_frames,)
        
        # Create rotation matrices for each frame (rotate to frontal view)
        cos_angle = np.cos(-angle)  # (num_frames,)
        sin_angle = np.sin(-angle)  # (num_frames,)
        zeros = np.zeros_like(cos_angle)
        ones = np.ones_like(cos_angle)
        
        # Rotation matrix (num_frames, 3, 3) for Z-axis rotation
        rotation_matrices = np.stack([
            np.stack([cos_angle, -sin_angle, zeros], axis=1),
            np.stack([sin_angle, cos_angle, zeros], axis=1),
            np.stack([zeros, zeros, ones], axis=1)
        ], axis=1)  # (num_frames, 3, 3)
        
        # Apply rotation to all joints: R @ joints^T
        # skeleton: (num_frames, num_joints, 3)
        # rotation_matrices: (num_frames, 3, 3)
        # Result: (num_frames, num_joints, 3)
        aligned = np.einsum('bij,bkj->bki', rotation_matrices, skeleton)
        
        return aligned
    
    def interpolate_sequence(self, skeleton, target_frames):
        """
        Purpose: Resize sequence to fixed number of frames (e.g., 60 frames)
        Why needed: Videos have different lengths - model needs fixed input size
        
        Args:
            skeleton: (num_frames, num_joints, joint_dim)
            target_frames: Desired number of frames
        Returns:
            resampled: (target_frames, num_joints, joint_dim)
        """
        num_frames = skeleton.shape[0]
        
        if num_frames == target_frames:
            return skeleton
        
        # Linear interpolation
        indices = np.linspace(0, num_frames - 1, target_frames)
        resampled = np.zeros((target_frames, self.num_joints, self.joint_dim))
        
        for j in range(self.num_joints):
            for d in range(self.joint_dim):
                resampled[:, j, d] = np.interp(indices, np.arange(num_frames), skeleton[:, j, d])
        
        return resampled
    
    def remove_outliers(self, skeleton, threshold=3.0):
        """
        Remove outlier joints (likely tracking errors)
        Args:
            skeleton: (num_frames, num_joints, joint_dim)
            threshold: Z-score threshold
        Returns:
            cleaned: (num_frames, num_joints, joint_dim)
        """
        # Calculate z-scores
        mean = np.mean(skeleton, axis=0, keepdims=True)
        std = np.std(skeleton, axis=0, keepdims=True)
        z_scores = np.abs((skeleton - mean) / (std + 1e-8))
        
        # Replace outliers with interpolated values
        cleaned = skeleton.copy()
        outliers = z_scores > threshold
        
        # Simple replacement with mean (better: use temporal interpolation)
        cleaned[outliers] = mean[outliers]
        
        return cleaned
    
    def preprocess(self, skeleton, target_frames=None):
        """
        Complete preprocessing pipeline for gain sequences
        """
        # Remove outliers
        skeleton = self.remove_outliers(skeleton)
        
        # Normalize
        skeleton = self.normalize_skeleton(skeleton)
        
        # Align
        skeleton = self.align_skeleton(skeleton)
        
        # Interpolate to target frames
        if target_frames is not None:
            skeleton = self.interpolate_sequence(skeleton, target_frames)
        
        return skeleton


# ============================================================================
# DATA AUGMENTATION
# ============================================================================

class GaitAugmentation:
    """
    Data augmentation for gait sequences
    """
    
    def __init__(self, config):
        self.config = config
        
    def temporal_crop(self, skeleton):
        """
        Random temporal cropping
        Args:
            skeleton: (num_frames, num_joints, joint_dim)
        Returns:
            cropped: (crop_frames, num_joints, joint_dim)
        """
        if not self.config.TEMPORAL_CROP:
            return skeleton
        
        num_frames = skeleton.shape[0]
        min_frames, max_frames = self.config.TEMPORAL_CROP_RANGE
        
        crop_frames = random.randint(min_frames, min(max_frames, num_frames))
        
        if crop_frames < num_frames:
            start = random.randint(0, num_frames - crop_frames)
            skeleton = skeleton[start:start + crop_frames]
        
        return skeleton
    
    def spatial_jitter(self, skeleton):
        """
        Add random spatial noise to joints
        Args:
            skeleton: (num_frames, num_joints, joint_dim)
        Returns:
            jittered: (num_frames, num_joints, joint_dim)
        """
        if not self.config.SPATIAL_JITTER:
            return skeleton
        
        noise = np.random.normal(0, self.config.JITTER_STD, skeleton.shape)
        return skeleton + noise
    
    def temporal_shift(self, skeleton):
        """
        Random temporal shift (circular)
        Args:
            skeleton: (num_frames, num_joints, joint_dim)
        Returns:
            shifted: (num_frames, num_joints, joint_dim)
        """
        if not self.config.TEMPORAL_SHIFT:
            return skeleton
        
        shift = random.randint(-self.config.SHIFT_RANGE, self.config.SHIFT_RANGE)
        return np.roll(skeleton, shift, axis=0)
    
    def joint_dropout(self, skeleton):
        """
        Random joint occlusion
        Args:
            skeleton: (num_frames, num_joints, joint_dim)
        Returns:
            occluded: (num_frames, num_joints, joint_dim)
        """
        if not self.config.JOINT_DROPOUT:
            return skeleton
        
        num_joints = skeleton.shape[1]
        mask = np.random.random(num_joints) > self.config.JOINT_DROPOUT_PROB
        
        skeleton_aug = skeleton.copy()
        skeleton_aug[:, ~mask, :] = 0
        
        return skeleton_aug
    
    def flip_horizontal(self, skeleton):
        """
        Horizontal flip (left-right)
        Args:
            skeleton: (num_frames, num_joints, joint_dim)
        Returns:
            flipped: (num_frames, num_joints, joint_dim)
        """
        if not self.config.FLIP_HORIZONTAL or random.random() > self.config.FLIP_PROB:
            return skeleton
        
        # Flip x-coordinate (assuming first coordinate is x)
        skeleton_flipped = skeleton.copy()
        skeleton_flipped[:, :, 0] = -skeleton_flipped[:, :, 0]
        
        # Also swap left-right joints (joint indices would need to be specified)
        # For simplicity, just flipping x-coordinate here
        
        return skeleton_flipped
    
    def speed_perturbation(self, skeleton):
        """
        Change gait speed by resampling
        Args:
            skeleton: (num_frames, num_joints, joint_dim)
        Returns:
            perturbed: (num_frames, num_joints, joint_dim)
        """
        if not self.config.SPEED_PERTURBATION:
            return skeleton
        
        speed_factor = random.uniform(*self.config.SPEED_RANGE)
        num_frames = skeleton.shape[0]
        new_frames = int(num_frames / speed_factor)
        
        # Resample to original frame count
        preprocessor = GaitPreprocessor(skeleton.shape[1], skeleton.shape[2])
        skeleton = preprocessor.interpolate_sequence(skeleton, new_frames)
        skeleton = preprocessor.interpolate_sequence(skeleton, num_frames)
        
        return skeleton
    
    def augment(self, skeleton):
        """
        Apply all augmentations
        Args:
            skeleton: (num_frames, num_joints, joint_dim)
        Returns:
            augmented: (num_frames, num_joints, joint_dim)
        """
        skeleton = self.temporal_crop(skeleton)
        skeleton = self.spatial_jitter(skeleton)
        skeleton = self.temporal_shift(skeleton)
        skeleton = self.joint_dropout(skeleton)
        skeleton = self.flip_horizontal(skeleton)
        skeleton = self.speed_perturbation(skeleton)
        
        return skeleton


# ============================================================================
# DATASET
# ============================================================================

class GaitDataset(Dataset):
    """
    Dataset for gait sequences
    For Siamese training, returns pairs or triplets
    """
    
    def __init__(self, data, labels, config, mode='pair', augment=True):
        """
        Args:
            data: List of gait sequences, each (num_frames, num_joints, joint_dim)
            labels: List of person IDs
            config: Configuration object
            mode: 'pair' for contrastive loss, 'triplet' for triplet loss
            augment: Whether to apply data augmentation
        """
        self.data = data
        self.labels = np.array(labels)
        self.config = config
        self.mode = mode
        self.augment = augment
        
        self.preprocessor = GaitPreprocessor(config.NUM_JOINTS, config.JOINT_DIM)
        self.augmentation = GaitAugmentation(config) if augment else None
        
        # Create label to indices mapping
        self.label_to_indices = {}
        for idx, label in enumerate(self.labels):
            if label not in self.label_to_indices:
                self.label_to_indices[label] = []
            self.label_to_indices[label].append(idx)
        
        self.unique_labels = list(self.label_to_indices.keys())
    
    def __len__(self):
        return len(self.data)
    
    def _get_sequence(self, idx):
        """Get preprocessed and augmented sequence"""
        skeleton = self.data[idx].copy()
        
        # Preprocess
        skeleton = self.preprocessor.preprocess(skeleton, self.config.NUM_FRAMES)
        
        # Augment
        if self.augment and self.augmentation is not None:
            skeleton = self.augmentation.augment(skeleton)
        
        # Ensure correct shape
        if skeleton.shape[0] != self.config.NUM_FRAMES:
            skeleton = self.preprocessor.interpolate_sequence(skeleton, self.config.NUM_FRAMES)
        
        return torch.FloatTensor(skeleton)
    
    def __getitem__(self, idx):
        """
        Returns:
            If mode='pair': (seq1, seq2, label)
                label=1 if same person, 0 if different
            If mode='triplet': (anchor, positive, negative)
        """
        if self.mode == 'pair':
            return self._get_pair(idx)
        elif self.mode == 'triplet':
            return self._get_triplet(idx)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
    
    def _get_pair(self, idx):
        """Get a pair for contrastive learning"""
        # Get anchor
        anchor_seq = self._get_sequence(idx)
        anchor_label = self.labels[idx]
        
        # Randomly decide: same person (positive) or different (negative)
        if random.random() > 0.5:
            # Positive pair (same person)
            positive_indices = self.label_to_indices[anchor_label].copy()
            positive_indices.remove(idx)  # Don't use same sequence
            
            if len(positive_indices) > 0:
                pair_idx = random.choice(positive_indices)
                label = 1.0
            else:
                # Fall back to negative if no other sequences for this person
                negative_label = random.choice([l for l in self.unique_labels if l != anchor_label])
                pair_idx = random.choice(self.label_to_indices[negative_label])
                label = 0.0
        else:
            # Negative pair (different person)
            negative_label = random.choice([l for l in self.unique_labels if l != anchor_label])
            pair_idx = random.choice(self.label_to_indices[negative_label])
            label = 0.0
        
        pair_seq = self._get_sequence(pair_idx)
        
        return anchor_seq, pair_seq, torch.FloatTensor([label])
    
    def _get_triplet(self, idx):
        """Get a triplet (anchor, positive, negative)"""
        # Get anchor
        anchor_seq = self._get_sequence(idx)
        anchor_label = self.labels[idx]
        
        # Get positive (same person, different sequence)
        positive_indices = self.label_to_indices[anchor_label].copy()
        positive_indices.remove(idx)
        
        if len(positive_indices) > 0:
            positive_idx = random.choice(positive_indices)
        else:
            # If no other sequence, use same sequence (less ideal)
            positive_idx = idx
        
        positive_seq = self._get_sequence(positive_idx)
        
        # Get negative (different person)
        negative_label = random.choice([l for l in self.unique_labels if l != anchor_label])
        negative_idx = random.choice(self.label_to_indices[negative_label])
        negative_seq = self._get_sequence(negative_idx)
        
        return anchor_seq, positive_seq, negative_seq


# ============================================================================
# DATA GENERATION (for testing/demo)
# ============================================================================

def generate_synthetic_gait_data(num_sequences=1000, num_classes=100, 
                                 num_frames=60, num_joints=25, joint_dim=3):
    """
    Generate synthetic gait data for testing
    In practice, replace with real gait dataset loading
    
    Returns:
        data: List of (num_frames, num_joints, joint_dim) arrays
        labels: List of person IDs
    """
    data = []
    labels = []
    
    sequences_per_class = num_sequences // num_classes
    
    for person_id in range(num_classes):
        # Generate a base gait pattern for this person
        base_pattern = np.random.randn(num_frames, num_joints, joint_dim)
        
        # Add periodic motion (simulate walking)
        t = np.linspace(0, 4 * np.pi, num_frames)
        for joint in range(num_joints):
            amplitude = np.random.uniform(0.1, 0.5)
            phase = np.random.uniform(0, 2 * np.pi)
            base_pattern[:, joint, 0] += amplitude * np.sin(t + phase)
            base_pattern[:, joint, 1] += amplitude * np.cos(t + phase)
        
        # Generate variations of this gait pattern
        for _ in range(sequences_per_class):
            # Add noise and variations
            noise = np.random.randn(num_frames, num_joints, joint_dim) * 0.1
            sequence = base_pattern + noise
            
            data.append(sequence)
            labels.append(person_id)
    
    return data, labels


def split_dataset(data, labels, train_ratio=0.7, val_ratio=0.15):
    """
    Split dataset into train/val/test
    Ensures each split has samples from all classes
    
    Returns:
        train_data, train_labels, val_data, val_labels, test_data, test_labels
    """
    from sklearn.model_selection import train_test_split
    
    # First split: train vs (val + test)
    train_data, temp_data, train_labels, temp_labels = train_test_split(
        data, labels, train_size=train_ratio, stratify=labels, random_state=42
    )
    
    # Second split: val vs test
    val_ratio_adjusted = val_ratio / (1 - train_ratio)
    val_data, test_data, val_labels, test_labels = train_test_split(
        temp_data, temp_labels, train_size=val_ratio_adjusted, stratify=temp_labels, random_state=42
    )
    
    return train_data, train_labels, val_data, val_labels, test_data, test_labels


def create_dataloaders(config, mode='pair'):
    """
    Create train/val/test dataloaders
    
    Args:
        config: Configuration object
        mode: 'pair' or 'triplet'
    
    Returns:
        train_loader, val_loader, test_loader
    """
    print("Generating synthetic gait data...")
    data, labels = generate_synthetic_gait_data(
        num_sequences=1000,
        num_classes=100,
        num_frames=config.NUM_FRAMES,
        num_joints=config.NUM_JOINTS,
        joint_dim=config.JOINT_DIM
    )
    
    print("Splitting dataset...")
    train_data, train_labels, val_data, val_labels, test_data, test_labels = split_dataset(
        data, labels, config.TRAIN_SPLIT, config.VAL_SPLIT
    )
    
    print(f"Train: {len(train_data)} sequences")
    print(f"Val: {len(val_data)} sequences")
    print(f"Test: {len(test_data)} sequences")
    
    # Create datasets
    train_dataset = GaitDataset(train_data, train_labels, config, mode=mode, augment=True)
    val_dataset = GaitDataset(val_data, val_labels, config, mode=mode, augment=False)
    test_dataset = GaitDataset(test_data, test_labels, config, mode=mode, augment=False)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY
    )
    
    return train_loader, val_loader, test_loader


# ============================================================================
# VALIDATION & DEBUGGING FUNCTIONS
# ============================================================================

def validate_preprocessing(preprocessor, sample_skeleton, verbose=True):
    """
    Validate that preprocessing is working correctly
    
    Args:
        preprocessor: GaitPreprocessor instance
        sample_skeleton: Raw skeleton (num_frames, num_joints, joint_dim)
        verbose: Print detailed info
    
    Returns:
        status: Dict with validation results
    """
    status = {'passed': True, 'errors': [], 'warnings': []}
    
    if verbose:
        print("\n" + "="*80)
        print("PREPROCESSING VALIDATION")
        print("="*80)
    
    # 1. Check raw data
    if verbose:
        print(f"\n1. Raw Data:")
        print(f"   Shape: {sample_skeleton.shape}")
        print(f"   Range: [{sample_skeleton.min():.2f}, {sample_skeleton.max():.2f}]")
        print(f"   Contains NaN: {np.isnan(sample_skeleton).any()}")
        print(f"   Contains Inf: {np.isinf(sample_skeleton).any()}")
    
    if np.isnan(sample_skeleton).any() or np.isinf(sample_skeleton).any():
        status['errors'].append("Raw data contains NaN or Inf")
        status['passed'] = False
    
    # 2. Test normalization
    if verbose:
        print(f"\n2. Testing Normalization:")
    
    try:
        normalized = preprocessor.normalize_skeleton(sample_skeleton.copy())
        
        if verbose:
            print(f"   Shape: {normalized.shape}")
            print(f"   Range: [{normalized.min():.2f}, {normalized.max():.2f}]")
            print(f"   Spine at origin: {np.allclose(normalized[:, 0, :], 0, atol=1e-6)}")
        
        # Check if normalized values are reasonable
        if abs(normalized).max() > 10:
            status['warnings'].append(f"Normalized values too large: {abs(normalized).max():.2f}")
        
        if np.isnan(normalized).any():
            status['errors'].append("Normalization produced NaN")
            status['passed'] = False
            
    except Exception as e:
        status['errors'].append(f"Normalization failed: {e}")
        status['passed'] = False
    
    # 3. Test alignment
    if verbose:
        print(f"\n3. Testing Alignment:")
    
    try:
        # Check if shoulder indices exist
        left_idx = preprocessor.left_shoulder_idx
        right_idx = preprocessor.right_shoulder_idx
        
        if left_idx >= sample_skeleton.shape[1] or right_idx >= sample_skeleton.shape[1]:
            status['errors'].append(
                f"Shoulder indices ({left_idx}, {right_idx}) out of range for {sample_skeleton.shape[1]} joints"
            )
            status['passed'] = False
        else:
            aligned = preprocessor.align_skeleton(normalized.copy())
            
            if verbose:
                # Check shoulder alignment
                left_shoulder = aligned[:, left_idx, :]
                right_shoulder = aligned[:, right_idx, :]
                shoulder_vector = right_shoulder - left_shoulder
                angles = np.arctan2(shoulder_vector[:, 1], shoulder_vector[:, 0])
                
                print(f"   Shoulder indices: {left_idx} (left), {right_idx} (right)")
                print(f"   Shoulder angles before: varied")
                print(f"   Shoulder angles after: mean={np.mean(angles):.4f}, std={np.std(angles):.4f}")
                print(f"   Aligned: {np.std(angles) < 0.1}")  # Should be close to 0
            
            if np.isnan(aligned).any():
                status['errors'].append("Alignment produced NaN")
                status['passed'] = False
                
    except Exception as e:
        status['errors'].append(f"Alignment failed: {e}")
        status['passed'] = False
    
    # 4. Test interpolation
    if verbose:
        print(f"\n4. Testing Interpolation:")
    
    try:
        target_frames = 60
        interpolated = preprocessor.interpolate_sequence(normalized.copy(), target_frames)
        
        if verbose:
            print(f"   Input frames: {normalized.shape[0]}")
            print(f"   Target frames: {target_frames}")
            print(f"   Output frames: {interpolated.shape[0]}")
            print(f"   Shape correct: {interpolated.shape[0] == target_frames}")
        
        if interpolated.shape[0] != target_frames:
            status['errors'].append(f"Interpolation failed: got {interpolated.shape[0]} frames, expected {target_frames}")
            status['passed'] = False
            
    except Exception as e:
        status['errors'].append(f"Interpolation failed: {e}")
        status['passed'] = False
    
    # 5. Test complete pipeline
    if verbose:
        print(f"\n5. Testing Complete Pipeline:")
    
    try:
        processed = preprocessor.preprocess(sample_skeleton.copy(), target_frames=60)
        
        if verbose:
            print(f"   Output shape: {processed.shape}")
            print(f"   Output range: [{processed.min():.2f}, {processed.max():.2f}]")
            print(f"   No NaN: {not np.isnan(processed).any()}")
            print(f"   No Inf: {not np.isinf(processed).any()}")
        
        if np.isnan(processed).any() or np.isinf(processed).any():
            status['errors'].append("Complete pipeline produced NaN or Inf")
            status['passed'] = False
            
    except Exception as e:
        status['errors'].append(f"Complete pipeline failed: {e}")
        status['passed'] = False
    
    # Print summary
    if verbose:
        print("\n" + "="*80)
        if status['passed']:
            print("✅ VALIDATION PASSED")
        else:
            print("❌ VALIDATION FAILED")
        
        if status['errors']:
            print("\n🚨 ERRORS:")
            for error in status['errors']:
                print(f"   • {error}")
        
        if status['warnings']:
            print("\n⚠️  WARNINGS:")
            for warning in status['warnings']:
                print(f"   • {warning}")
        print("="*80)
    
    return status


def validate_dataset_sampling(dataset, num_samples=100, verbose=True):
    """
    Validate dataset pair/triplet sampling balance
    
    Args:
        dataset: GaitDataset instance
        num_samples: Number of samples to check
        verbose: Print detailed info
    
    Returns:
        status: Dict with sampling statistics
    """
    status = {'passed': True, 'stats': {}}
    
    if verbose:
        print("\n" + "="*80)
        print("DATASET SAMPLING VALIDATION")
        print("="*80)
    
    if dataset.mode == 'pair':
        positive_count = 0
        negative_count = 0
        
        if verbose:
            print(f"\nChecking {num_samples} pairs...")
        
        for i in range(min(num_samples, len(dataset))):
            _, _, label = dataset[i]
            if label.item() == 1.0:
                positive_count += 1
            else:
                negative_count += 1
        
        positive_ratio = positive_count / (positive_count + negative_count)
        
        status['stats'] = {
            'positive_count': positive_count,
            'negative_count': negative_count,
            'positive_ratio': positive_ratio
        }
        
        if verbose:
            print(f"\n📊 Sampling Statistics:")
            print(f"   Positive pairs: {positive_count} ({positive_ratio*100:.1f}%)")
            print(f"   Negative pairs: {negative_count} ({(1-positive_ratio)*100:.1f}%)")
        
        # Check balance (should be roughly 50/50)
        if positive_ratio < 0.3 or positive_ratio > 0.7:
            status['passed'] = False
            if verbose:
                print(f"   ⚠️  WARNING: Imbalanced sampling! Should be ~50%")
        else:
            if verbose:
                print(f"   ✅ Sampling is balanced")
    
    elif dataset.mode == 'triplet':
        same_anchor_positive = 0
        
        if verbose:
            print(f"\nChecking {num_samples} triplets...")
        
        for i in range(min(num_samples, len(dataset))):
            anchor, positive, negative = dataset[i]
            
            # Check if anchor and positive are identical (bad!)
            if torch.equal(anchor, positive):
                same_anchor_positive += 1
        
        status['stats'] = {
            'same_anchor_positive': same_anchor_positive,
            'same_ratio': same_anchor_positive / num_samples
        }
        
        if verbose:
            print(f"\n📊 Sampling Statistics:")
            print(f"   Triplets with same anchor/positive: {same_anchor_positive}/{num_samples}")
        
        if same_anchor_positive > num_samples * 0.1:  # More than 10% is bad
            status['passed'] = False
            if verbose:
                print(f"   ⚠️  WARNING: Too many identical anchor/positive pairs!")
        else:
            if verbose:
                print(f"   ✅ Triplet sampling looks good")
    
    if verbose:
        print("="*80)
    
    return status


def validate_augmentation(augmentor, sample_skeleton, verbose=True):
    """
    Validate that augmentation is not too aggressive
    
    Args:
        augmentor: GaitAugmentation instance
        sample_skeleton: Skeleton to augment
        verbose: Print detailed info
    
    Returns:
        status: Dict with validation results
    """
    status = {'passed': True, 'warnings': []}
    
    if verbose:
        print("\n" + "="*80)
        print("AUGMENTATION VALIDATION")
        print("="*80)
    
    # Apply augmentation multiple times
    differences = []
    for i in range(10):
        augmented = augmentor.augment(sample_skeleton.copy())
        diff = np.abs(augmented - sample_skeleton).mean()
        differences.append(diff)
    
    mean_diff = np.mean(differences)
    
    if verbose:
        print(f"\n📊 Augmentation Statistics:")
        print(f"   Mean difference from original: {mean_diff:.4f}")
        print(f"   Std of differences: {np.std(differences):.4f}")
    
    # Check if augmentation is too aggressive
    if mean_diff > 1.0:  # Arbitrary threshold
        status['warnings'].append(f"Augmentation very aggressive (mean diff: {mean_diff:.4f})")
        if verbose:
            print(f"   ⚠️  WARNING: Augmentation might be too aggressive!")
    elif mean_diff < 0.01:
        status['warnings'].append(f"Augmentation very weak (mean diff: {mean_diff:.4f})")
        if verbose:
            print(f"   ⚠️  WARNING: Augmentation might be too weak!")
    else:
        if verbose:
            print(f"   ✅ Augmentation strength looks reasonable")
    
    if verbose:
        print("="*80)
    
    return status


def run_full_validation(config):
    """
    Run complete validation pipeline
    
    Args:
        config: GaitConfig instance
    
    Returns:
        all_passed: Boolean indicating if all validations passed
    """
    print("\n" + "="*80)
    print("RUNNING FULL VALIDATION SUITE")
    print("="*80)
    
    # Generate sample data
    print("\nGenerating test data...")
    data, labels = generate_synthetic_gait_data(
        num_sequences=50, 
        num_classes=10,
        num_frames=config.NUM_FRAMES,
        num_joints=config.NUM_JOINTS,
        joint_dim=config.JOINT_DIM
    )
    
    all_passed = True
    
    # 1. Validate preprocessing
    preprocessor = GaitPreprocessor(config.NUM_JOINTS, config.JOINT_DIM)
    result1 = validate_preprocessing(preprocessor, data[0])
    all_passed = all_passed and result1['passed']
    
    # 2. Validate augmentation
    augmentor = GaitAugmentation(config)
    result2 = validate_augmentation(augmentor, data[0])
    all_passed = all_passed and result2['passed']
    
    # 3. Validate dataset sampling (pair mode)
    dataset_pair = GaitDataset(data, labels, config, mode='pair', augment=False)
    result3 = validate_dataset_sampling(dataset_pair, num_samples=50)
    all_passed = all_passed and result3['passed']
    
    # 4. Validate dataset sampling (triplet mode)
    dataset_triplet = GaitDataset(data, labels, config, mode='triplet', augment=False)
    result4 = validate_dataset_sampling(dataset_triplet, num_samples=50)
    all_passed = all_passed and result4['passed']
    
    # Final summary
    print("\n" + "="*80)
    if all_passed:
        print("✅ ALL VALIDATIONS PASSED - READY TO TRAIN!")
    else:
        print("❌ SOME VALIDATIONS FAILED - FIX ISSUES BEFORE TRAINING!")
    print("="*80 + "\n")
    
    return all_passed


if __name__ == "__main__":
    from gait_config import GaitConfig
    
    print("=" * 80)
    print("TESTING GAIT DATA UTILITIES")
    print("=" * 80)
    
    config = GaitConfig()
    
    # Run full validation suite first
    print("\n🔍 Running validation suite...")
    all_passed = run_full_validation(config)
    
    if not all_passed:
        print("\n⚠️  Some validations failed. Review the output above.")
        print("Fix the issues before training the model.\n")
    
    # Original tests
    print("\n" + "=" * 80)
    print("RUNNING BASIC TESTS")
    print("=" * 80)
    
    # Test data generation
    print("\n1. Generating synthetic data...")
    data, labels = generate_synthetic_gait_data(num_sequences=100, num_classes=10)
    print(f"   Generated {len(data)} sequences")
    print(f"   Sequence shape: {data[0].shape}")
    print(f"   Unique labels: {len(set(labels))}")
    
    # Test preprocessing
    print("\n2. Testing preprocessing...")
    preprocessor = GaitPreprocessor(config.NUM_JOINTS, config.JOINT_DIM)
    processed = preprocessor.preprocess(data[0], target_frames=config.NUM_FRAMES)
    print(f"   Preprocessed shape: {processed.shape}")
    
    # Test augmentation
    print("\n3. Testing augmentation...")
    augmentor = GaitAugmentation(config)
    augmented = augmentor.augment(data[0])
    print(f"   Augmented shape: {augmented.shape}")
    
    # Test dataset
    print("\n4. Testing dataset...")
    dataset = GaitDataset(data, labels, config, mode='pair', augment=True)
    seq1, seq2, label = dataset[0]
    print(f"   Pair shapes: {seq1.shape}, {seq2.shape}")
    print(f"   Label: {label.item()}")
    
    dataset_triplet = GaitDataset(data, labels, config, mode='triplet', augment=True)
    anchor, positive, negative = dataset_triplet[0]
    print(f"   Triplet shapes: {anchor.shape}, {positive.shape}, {negative.shape}")
    
    # Test dataloaders
    print("\n5. Testing dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(config, mode='pair')
    
    batch = next(iter(train_loader))
    print(f"   Batch shapes: {batch[0].shape}, {batch[1].shape}, {batch[2].shape}")
    
    print("\n" + "=" * 80)
    print("ALL TESTS PASSED! ✓")
    print("=" * 80)
