"""
Training script for Siamese Spatio-Temporal Attention Transformer
Handles training loop, validation, checkpointing, and visualization
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import os
import time
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from gait_config import GaitConfig
from gait_model import SiameseGaitTransformer, ContrastiveLoss, TripletLoss, count_parameters, get_model_size
from gait_data_utils import create_dataloaders


# ============================================================================
# TRAINER CLASS
# ============================================================================

class GaitTrainer:
    """
    Trainer for Siamese Gait Recognition Model
    """
    
    def __init__(self, config):
        self.config = config
        self.device = config.DEVICE
        
        # Create model
        print("Initializing model...")
        self.model = SiameseGaitTransformer(config).to(self.device)
        print(f"Model parameters: {count_parameters(self.model):,}")
        print(f"Model size: {get_model_size(self.model):.2f} MB")
        
        # Loss function
        if config.LOSS_TYPE == 'contrastive':
            self.criterion = ContrastiveLoss(margin=config.MARGIN)
            self.mode = 'pair'
        elif config.LOSS_TYPE == 'triplet':
            self.criterion = TripletLoss(margin=config.MARGIN, mining=config.TRIPLET_MINING)
            self.mode = 'triplet'
        elif config.LOSS_TYPE == 'combined':
            self.criterion_contrastive = ContrastiveLoss(margin=config.MARGIN)
            self.criterion_triplet = TripletLoss(margin=config.MARGIN)
            self.mode = 'triplet'  # Use triplet mode for data loading
        else:
            raise ValueError(f"Unknown loss type: {config.LOSS_TYPE}")
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )
        
        # Learning rate scheduler
        if config.LR_SCHEDULER == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, step_size=config.LR_STEP_SIZE, gamma=config.LR_GAMMA
            )
        elif config.LR_SCHEDULER == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=config.NUM_EPOCHS
            )
        elif config.LR_SCHEDULER == 'plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', patience=5, factor=0.5
            )
        else:
            self.scheduler = None
        
        # Mixed precision training
        self.scaler = GradScaler() if config.USE_AMP else None
        
        # Create data loaders
        print("\nLoading data...")
        self.train_loader, self.val_loader, self.test_loader = create_dataloaders(config, mode=self.mode)
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        # History
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []
        
        # Create directories
        os.makedirs(config.MODEL_SAVE_PATH, exist_ok=True)
        os.makedirs(config.LOG_DIR, exist_ok=True)
        os.makedirs(config.RESULTS_DIR, exist_ok=True)
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")
        
        for batch_idx, batch_data in enumerate(progress_bar):
            # Move data to device
            if self.mode == 'pair':
                seq1, seq2, labels = batch_data
                seq1 = seq1.to(self.device)
                seq2 = seq2.to(self.device)
                labels = labels.to(self.device).squeeze()
            else:  # triplet
                anchor, positive, negative = batch_data
                anchor = anchor.to(self.device)
                positive = positive.to(self.device)
                negative = negative.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision
            if self.config.USE_AMP:
                with autocast():
                    loss = self._compute_loss(batch_data)
                
                # Backward pass with gradient scaling
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                if self.config.GRAD_CLIP > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.GRAD_CLIP)
                
                # Optimizer step
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss = self._compute_loss(batch_data)
                loss.backward()
                
                # Gradient clipping
                if self.config.GRAD_CLIP > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.GRAD_CLIP)
                
                self.optimizer.step()
            
            # Update metrics
            epoch_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{epoch_loss / num_batches:.4f}'
            })
        
        avg_loss = epoch_loss / num_batches
        return avg_loss
    
    def _compute_loss(self, batch_data):
        """Compute loss for a batch"""
        if self.mode == 'pair':
            seq1, seq2, labels = batch_data
            seq1 = seq1.to(self.device)
            seq2 = seq2.to(self.device)
            labels = labels.to(self.device).squeeze()
            
            # Forward pass
            emb1, emb2, _, _ = self.model(seq1, seq2)
            
            # Compute loss
            loss = self.criterion(emb1, emb2, labels)
            
        else:  # triplet
            anchor, positive, negative = batch_data
            anchor = anchor.to(self.device)
            positive = positive.to(self.device)
            negative = negative.to(self.device)
            
            # Forward pass
            anchor_emb, _, _ = self.model.forward_once(anchor)
            positive_emb, _, _ = self.model.forward_once(positive)
            negative_emb, _, _ = self.model.forward_once(negative)
            
            # Compute loss
            if self.config.LOSS_TYPE == 'combined':
                # Combined contrastive and triplet loss
                loss_triplet = self.criterion_triplet(anchor_emb, positive_emb, negative_emb)
                
                # Create contrastive pairs
                labels_pos = torch.ones(anchor_emb.size(0)).to(self.device)
                labels_neg = torch.zeros(anchor_emb.size(0)).to(self.device)
                loss_contrastive_pos = self.criterion_contrastive(anchor_emb, positive_emb, labels_pos)
                loss_contrastive_neg = self.criterion_contrastive(anchor_emb, negative_emb, labels_neg)
                loss_contrastive = (loss_contrastive_pos + loss_contrastive_neg) / 2
                
                loss = (self.config.TRIPLET_WEIGHT * loss_triplet + 
                       self.config.CONTRASTIVE_WEIGHT * loss_contrastive)
            else:
                loss = self.criterion(anchor_emb, positive_emb, negative_emb)
        
        return loss
    
    @torch.no_grad()
    def validate(self):
        """Validate on validation set"""
        self.model.eval()
        val_loss = 0.0
        num_batches = 0
        
        for batch_data in tqdm(self.val_loader, desc="Validation"):
            loss = self._compute_loss(batch_data)
            val_loss += loss.item()
            num_batches += 1
        
        avg_val_loss = val_loss / num_batches
        return avg_val_loss
    
    def train(self):
        """Complete training loop"""
        print("\n" + "=" * 80)
        print("STARTING TRAINING")
        print("=" * 80)
        
        start_time = time.time()
        
        for epoch in range(self.config.NUM_EPOCHS):
            self.current_epoch = epoch + 1
            
            # Train
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss = self.validate()
            self.val_losses.append(val_loss)
            
            # Learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            self.learning_rates.append(current_lr)
            
            # Print epoch summary
            print(f"\nEpoch {self.current_epoch}/{self.config.NUM_EPOCHS}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  Learning Rate: {current_lr:.6f}")
            
            # Learning rate scheduling
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.save_checkpoint('best_model.pth')
                print(f"  ✓ Best model saved (val_loss: {val_loss:.4f})")
            else:
                self.patience_counter += 1
            
            # Save periodic checkpoint
            if (self.current_epoch) % self.config.SAVE_INTERVAL == 0:
                self.save_checkpoint(f'checkpoint_epoch_{self.current_epoch}.pth')
            
            # Early stopping
            if self.patience_counter >= self.config.PATIENCE:
                print(f"\n⚠️  Early stopping triggered after {self.current_epoch} epochs")
                break
        
        # Training completed
        total_time = time.time() - start_time
        print("\n" + "=" * 80)
        print("TRAINING COMPLETED")
        print("=" * 80)
        print(f"Total time: {total_time / 3600:.2f} hours")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        
        # Plot training history
        self.plot_training_history()
        
        # Save final checkpoint
        self.save_checkpoint('final_model.pth')
    
    def save_checkpoint(self, filename):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_loss': self.best_val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'config': self.config,
        }
        
        filepath = os.path.join(self.config.MODEL_SAVE_PATH, filename)
        torch.save(checkpoint, filepath)
    
    def load_checkpoint(self, filename):
        """Load model checkpoint"""
        filepath = os.path.join(self.config.MODEL_SAVE_PATH, filename)
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        
        print(f"Checkpoint loaded from epoch {self.current_epoch}")
    
    def plot_training_history(self):
        """Plot and save training history"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss curves
        epochs = range(1, len(self.train_losses) + 1)
        axes[0].plot(epochs, self.train_losses, 'b-', label='Train Loss', linewidth=2)
        axes[0].plot(epochs, self.val_losses, 'r-', label='Val Loss', linewidth=2)
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Loss', fontsize=12)
        axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)
        
        # Learning rate
        axes[1].plot(epochs, self.learning_rates, 'g-', linewidth=2)
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Learning Rate', fontsize=12)
        axes[1].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        axes[1].set_yscale('log')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        save_path = os.path.join(self.config.RESULTS_DIR, 'training_history.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Training history plot saved to {save_path}")
        plt.close()


# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def main():
    """Main training function"""
    # Load configuration
    config = GaitConfig()
    config.print_config()
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    # Create trainer
    trainer = GaitTrainer(config)
    
    # Train model
    trainer.train()
    
    print("\n✓ Training pipeline completed successfully!")
    print(f"Model saved to: {config.MODEL_SAVE_PATH}")


if __name__ == "__main__":
    main()
