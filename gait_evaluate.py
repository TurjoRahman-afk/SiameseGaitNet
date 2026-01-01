"""
Evaluation and inference for Siamese Gait Recognition Model
"""
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_curve, auc, accuracy_score
import matplotlib.pyplot as plt
import os

from gait_config import GaitConfig
from gait_model import SiameseGaitTransformer
from gait_data_utils import GaitPreprocessor


# ============================================================================
# INFERENCE CLASS
# ============================================================================

class GaitInference:
    """
    Inference class for gait recognition
    Handles model loading, embedding extraction, and similarity computation
    """
    
    def __init__(self, model_path, config=None):
        """
        Args:
            model_path: Path to saved model checkpoint
            config: Configuration object (loaded from checkpoint if None)
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load checkpoint
        print(f"Loading model from {model_path}...")
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Get configuration
        if config is None:
            self.config = checkpoint['config']
        else:
            self.config = config
        
        # Create model
        self.model = SiameseGaitTransformer(self.config).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Create preprocessor
        self.preprocessor = GaitPreprocessor(
            num_joints=self.config.NUM_JOINTS,
            joint_dim=self.config.JOINT_DIM
        )
        
        print("✓ Model loaded successfully!")
    
    @torch.no_grad()
    def get_embedding(self, gait_sequence):
        """
        Extract embedding from a gait sequence
        Args:
            gait_sequence: (num_frames, num_joints, joint_dim) numpy array
        Returns:
            embedding: (embedding_dim,) numpy array
        """
        # Preprocess
        sequence = self.preprocessor.preprocess(gait_sequence, self.config.NUM_FRAMES)
        
        # Convert to tensor and add batch dimension
        sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)
        
        # Get embedding
        embedding, _ = self.model.forward_once(sequence_tensor)
        
        return embedding.cpu().numpy().squeeze()
    
    @torch.no_grad()
    def compute_similarity(self, seq1, seq2, metric='cosine'):
        """
        Compute similarity between two gait sequences
        Args:
            seq1, seq2: (num_frames, num_joints, joint_dim) numpy arrays
            metric: 'cosine' or 'euclidean'
        Returns:
            similarity: Similarity score
            distance: Euclidean distance
        """
        # Get embeddings
        emb1 = self.get_embedding(seq1)
        emb2 = self.get_embedding(seq2)
        
        # Convert to tensors
        emb1_tensor = torch.FloatTensor(emb1).unsqueeze(0)
        emb2_tensor = torch.FloatTensor(emb2).unsqueeze(0)
        
        # Compute similarity
        if metric == 'cosine':
            similarity = F.cosine_similarity(emb1_tensor, emb2_tensor).item()
        else:
            distance = F.pairwise_distance(emb1_tensor, emb2_tensor).item()
            similarity = 1 / (1 + distance)
        
        # Compute distance
        distance = F.pairwise_distance(emb1_tensor, emb2_tensor).item()
        
        return similarity, distance
    
    @torch.no_grad()
    def batch_similarity(self, sequences):
        """
        Compute pairwise similarities for a list of sequences
        Args:
            sequences: List of (num_frames, num_joints, joint_dim) arrays
        Returns:
            similarity_matrix: (num_sequences, num_sequences) similarity matrix
        """
        # Get all embeddings
        embeddings = []
        for seq in sequences:
            emb = self.get_embedding(seq)
            embeddings.append(emb)
        
        embeddings = np.array(embeddings)
        embeddings_tensor = torch.FloatTensor(embeddings)
        
        # Compute pairwise cosine similarities
        embeddings_norm = F.normalize(embeddings_tensor, p=2, dim=1)
        similarity_matrix = torch.mm(embeddings_norm, embeddings_norm.t())
        
        return similarity_matrix.numpy()
    
    @torch.no_grad()
    def identify_person(self, query_sequence, gallery_sequences, gallery_labels, top_k=5):
        """
        Identify person from gallery
        Args:
            query_sequence: Query gait sequence
            gallery_sequences: List of gallery sequences
            gallery_labels: List of person IDs for gallery
            top_k: Return top-k matches
        Returns:
            top_matches: List of (person_id, similarity, distance) tuples
        """
        # Get query embedding
        query_emb = self.get_embedding(query_sequence)
        query_tensor = torch.FloatTensor(query_emb).unsqueeze(0)
        
        # Compute similarities with all gallery sequences
        similarities = []
        for gallery_seq in gallery_sequences:
            gallery_emb = self.get_embedding(gallery_seq)
            gallery_tensor = torch.FloatTensor(gallery_emb).unsqueeze(0)
            
            sim = F.cosine_similarity(query_tensor, gallery_tensor).item()
            dist = F.pairwise_distance(query_tensor, gallery_tensor).item()
            
            similarities.append((sim, dist))
        
        # Sort by similarity (descending)
        indices = np.argsort([s[0] for s in similarities])[::-1]
        
        # Get top-k matches
        top_matches = []
        for idx in indices[:top_k]:
            person_id = gallery_labels[idx]
            sim, dist = similarities[idx]
            top_matches.append((person_id, sim, dist))
        
        return top_matches
    
    @torch.no_grad()
    def visualize_attention(self, gait_sequence, save_path=None):
        """
        Visualize attention weights for a gait sequence
        Args:
            gait_sequence: (num_frames, num_joints, joint_dim)
            save_path: Path to save visualization
        """
        # Preprocess
        sequence = self.preprocessor.preprocess(gait_sequence, self.config.NUM_FRAMES)
        sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)
        
        # Get embedding and attention maps
        embedding, attention_maps = self.model.forward_once(sequence_tensor)
        
        # Extract attention weights
        spatial_attn = attention_maps['spatial']
        temporal_attn = attention_maps['temporal']
        
        # Visualize
        from gait_model import visualize_attention
        visualize_attention(attention_maps, save_path)


# ============================================================================
# EVALUATION METRICS
# ============================================================================

class GaitEvaluator:
    """
    Evaluation metrics for gait recognition
    """
    
    def __init__(self, inference_model):
        self.inference = inference_model
    
    def evaluate_verification(self, test_pairs, test_labels, threshold=0.5):
        """
        Evaluate verification performance (same/different person)
        Args:
            test_pairs: List of (seq1, seq2) tuples
            test_labels: List of labels (1=same person, 0=different)
            threshold: Distance threshold for classification
        Returns:
            metrics: Dict with accuracy, EER, AUC, etc.
        """
        print("Evaluating verification performance...")
        
        similarities = []
        distances = []
        
        for seq1, seq2 in test_pairs:
            sim, dist = self.inference.compute_similarity(seq1, seq2)
            similarities.append(sim)
            distances.append(dist)
        
        # Convert to arrays
        similarities = np.array(similarities)
        distances = np.array(distances)
        labels = np.array(test_labels)
        
        # Predict using distance threshold
        predictions = (distances < threshold).astype(int)
        accuracy = accuracy_score(labels, predictions)
        
        # ROC curve
        fpr, tpr, thresholds = roc_curve(labels, -distances)  # Negative distance (lower is more similar)
        roc_auc = auc(fpr, tpr)
        
        # Equal Error Rate (EER)
        fnr = 1 - tpr
        eer_idx = np.nanargmin(np.absolute(fnr - fpr))
        eer = (fpr[eer_idx] + fnr[eer_idx]) / 2
        
        metrics = {
            'accuracy': accuracy,
            'auc': roc_auc,
            'eer': eer,
            'fpr': fpr,
            'tpr': tpr,
            'thresholds': thresholds
        }
        
        return metrics
    
    def evaluate_identification(self, query_sequences, query_labels, 
                                gallery_sequences, gallery_labels, top_k=[1, 5, 10]):
        """
        Evaluate identification performance (find person in gallery)
        Args:
            query_sequences: List of query sequences
            query_labels: List of person IDs for queries
            gallery_sequences: List of gallery sequences
            gallery_labels: List of person IDs for gallery
            top_k: List of k values for top-k accuracy
        Returns:
            metrics: Dict with rank-k accuracies
        """
        print("Evaluating identification performance...")
        
        num_queries = len(query_sequences)
        max_k = max(top_k)
        
        rank_scores = np.zeros((num_queries, max_k))
        
        for i, (query_seq, query_label) in enumerate(zip(query_sequences, query_labels)):
            # Get top matches
            matches = self.inference.identify_person(
                query_seq, gallery_sequences, gallery_labels, top_k=max_k
            )
            
            # Check if correct person is in top-k
            for rank, (person_id, _, _) in enumerate(matches):
                if person_id == query_label:
                    rank_scores[i, rank:] = 1  # Correct for this rank and all higher ranks
                    break
        
        # Calculate rank-k accuracies
        metrics = {}
        for k in top_k:
            rank_k_acc = np.mean(rank_scores[:, k-1])
            metrics[f'rank_{k}_accuracy'] = rank_k_acc
        
        return metrics
    
    def plot_roc_curve(self, metrics, save_path=None):
        """Plot ROC curve"""
        plt.figure(figsize=(8, 6))
        
        plt.plot(metrics['fpr'], metrics['tpr'], 'b-', linewidth=2,
                label=f"ROC (AUC = {metrics['auc']:.4f})")
        plt.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Random')
        
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curve - Gait Verification', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ ROC curve saved to {save_path}")
        else:
            plt.show()
        
        plt.close()


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

def example_basic_similarity():
    """Example 1: Basic similarity computation"""
    print("\n" + "=" * 80)
    print("EXAMPLE 1: Basic Gait Similarity")
    print("=" * 80)
    
    config = GaitConfig()
    model_path = os.path.join(config.MODEL_SAVE_PATH, 'best_model.pth')
    
    if not os.path.exists(model_path):
        print("❌ Model not found! Please train the model first.")
        return
    
    # Load inference model
    inference = GaitInference(model_path)
    
    # Generate sample gait sequences
    from gait_data_utils import generate_synthetic_gait_data
    data, labels = generate_synthetic_gait_data(
        num_sequences=10, num_classes=5,
        num_frames=config.NUM_FRAMES,
        num_joints=config.NUM_JOINTS,
        joint_dim=config.JOINT_DIM
    )
    
    # Compare two sequences from same person
    seq1 = data[0]
    seq2 = data[1]  # Same label
    
    similarity, distance = inference.compute_similarity(seq1, seq2)
    
    print(f"\nComparing sequences from same person:")
    print(f"  Cosine Similarity: {similarity:.4f}")
    print(f"  Euclidean Distance: {distance:.4f}")
    print(f"  Prediction: {'✓ SAME PERSON' if distance < 0.5 else '✗ DIFFERENT PERSON'}")
    
    # Compare sequences from different people
    seq3 = data[5]  # Different label
    similarity2, distance2 = inference.compute_similarity(seq1, seq3)
    
    print(f"\nComparing sequences from different people:")
    print(f"  Cosine Similarity: {similarity2:.4f}")
    print(f"  Euclidean Distance: {distance2:.4f}")
    print(f"  Prediction: {'✓ SAME PERSON' if distance2 < 0.5 else '✗ DIFFERENT PERSON'}")


def example_person_identification():
    """Example 2: Person identification from gallery"""
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Person Identification")
    print("=" * 80)
    
    config = GaitConfig()
    model_path = os.path.join(config.MODEL_SAVE_PATH, 'best_model.pth')
    
    if not os.path.exists(model_path):
        print("❌ Model not found! Please train the model first.")
        return
    
    # Load inference model
    inference = GaitInference(model_path)
    
    # Generate sample data
    from gait_data_utils import generate_synthetic_gait_data
    data, labels = generate_synthetic_gait_data(
        num_sequences=50, num_classes=10,
        num_frames=config.NUM_FRAMES,
        num_joints=config.NUM_JOINTS,
        joint_dim=config.JOINT_DIM
    )
    
    # Split into query and gallery
    query_seq = data[0]
    query_label = labels[0]
    
    gallery_sequences = data[10:]
    gallery_labels = labels[10:]
    
    print(f"\nQuery person ID: {query_label}")
    print(f"Gallery size: {len(gallery_sequences)} sequences")
    
    # Identify person
    top_matches = inference.identify_person(query_seq, gallery_sequences, gallery_labels, top_k=5)
    
    print("\nTop 5 matches:")
    for rank, (person_id, sim, dist) in enumerate(top_matches, 1):
        status = "✓ CORRECT" if person_id == query_label else "✗"
        print(f"  {rank}. Person {person_id}: similarity={sim:.4f}, distance={dist:.4f} {status}")


def example_attention_visualization():
    """Example 3: Visualize attention weights"""
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Attention Visualization")
    print("=" * 80)
    
    config = GaitConfig()
    model_path = os.path.join(config.MODEL_SAVE_PATH, 'best_model.pth')
    
    if not os.path.exists(model_path):
        print("❌ Model not found! Please train the model first.")
        return
    
    # Load inference model
    inference = GaitInference(model_path)
    
    # Generate sample gait sequence
    from gait_data_utils import generate_synthetic_gait_data
    data, _ = generate_synthetic_gait_data(
        num_sequences=1, num_classes=1,
        num_frames=config.NUM_FRAMES,
        num_joints=config.NUM_JOINTS,
        joint_dim=config.JOINT_DIM
    )
    
    # Visualize attention
    save_path = os.path.join(config.RESULTS_DIR, 'attention_visualization.png')
    inference.visualize_attention(data[0], save_path)
    
    print(f"\n✓ Attention visualization saved to {save_path}")


def main():
    """Run all examples"""
    print("\n" + "=" * 80)
    print("GAIT RECOGNITION - EVALUATION & EXAMPLES")
    print("=" * 80)
    
    try:
        example_basic_similarity()
        example_person_identification()
        example_attention_visualization()
        
        print("\n" + "=" * 80)
        print("ALL EXAMPLES COMPLETED! ✓")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Make sure to train the model first: python gait_train.py")


if __name__ == "__main__":
    main()
