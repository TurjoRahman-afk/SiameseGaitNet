"""
Siamese Spatio-Temporal Attention Transformer for Gait Recognition
Complete model architecture with all components
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math



# SPATIAL ATTENTION MODULE


class SpatialAttention(nn.Module):
    """
    Multi-head spatial attention to focus on important body joints/parts
    Learns which joints are discriminative for gait recognition
    """
    
    def __init__(self, feature_dim, num_heads=8, dropout=0.1):
        super(SpatialAttention, self).__init__()
        
        assert feature_dim % num_heads == 0, "feature_dim must be divisible by num_heads"
        
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Query, Key, Value projections
        self.q_proj = nn.Linear(feature_dim, feature_dim)
        self.k_proj = nn.Linear(feature_dim, feature_dim)
        self.v_proj = nn.Linear(feature_dim, feature_dim)
        
        # Output projection
        self.out_proj = nn.Linear(feature_dim, feature_dim)
        
        # Dropout
        self.attn_dropout = nn.Dropout(dropout) # applied to attention weights 
        self.proj_dropout = nn.Dropout(dropout) # applied to output projection 
        
        # Layer normalization for stable training 
        self.norm = nn.LayerNorm(feature_dim)
        
    def forward(self, x):
        """
        Args:
            x: (batch, num_joints, feature_dim) - Joint features
        Returns:
            out: (batch, num_joints, feature_dim) - Attended features
            attn_weights: (batch, num_heads, num_joints, num_joints) - Attention weights
        """
        batch_size, num_joints, _ = x.size()
        
        # Residual connection
        residual = x
        
        # Project to Q, K, V
        Q = self.q_proj(x)  # (batch, num_joints, feature_dim)
        K = self.k_proj(x)
        V = self.v_proj(x)
        
        # Reshape for multi-head attention
        # (batch, num_joints, num_heads, head_dim) -> (batch, num_heads, num_joints, head_dim)
        Q = Q.view(batch_size, num_joints, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, num_joints, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, num_joints, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        # (batch, num_heads, num_joints, num_joints)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        # Apply attention to values
        # (batch, num_heads, num_joints, head_dim)
        attn_output = torch.matmul(attn_weights, V)
        
        # Reshape back
        # (batch, num_heads, num_joints, head_dim) -> (batch, num_joints, feature_dim)
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, num_joints, self.feature_dim
        )
        
        # Output projection
        out = self.out_proj(attn_output)
        out = self.proj_dropout(out)
        
        # Residual connection and normalization
        out = self.norm(residual + out)
        
        return out, attn_weights




#Temporal Attention Module

class TemporalAttention(nn.Module):
    """
    Multi-head temporal attention to capture key gait phases
    Learns which frames are important in the gait cycle
    """
    
    def __init__(self, feature_dim, num_heads=8, dropout=0.1):
        super(TemporalAttention, self).__init__()
        
        assert feature_dim % num_heads == 0, "feature_dim must be divisible by num_heads"
        
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Query, Key, Value projections
        self.q_proj = nn.Linear(feature_dim, feature_dim)
        self.k_proj = nn.Linear(feature_dim, feature_dim)
        self.v_proj = nn.Linear(feature_dim, feature_dim)
        
        # Output projection
        self.out_proj = nn.Linear(feature_dim, feature_dim)
        
        # Dropout
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)
        
        # Layer normalization
        self.norm = nn.LayerNorm(feature_dim)
        
    def forward(self, x):
        """
        Args:
            x: (batch, num_frames, feature_dim) - Frame features
        Returns:
            out: (batch, num_frames, feature_dim) - Attended features
            attn_weights: (batch, num_heads, num_frames, num_frames) - Attention weights
        """
        batch_size, num_frames, _ = x.size()
        
        # Residual connection
        residual = x
        
        # Project to Q, K, V
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, num_frames, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, num_frames, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, num_frames, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, V)
        
        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, num_frames, self.feature_dim
        )
        
        # Output projection
        out = self.out_proj(attn_output)
        out = self.proj_dropout(out)
        
        # Residual connection and normalization
        out = self.norm(residual + out)
        
        return out, attn_weights


# ============================================================================
# 3. FRAME-LEVEL FEATURE EXTRACTOR (GCN-based)
# ============================================================================

class GraphConvLayer(nn.Module):
    """
    Single Graph Convolutional Layer for skeleton data
    """
    
    def __init__(self, in_channels, out_channels, adjacency_matrix):
        super(GraphConvLayer, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Register adjacency matrix as buffer
        self.register_buffer('adjacency', adjacency_matrix)
        
        # Learnable weight matrix
        self.weight = nn.Parameter(torch.FloatTensor(in_channels, out_channels))
        self.bias = nn.Parameter(torch.FloatTensor(out_channels))
        
        # Initialize parameters
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)
        
    def forward(self, x):
        """
        Args:
            x: (batch, num_joints, in_channels)
        Returns:
            out: (batch, num_joints, out_channels)
        """
        # Graph convolution: A * X * W
        # x: (batch, num_joints, in_channels)
        # adjacency: (num_joints, num_joints)
        # weight: (in_channels, out_channels)
        
        # Step 1: X * W
        x = torch.matmul(x, self.weight)  # (batch, num_joints, out_channels)
        
        # Step 2: A * (X * W)
        x = torch.matmul(self.adjacency, x)  # (batch, num_joints, out_channels)
        
        # Add bias
        x = x + self.bias
        
        return x


class GCNFeatureExtractor(nn.Module):
    """
    Graph Convolutional Network for extracting features from skeleton frames
    Models spatial relationships between body joints
    """
    
    def __init__(self, num_joints=25, joint_dim=3, hidden_dim=128, output_dim=256, dropout=0.1):
        super(GCNFeatureExtractor, self).__init__()
        
        self.num_joints = num_joints
        
        # Create adjacency matrix for human skeleton
        # This is a simplified version - should be customized based on skeleton structure
        adjacency = self._create_adjacency_matrix(num_joints)# (25, 25) matrix
        
        # GCN layers
        self.gcn1 = GraphConvLayer(joint_dim, hidden_dim, adjacency)    #Extract basic features
        self.gcn2 = GraphConvLayer(hidden_dim, hidden_dim, adjacency)    # Refine features 
        self.gcn3 = GraphConvLayer(hidden_dim, output_dim, adjacency)     # Rich Representations
        
        # Batch normalization
        self.bn1 = nn.BatchNorm1d(num_joints)
        self.bn2 = nn.BatchNorm1d(num_joints)
        self.bn3 = nn.BatchNorm1d(num_joints)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def _create_adjacency_matrix(self, num_joints):
        """
        Create adjacency matrix for skeleton graph
        For simplicity, using a chain connection + self-loops
        In practice, use actual skeleton connectivity (e.g., Kinect V2 skeleton)
        """
        adj = torch.eye(num_joints)  # Self-loops
        
        # Chain connections (simplified skeleton structure)
        for i in range(num_joints - 1):
            adj[i, i + 1] = 1
            adj[i + 1, i] = 1
        
        # Normalize adjacency matrix
        degree = adj.sum(dim=1) # counts connection per joints 
        degree_inv_sqrt = torch.pow(degree, -0.5)
        degree_inv_sqrt[torch.isinf(degree_inv_sqrt)] = 0
        adj_norm = degree_inv_sqrt.unsqueeze(1) * adj * degree_inv_sqrt.unsqueeze(0)   # normalizze adjacency
        
        return adj_norm
    
    def forward(self, x):    # forward pass proessing a frame 
        """
        Args:
            x: (batch, num_joints, joint_dim) - Single frame skeleton
        Returns:
            features: (batch, num_joints, output_dim) - Joint features
        """
        # GCN layer 1
        x = self.gcn1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # GCN layer 2
        x = self.gcn2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # GCN layer 3
        x = self.gcn3(x)
        x = self.bn3(x)
        x = F.relu(x)
        
        return x


# ============================================================================
# 4. POSITIONAL ENCODING
# ============================================================================

class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for temporal information
    """
    
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model) #shape = 5000, 512 // Frame,Feature
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            x: (batch, seq_len, d_model) with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# ============================================================================
# 5. TRANSFORMER ENCODER
# ============================================================================

class TransformerEncoderLayer(nn.Module):
    """
    Single Transformer Encoder Layer with multi-head self-attention and FFN
    """
    
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        
        # Multi-head self-attention
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, 
        dropout=dropout, batch_first=True)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            out: (batch, seq_len, d_model)
        """
        # Self-attention with residual connection
        attn_out, _ = self.self_attn(x, x, x)
        x = self.norm1(x + self.dropout1(attn_out))
        
        # Feed-forward with residual connection
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_out))
        
        return x


class TransformerEncoder(nn.Module):
    """
    Stack of Transformer Encoder Layers for long-range temporal dependencies
    """
    
    def __init__(self, d_model, num_heads, d_ff, num_layers, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            out: (batch, seq_len, d_model)
        """
        for layer in self.layers:
            x = layer(x)
        
        return self.norm(x)


# 6. HIERARCHICAL FEATURE AGGREGATION
class HierarchicalAggregation(nn.Module):
    """
    Multi-scale temporal pooling and feature aggregation
    Combines features at different temporal resolutions
    """
    
    def __init__(self, input_dim, output_dim, aggregation_levels=[1, 2, 4, 8]):
        super(HierarchicalAggregation, self).__init__()
        
        self.aggregation_levels = aggregation_levels # this stores which temporal scales to use: [1, 2, 4, 8]
        
        # Pooling layers for each level 4 level = 4 pooling layer 
        self.pools = nn.ModuleList([
            nn.AdaptiveAvgPool1d(1) for _ in aggregation_levels
        ])
        
        # Feature fusion
        total_features = input_dim * len(aggregation_levels)
        self.fusion = nn.Sequential(
            nn.Linear(total_features, input_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(input_dim, output_dim)
        )
        
        # Additional statistical aggregations
        self.use_statistics = True
        
    def forward(self, x):
        """
        Args:
            x: (batch, num_frames, feature_dim)
        Returns:
            aggregated: (batch, output_dim)
        """
        batch_size, num_frames, feature_dim = x.size()
        
        # (batch, frames, features) → (batch, features, frames)
        x_transposed = x.transpose(1, 2)
        
        # Multi-scale pooling
        pooled_features = []
        
        for level, pool in zip(self.aggregation_levels, self.pools): # for loop for level 1, 2, 4, 8
            # Segment-level pooling
            if level > 1:
                # Split into segments
                segment_size = num_frames // level
                segments = []
                for i in range(level):
                    start = i * segment_size
                    end = start + segment_size if i < level - 1 else num_frames
                    segment = x_transposed[:, :, start:end]
                    segment_pooled = F.adaptive_avg_pool1d(segment, 1)
                    segments.append(segment_pooled)
                level_features = torch.cat(segments, dim=2)  # concatenate the 2 polled segments
                level_features = F.adaptive_avg_pool1d(level_features, 1)  # pool again to get single representation
            else: #level == 1
                # Global pooling
                level_features = pool(x_transposed)  # (batch, feature_dim, 1)
            
            pooled_features.append(level_features.squeeze(-1))  # (batch, feature_dim), collect all scale features
        """pooled_features = [
                (4, 512),  # Level 1 - global
                (4, 512),  # Level 2 - halves
                (4, 512),  # Level 4 - quarters  
                (4, 512)   # Level 8 - eighths
            ]
        """

        # Concatenate all levels
        concatenated = torch.cat(pooled_features, dim=1)  # (batch, feature_dim * num_levels)
        
        # Additional statisticAL aggregations
        if self.use_statistics:
            # Max pooling
            max_pool = torch.max(x, dim=1)[0]  # (batch, feature_dim)
            # Standard deviation
            std_pool = torch.std(x, dim=1)  # (batch, feature_dim)
            
            concatenated = torch.cat([concatenated, max_pool, std_pool], dim=1)
        
        # Fusion to output dimension
        aggregated = self.fusion(concatenated)
        
        return aggregated

# 7. COMPLETE GAIT ENCODER
class GaitEncoder(nn.Module):
    """
    Complete gait sequence encoder combining all components:
    GCN -> Spatial Attention -> Temporal Attention -> Transformer -> Hierarchical Aggregation
    """
    
    def __init__(self, config):
        super(GaitEncoder, self).__init__()
        
        self.num_frames = config.NUM_FRAMES
        self.num_joints = config.NUM_JOINTS
        
        # 1. Frame-level feature extractor (GCN)
        # extracts featues from each frames skeleton using GCN
        self.feature_extractor = GCNFeatureExtractor(
            num_joints=config.NUM_JOINTS,
            joint_dim=config.JOINT_DIM,
            hidden_dim=config.GCN_HIDDEN_DIM,
            output_dim=config.FEATURE_DIM,
            dropout=config.DROPOUT
        )
        
        # 2. Spatial attention (across joints)
        # learns which body joints are important for gait recognition 
        self.spatial_attention = SpatialAttention(
            feature_dim=config.SPATIAL_DIM,
            num_heads=config.SPATIAL_HEADS,
            dropout=config.SPATIAL_DROPOUT
        )
        
        # Spatial feature projection
        # shows GCN features to spatial attention dimension
        self.spatial_proj = nn.Linear(config.FEATURE_DIM, config.SPATIAL_DIM)
        
        # 3. Temporal feature projection 
        # compresses all joint features into single frame vector 
        self.temporal_reduce = nn.Sequential(
            nn.Linear(config.NUM_JOINTS * config.SPATIAL_DIM, config.TEMPORAL_DIM),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT)
        )
        
        # 4. Temporal attention (across frames)
        # learns which frams/ gait phases are important
        self.temporal_attention = TemporalAttention(
            feature_dim=config.TEMPORAL_DIM,
            num_heads=config.TEMPORAL_HEADS,
            dropout=config.TEMPORAL_DROPOUT
        )
        
        # 5. Positional encoding
        self.positional_encoding = PositionalEncoding(
            d_model=config.TRANSFORMER_DIM,
            max_len=config.NUM_FRAMES,
            dropout=config.TRANSFORMER_DROPOUT
        )
        
        # Projection to transformer dimension
        #Projects to transformer dimension before positional encoding
        self.transformer_proj = nn.Linear(config.TEMPORAL_DIM, config.TRANSFORMER_DIM)
        
        # 6. Transformer encoder
        # captures long-range temporal dependeicies across entire sequence
        self.transformer = TransformerEncoder(
            d_model=config.TRANSFORMER_DIM,
            num_heads=config.TRANSFORMER_HEADS,
            d_ff=config.TRANSFORMER_FF_DIM,
            num_layers=config.TRANSFORMER_LAYERS,
            dropout=config.TRANSFORMER_DROPOUT
        )
        
        # 7. Hierarchical aggregation

        self.aggregation = HierarchicalAggregation(
            input_dim=config.TRANSFORMER_DIM,
            output_dim=config.EMBEDDING_DIM,
            aggregation_levels=config.AGGREGATION_LEVELS
        )
        
        # Batch normalization for final embedding
        #Normalizes final embeddings for stable training and better similarity computation
        self.bn_final = nn.BatchNorm1d(config.EMBEDDING_DIM)
        
    def forward(self, x):
        """
        Args:
            x: (batch, num_frames, num_joints, joint_dim) - Gait sequence
        Returns:
            embedding: (batch, embedding_dim) - Final gait embedding
            attention_maps: dict - Spatial and temporal attention weights
        """
        batch_size, num_frames, num_joints, joint_dim = x.size()
        
        # Process each frame through GCN
        frame_features = []
        for t in range(num_frames):
            frame = x[:, t, :, :]  # (batch, num_joints, joint_dim)
            features = self.feature_extractor(frame)  # (batch, num_joints, feature_dim)
            frame_features.append(features)
        
        # Stack frame features: (batch, num_frames, num_joints, feature_dim)
        frame_features = torch.stack(frame_features, dim=1)
        
        # Apply spatial attention to each frame
        spatial_features = []
        spatial_attn_maps = []
        for t in range(num_frames):
            frame_feat = frame_features[:, t, :, :]  # (batch, num_joints, feature_dim)
            frame_feat = self.spatial_proj(frame_feat)  # Project to spatial_dim
            attended_feat, attn_map = self.spatial_attention(frame_feat)
            spatial_features.append(attended_feat)
            spatial_attn_maps.append(attn_map)
        
        # Stack: (batch, num_frames, num_joints, spatial_dim)
        spatial_features = torch.stack(spatial_features, dim=1)
        
        # Reduce joints to single frame representation
        # (batch, num_frames, num_joints * spatial_dim)
        spatial_features = spatial_features.reshape(batch_size, num_frames, -1)
        temporal_features = self.temporal_reduce(spatial_features)  # (batch, num_frames, temporal_dim)
        
        # Apply temporal attention
        temporal_features, temporal_attn_map = self.temporal_attention(temporal_features)
        
        # Project to transformer dimension
        transformer_input = self.transformer_proj(temporal_features)  # (batch, num_frames, transformer_dim)
        
        # Add positional encoding
        transformer_input = self.positional_encoding(transformer_input)
        
        # Pass through transformer encoder
        transformer_output = self.transformer(transformer_input)  # (batch, num_frames, transformer_dim)
        
        # Hierarchical aggregation
        embedding = self.aggregation(transformer_output)  # (batch, embedding_dim)
        
        # Batch normalization
        embedding = self.bn_final(embedding)
        
        # L2 normalization
        embedding = F.normalize(embedding, p=2, dim=1)
        
        # Collect attention maps for visualization
        attention_maps = {
            'spatial': spatial_attn_maps,  # List of (batch, num_heads, num_joints, num_joints)
            'temporal': temporal_attn_map   # (batch, num_heads, num_frames, num_frames)
        }
        
        return embedding, attention_maps

# 8. SIAMESE NETWORK
class SiameseGaitTransformer(nn.Module):
    """
    Siamese Network wrapper with shared GaitEncoder
    For metric learning and person identification
    """
    
    def __init__(self, config):
        super(SiameseGaitTransformer, self).__init__()
        
        # Shared encoder for both inputs
        self.encoder = GaitEncoder(config)
        
    def forward_once(self, x):
        #Forward pass for one gait sequence
        embedding, attention_maps = self.encoder(x)
        return embedding, attention_maps
    
    def forward(self, x1, x2):
        """
        Forward pass for both gait sequences
        Args:
            x1: (batch, num_frames, num_joints, joint_dim) - First gait sequence
            x2: (batch, num_frames, num_joints, joint_dim) - Second gait sequence
        Returns:
            embedding1: (batch, embedding_dim) - First embedding
            embedding2: (batch, embedding_dim) - Second embedding
            attention_maps1: dict - Attention maps for first sequence
            attention_maps2: dict - Attention maps for second sequence
        """
        embedding1, attention_maps1 = self.forward_once(x1)
        embedding2, attention_maps2 = self.forward_once(x2)
        
        return embedding1, embedding2, attention_maps1, attention_maps2
    
    def compute_similarity(self, embedding1, embedding2, metric='cosine'):
        """
        Compute similarity between two embeddings
        Args:
            embedding1, embedding2: (batch, embedding_dim)
            metric: 'cosine' or 'euclidean'
        Returns:
            similarity: (batch,) - Similarity scores
        """
        if metric == 'cosine':
            # Cosine similarity (embeddings are already L2 normalized)
            similarity = F.cosine_similarity(embedding1, embedding2, dim=1)
        elif metric == 'euclidean':
            # Euclidean distance (convert to similarity)
            distance = F.pairwise_distance(embedding1, embedding2)
            similarity = 1 / (1 + distance)
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        return similarity
    
    def compute_distance(self, embedding1, embedding2):
        """Compute Euclidean distance between embeddings"""
        return F.pairwise_distance(embedding1, embedding2)


# 9. LOSS FUNCTIONS
class ContrastiveLoss(nn.Module):
    """
    Contrastive Loss for Siamese Networks
    Pulls similar pairs together, pushes dissimilar pairs apart
    """
    
    def __init__(self, margin=0.5):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        
    def forward(self, embedding1, embedding2, label):
        """
        Args:
            embedding1, embedding2: (batch, embedding_dim)
            label: (batch,) - 1 for same person, 0 for different person
        Returns:
            loss: Scalar contrastive loss
        """
        # Euclidean distance
        distance = F.pairwise_distance(embedding1, embedding2)
        
        # Contrastive loss
        loss_similar = label * torch.pow(distance, 2)
        loss_dissimilar = (1 - label) * torch.pow(torch.clamp(self.margin - distance, min=0.0), 2)
        
        loss = torch.mean(loss_similar + loss_dissimilar)
        
        return loss


class TripletLoss(nn.Module):
    """
    Triplet Loss for Siamese Networks
    Ensures anchor is closer to positive than to negative by margin
    """
    
    def __init__(self, margin=0.5, mining='hard'):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.mining = mining  # 'random', 'hard', 'semi-hard'
        
    def forward(self, anchor, positive, negative):
        """
        Args:
            anchor: (batch, embedding_dim) - Anchor embeddings
            positive: (batch, embedding_dim) - Positive embeddings
            negative: (batch, embedding_dim) - Negative embeddings
        Returns:
            loss: Scalar triplet loss
        """
        # Compute distances
        distance_positive = F.pairwise_distance(anchor, positive)
        distance_negative = F.pairwise_distance(anchor, negative)
        
        # Triplet loss
        losses = F.relu(distance_positive - distance_negative + self.margin)
        
        return losses.mean()
    
    def hard_triplet_loss(self, embeddings, labels):
        """
        Hard triplet mining within batch
        Args:
            embeddings: (batch, embedding_dim)
            labels: (batch,) - Person IDs
        Returns:
            loss: Scalar triplet loss
        """
        # Compute pairwise distances
        pairwise_dist = torch.cdist(embeddings, embeddings, p=2)
        
        batch_size = embeddings.size(0)
        loss = torch.tensor(0.0, device=embeddings.device)
        num_triplets = 0
        
        for i in range(batch_size):
            # Get anchor label
            anchor_label = labels[i]
            
            # Find hardest positive (same label, furthest distance)
            positive_mask = labels == anchor_label
            positive_mask[i] = False  # Exclude anchor itself
            
            if positive_mask.sum() == 0:
                continue
            
            positive_distances = pairwise_dist[i][positive_mask]
            hardest_positive_dist = positive_distances.max()
            
            # Find hardest negative (different label, closest distance)
            negative_mask = labels != anchor_label
            
            if negative_mask.sum() == 0:
                continue
            
            negative_distances = pairwise_dist[i][negative_mask]
            hardest_negative_dist = negative_distances.min()
            
            # Compute triplet loss
            triplet_loss = F.relu(hardest_positive_dist - hardest_negative_dist + self.margin)
            loss += triplet_loss
            num_triplets += 1
        
        if num_triplets > 0:
            loss = loss / num_triplets
        
        return loss


# ============================================================================
# 10. UTILITY FUNCTIONS
# ============================================================================

def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_size(model):
    """Get model size in MB"""
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    size_mb = (param_size + buffer_size) / 1024**2
    return size_mb


def visualize_attention(attention_maps, save_path=None):
    """
    Visualize spatial and temporal attention weights
    Args:
        attention_maps: dict with 'spatial' and 'temporal' keys
        save_path: Path to save visualization (optional)
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Extract attention weights (use first sample in batch)
    spatial_attn = attention_maps['spatial'][0][0].mean(dim=0).cpu().detach().numpy()  # Average over heads
    temporal_attn = attention_maps['temporal'][0].mean(dim=0).cpu().detach().numpy()  # Average over heads
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Spatial attention
    im1 = axes[0].imshow(spatial_attn, cmap='hot', interpolation='nearest')
    axes[0].set_title('Spatial Attention (Joint Importance)')
    axes[0].set_xlabel('Joint Index')
    axes[0].set_ylabel('Joint Index')
    plt.colorbar(im1, ax=axes[0])
    
    # Temporal attention
    im2 = axes[1].imshow(temporal_attn, cmap='hot', interpolation='nearest')
    axes[1].set_title('Temporal Attention (Frame Importance)')
    axes[1].set_xlabel('Frame Index')
    axes[1].set_ylabel('Frame Index')
    plt.colorbar(im2, ax=axes[1])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


if __name__ == "__main__":
    # Test model
    from gait_config import GaitConfig
    
    print("=" * 80)
    print("TESTING SIAMESE SPATIO-TEMPORAL ATTENTION TRANSFORMER")
    print("=" * 80)
    
    config = GaitConfig()
    
    # Create model
    model = SiameseGaitTransformer(config)
    print(f"\n✓ Model created successfully!")
    print(f"  Parameters: {count_parameters(model):,}")
    print(f"  Size: {get_model_size(model):.2f} MB")
    
    # Test forward pass
    batch_size = 4
    x1 = torch.randn(batch_size, config.NUM_FRAMES, config.NUM_JOINTS, config.JOINT_DIM)
    x2 = torch.randn(batch_size, config.NUM_FRAMES, config.NUM_JOINTS, config.JOINT_DIM)
    
    print(f"\n✓ Testing forward pass...")
    print(f"  Input shape: {x1.shape}")
    
    with torch.no_grad():
        emb1, emb2, attn1, attn2 = model(x1, x2)
    
    print(f"  Output embedding shape: {emb1.shape}")
    print(f"  Spatial attention maps: {len(attn1['spatial'])} frames")
    print(f"  Temporal attention shape: {attn1['temporal'].shape}")
    
    # Test similarity computation
    similarity = model.compute_similarity(emb1, emb2)
    distance = model.compute_distance(emb1, emb2)
    
    print(f"\n✓ Similarity computation:")
    print(f"  Cosine similarity range: [{similarity.min():.4f}, {similarity.max():.4f}]")
    print(f"  Euclidean distance range: [{distance.min():.4f}, {distance.max():.4f}]")
    
    print("\n" + "=" * 80)
    print("ALL TESTS PASSED! ✓")
    print("=" * 80)
