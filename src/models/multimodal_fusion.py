"""
Multimodal Deep Learning Architecture for Alzheimer's Disease Detection
Implements attention-based fusion of MRI, PET, and clinical data
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import numpy as np
from einops import rearrange
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SelfAttention3D(nn.Module):
    """3D Self-Attention mechanism for medical imaging"""
    
    def __init__(self, in_channels: int, reduction: int = 8):
        super().__init__()
        self.in_channels = in_channels
        self.reduction = reduction
        
        self.query_conv = nn.Conv3d(in_channels, in_channels // reduction, 1)
        self.key_conv = nn.Conv3d(in_channels, in_channels // reduction, 1)
        self.value_conv = nn.Conv3d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, depth, height, width = x.size()
        
        # Generate query, key, value
        query = self.query_conv(x).view(batch_size, -1, depth * height * width)
        key = self.key_conv(x).view(batch_size, -1, depth * height * width)
        value = self.value_conv(x).view(batch_size, -1, depth * height * width)
        
        # Compute attention
        attention = torch.bmm(query.transpose(1, 2), key)
        attention = F.softmax(attention, dim=-1)
        
        # Apply attention to value
        out = torch.bmm(value, attention.transpose(1, 2))
        out = out.view(batch_size, channels, depth, height, width)
        
        # Residual connection
        out = self.gamma * out + x
        return out

class MedicalCNN3D(nn.Module):
    """3D CNN backbone for medical imaging analysis"""
    
    def __init__(self, in_channels: int = 1, base_filters: int = 32):
        super().__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, base_filters, kernel_size=3, padding=1),
            nn.BatchNorm3d(base_filters),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2, 2)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv3d(base_filters, base_filters * 2, kernel_size=3, padding=1),
            nn.BatchNorm3d(base_filters * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2, 2)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv3d(base_filters * 2, base_filters * 4, kernel_size=3, padding=1),
            nn.BatchNorm3d(base_filters * 4),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2, 2)
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv3d(base_filters * 4, base_filters * 8, kernel_size=3, padding=1),
            nn.BatchNorm3d(base_filters * 8),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2, 2)
        )
        
        # Attention mechanism
        self.attention = SelfAttention3D(base_filters * 8)
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        
        # Feature dimension
        self.feature_dim = base_filters * 8
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        
        # Apply attention
        x = self.attention(x)
        
        # Global pooling
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        
        return x

class ClinicalFeatureEncoder(nn.Module):
    """Encoder for clinical and cognitive data"""
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = [128, 64]):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3)
            ])
            prev_dim = hidden_dim
        
        self.encoder = nn.Sequential(*layers)
        self.feature_dim = hidden_dims[-1]
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

class CrossModalAttention(nn.Module):
    """Cross-modal attention for feature fusion"""
    
    def __init__(self, feature_dim: int, num_modalities: int):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.num_modalities = num_modalities
        
        # Query, Key, Value projections for each modality
        self.query_proj = nn.ModuleList([
            nn.Linear(feature_dim, feature_dim) for _ in range(num_modalities)
        ])
        
        self.key_proj = nn.ModuleList([
            nn.Linear(feature_dim, feature_dim) for _ in range(num_modalities)
        ])
        
        self.value_proj = nn.ModuleList([
            nn.Linear(feature_dim, feature_dim) for _ in range(num_modalities)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(feature_dim * num_modalities, feature_dim)
        
        # Learnable temperature parameter
        self.temperature = nn.Parameter(torch.ones(1) * np.sqrt(feature_dim))
        
    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        batch_size = features[0].size(0)
        
        # Project features to query, key, value
        queries = [proj(feat) for proj, feat in zip(self.query_proj, features)]
        keys = [proj(feat) for proj, feat in zip(self.key_proj, features)]
        values = [proj(feat) for proj, feat in zip(self.value_proj, features)]
        
        # Stack for batch processing
        queries = torch.stack(queries, dim=1)  # (batch, num_modalities, feature_dim)
        keys = torch.stack(keys, dim=1)
        values = torch.stack(values, dim=1)
        
        # Compute cross-modal attention
        attention_scores = torch.bmm(queries, keys.transpose(1, 2)) / self.temperature
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # Apply attention to values
        attended_features = torch.bmm(attention_weights, values)
        
        # Concatenate and project
        attended_features = attended_features.view(batch_size, -1)
        output = self.output_proj(attended_features)
        
        return output

class MultimodalAlzheimerNet(nn.Module):
    """
    Main multimodal architecture for Alzheimer's disease detection
    Integrates MRI, PET, and clinical data with attention mechanisms
    """
    
    def __init__(self, 
                 mri_channels: int = 1,
                 pet_channels: int = 1,
                 clinical_dim: int = 20,
                 num_classes: int = 3,  # CN, MCI, AD
                 base_filters: int = 32,
                 feature_dim: int = 256):
        super().__init__()
        
        self.num_classes = num_classes
        
        # Imaging encoders
        self.mri_encoder = MedicalCNN3D(mri_channels, base_filters)
        self.pet_encoder = MedicalCNN3D(pet_channels, base_filters)
        
        # Clinical encoder
        self.clinical_encoder = ClinicalFeatureEncoder(clinical_dim, [128, 64])
        
        # Feature projection layers
        self.mri_proj = nn.Linear(self.mri_encoder.feature_dim, feature_dim)
        self.pet_proj = nn.Linear(self.pet_encoder.feature_dim, feature_dim)
        self.clinical_proj = nn.Linear(self.clinical_encoder.feature_dim, feature_dim)
        
        # Cross-modal attention fusion
        self.cross_attention = CrossModalAttention(feature_dim, num_modalities=3)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.BatchNorm1d(feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(feature_dim // 2, num_classes)
        )
        
        # Uncertainty estimation
        self.uncertainty_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, mri: torch.Tensor, pet: torch.Tensor, 
                clinical: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass of multimodal network
        
        Returns:
            Dictionary with predictions, features, and uncertainty
        """
        # Encode each modality
        mri_features = self.mri_encoder(mri)
        pet_features = self.pet_encoder(pet)
        clinical_features = self.clinical_encoder(clinical)
        
        # Project to common feature space
        mri_proj = self.mri_proj(mri_features)
        pet_proj = self.pet_proj(pet_features)
        clinical_proj = self.clinical_proj(clinical_features)
        
        # Cross-modal attention fusion
        features = [mri_proj, pet_proj, clinical_proj]
        fused_features = self.cross_attention(features)
        
        # Classification
        logits = self.classifier(fused_features)
        
        # Uncertainty estimation
        uncertainty = self.uncertainty_head(fused_features)
        
        return {
            'logits': logits,
            'probabilities': F.softmax(logits, dim=1),
            'features': fused_features,
            'mri_features': mri_features,
            'pet_features': pet_features,
            'clinical_features': clinical_features,
            'uncertainty': uncertainty
        }

class AlzheimerDetectionModel:
    """
    High-level interface for Alzheimer's disease detection model
    Handles training, inference, and explanation
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize model
        self.model = MultimodalAlzheimerNet(
            mri_channels=config.get('mri_channels', 1),
            pet_channels=config.get('pet_channels', 1),
            clinical_dim=config.get('clinical_dim', 20),
            num_classes=config.get('num_classes', 3),
            base_filters=config.get('base_filters', 32),
            feature_dim=config.get('feature_dim', 256)
        ).to(self.device)
        
        # Initialize optimizer and loss
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.get('learning_rate', 1e-3),
            weight_decay=config.get('weight_decay', 1e-4)
        )
        
        self.criterion = nn.CrossEntropyLoss(
            weight=config.get('class_weights', None)
        )
        
        logger.info(f"Model initialized on {self.device}")
        
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step"""
        self.model.train()
        
        # Move data to device
        mri = batch['mri'].to(self.device)
        pet = batch['pet'].to(self.device)
        clinical = batch['clinical'].to(self.device)
        labels = batch['labels'].to(self.device)
        
        # Forward pass
        outputs = self.model(mri, pet, clinical)
        loss = self.criterion(outputs['logits'], labels)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        # Compute metrics
        predictions = torch.argmax(outputs['probabilities'], dim=1)
        accuracy = (predictions == labels).float().mean()
        
        return {
            'loss': loss.item(),
            'accuracy': accuracy.item(),
            'uncertainty': outputs['uncertainty'].mean().item()
        }
    
    def evaluate(self, dataloader) -> Dict[str, float]:
        """Evaluate model on validation/test set"""
        self.model.eval()
        
        total_loss = 0
        correct = 0
        total = 0
        uncertainties = []
        
        with torch.no_grad():
            for batch in dataloader:
                mri = batch['mri'].to(self.device)
                pet = batch['pet'].to(self.device)
                clinical = batch['clinical'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(mri, pet, clinical)
                loss = self.criterion(outputs['logits'], labels)
                
                total_loss += loss.item()
                
                predictions = torch.argmax(outputs['probabilities'], dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
                
                uncertainties.extend(outputs['uncertainty'].cpu().numpy())
        
        return {
            'loss': total_loss / len(dataloader),
            'accuracy': correct / total,
            'uncertainty': np.mean(uncertainties)
        }
    
    def predict_with_explanation(self, mri: np.ndarray, pet: np.ndarray, 
                                clinical: np.ndarray) -> Dict:
        """
        Make prediction with model explanation
        """
        self.model.eval()
        
        # Convert to tensors
        mri_tensor = torch.FloatTensor(mri).unsqueeze(0).to(self.device)
        pet_tensor = torch.FloatTensor(pet).unsqueeze(0).to(self.device)
        clinical_tensor = torch.FloatTensor(clinical).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(mri_tensor, pet_tensor, clinical_tensor)
        
        return {
            'prediction': torch.argmax(outputs['probabilities'], dim=1).item(),
            'probabilities': outputs['probabilities'].cpu().numpy()[0],
            'uncertainty': outputs['uncertainty'].cpu().numpy()[0],
            'features': outputs['features'].cpu().numpy()[0],
            'mri_features': outputs['mri_features'].cpu().numpy()[0],
            'pet_features': outputs['pet_features'].cpu().numpy()[0],
            'clinical_features': outputs['clinical_features'].cpu().numpy()[0]
        }
    
    def save_checkpoint(self, filepath: str, epoch: int, metrics: Dict[str, float]):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'config': self.config
        }
        torch.save(checkpoint, filepath)
        logger.info(f"Checkpoint saved: {filepath}")
    
    def load_checkpoint(self, filepath: str) -> Dict:
        """Load model checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logger.info(f"Checkpoint loaded: {filepath}")
        return checkpoint

# Example usage
if __name__ == "__main__":
    # Model configuration
    config = {
        'mri_channels': 1,
        'pet_channels': 1,
        'clinical_dim': 20,
        'num_classes': 3,
        'base_filters': 32,
        'feature_dim': 256,
        'learning_rate': 1e-3,
        'weight_decay': 1e-4
    }
    
    # Initialize model
    model = AlzheimerDetectionModel(config)
    
    # Test with dummy data
    dummy_mri = torch.randn(1, 1, 256, 256, 256)
    dummy_pet = torch.randn(1, 1, 256, 256, 256)
    dummy_clinical = torch.randn(1, 20)
    dummy_labels = torch.randint(0, 3, (1,))
    
    batch = {
        'mri': dummy_mri,
        'pet': dummy_pet,
        'clinical': dummy_clinical,
        'labels': dummy_labels
    }
    
    # Test forward pass
    metrics = model.train_step(batch)
    print(f"Training metrics: {metrics}")
    
    print("Multimodal model initialized successfully")