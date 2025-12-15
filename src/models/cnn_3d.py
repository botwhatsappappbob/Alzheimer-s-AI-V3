"""
3D Convolutional Neural Networks for Medical Imaging
Implements various 3D CNN architectures for MRI and PET analysis
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import numpy as np
from einops import rearrange
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BasicBlock3D(nn.Module):
    """Basic 3D residual block"""
    expansion = 1
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1,
                 downsample: Optional[nn.Module] = None):
        super().__init__()
        
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, 
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        
        self.downsample = downsample
        self.stride = stride
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
            
        out += identity
        out = self.relu(out)
        
        return out

class Bottleneck3D(nn.Module):
    """Bottleneck 3D residual block"""
    expansion = 4
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1,
                 downsample: Optional[nn.Module] = None):
        super().__init__()
        
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        
        self.conv3 = nn.Conv3d(out_channels, out_channels * self.expansion, 
                              kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(out_channels * self.expansion)
        
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
            
        out += identity
        out = self.relu(out)
        
        return out

class ResNet3D(nn.Module):
    """3D ResNet architecture for medical imaging"""
    
    def __init__(self, block: nn.Module, layers: List[int], in_channels: int = 1,
                 num_classes: int = 1000, zero_init_residual: bool = False):
        super().__init__()
        
        self.in_channels = 64
        
        # Initial convolution
        self.conv1 = nn.Conv3d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        
        # ResNet layers
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        # Global average pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        
        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
        # Zero-initialize the last BN in each residual branch
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck3D):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock3D):
                    nn.init.constant_(m.bn2.weight, 0)
                    
    def _make_layer(self, block: nn.Module, out_channels: int, num_blocks: int,
                   stride: int = 1) -> nn.Sequential:
        """Make a layer of blocks"""
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.in_channels, out_channels * block.expansion,
                         kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels * block.expansion),
            )
            
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        
        for _ in range(1, num_blocks):
            layers.append(block(self.in_channels, out_channels))
            
        return nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x

def resnet18_3d(in_channels: int = 1, num_classes: int = 1000) -> ResNet3D:
    """3D ResNet-18"""
    return ResNet3D(BasicBlock3D, [2, 2, 2, 2], in_channels, num_classes)

def resnet34_3d(in_channels: int = 1, num_classes: int = 1000) -> ResNet3D:
    """3D ResNet-34"""
    return ResNet3D(BasicBlock3D, [3, 4, 6, 3], in_channels, num_classes)

def resnet50_3d(in_channels: int = 1, num_classes: int = 1000) -> ResNet3D:
    """3D ResNet-50"""
    return ResNet3D(Bottleneck3D, [3, 4, 6, 3], in_channels, num_classes)

def resnet101_3d(in_channels: int = 1, num_classes: int = 1000) -> ResNet3D:
    """3D ResNet-101"""
    return ResNet3D(Bottleneck3D, [3, 4, 23, 3], in_channels, num_classes)

class DenseBlock3D(nn.Module):
    """3D DenseNet block"""
    
    def __init__(self, in_channels: int, growth_rate: int, num_layers: int):
        super().__init__()
        
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer = self._make_dense_layer(in_channels + i * growth_rate, growth_rate)
            self.layers.append(layer)
            
    def _make_dense_layer(self, in_channels: int, growth_rate: int) -> nn.Sequential:
        """Make a dense layer"""
        return nn.Sequential(
            nn.BatchNorm3d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels, 4 * growth_rate, kernel_size=1, bias=False),
            nn.BatchNorm3d(4 * growth_rate),
            nn.ReLU(inplace=True),
            nn.Conv3d(4 * growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = [x]
        for layer in self.layers:
            new_features = layer(torch.cat(features, 1))
            features.append(new_features)
        return torch.cat(features, 1)

class DenseNet3D(nn.Module):
    """3D DenseNet architecture"""
    
    def __init__(self, growth_rate: int = 32, block_config: List[int] = [6, 12, 24, 16],
                 num_init_features: int = 64, in_channels: int = 1, num_classes: int = 1000):
        super().__init__()
        
        # Initial convolution
        self.features = nn.Sequential(
            nn.Conv3d(in_channels, num_init_features, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm3d(num_init_features),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        )
        
        # Dense blocks
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = DenseBlock3D(num_features, growth_rate, num_layers)
            self.features.add_module(f'denseblock{i+1}', block)
            num_features += num_layers * growth_rate
            
            if i != len(block_config) - 1:
                trans = self._make_transition(num_features, num_features // 2)
                self.features.add_module(f'transition{i+1}', trans)
                num_features = num_features // 2
                
        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm3d(num_features))
        
        # Classifier
        self.classifier = nn.Linear(num_features, num_classes)
        
        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)
                
    def _make_transition(self, in_channels: int, out_channels: int) -> nn.Sequential:
        """Make transition layer"""
        return nn.Sequential(
            nn.BatchNorm3d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.AvgPool3d(kernel_size=2, stride=2)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool3d(out, (1, 1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out

def densenet121_3d(in_channels: int = 1, num_classes: int = 1000) -> DenseNet3D:
    """3D DenseNet-121"""
    return DenseNet3D(growth_rate=32, block_config=[6, 12, 24, 16],
                     num_init_features=64, in_channels=in_channels, num_classes=num_classes)

def densenet169_3d(in_channels: int = 1, num_classes: int = 1000) -> DenseNet3D:
    """3D DenseNet-169"""
    return DenseNet3D(growth_rate=32, block_config=[6, 12, 32, 32],
                     num_init_features=64, in_channels=in_channels, num_classes=num_classes)

class Attention3D(nn.Module):
    """3D Attention mechanism for medical imaging"""
    
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
    """Custom 3D CNN for medical imaging with attention"""
    
    def __init__(self, in_channels: int = 1, base_filters: int = 32, 
                 use_attention: bool = True):
        super().__init__()
        
        self.use_attention = use_attention
        
        # Encoder
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
        if self.use_attention:
            self.attention = Attention3D(base_filters * 8)
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        
        # Feature dimension
        self.feature_dim = base_filters * 8
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        
        # Apply attention if enabled
        if self.use_attention:
            x = self.attention(x)
        
        # Global pooling
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        
        return x

# Model configurations
MODEL_CONFIGS = {
    'resnet18_3d': {
        'class': resnet18_3d,
        'params': {'in_channels': 1, 'num_classes': 3},
        'description': '3D ResNet-18 for medical imaging'
    },
    'resnet34_3d': {
        'class': resnet34_3d,
        'params': {'in_channels': 1, 'num_classes': 3},
        'description': '3D ResNet-34 for medical imaging'
    },
    'resnet50_3d': {
        'class': resnet50_3d,
        'params': {'in_channels': 1, 'num_classes': 3},
        'description': '3D ResNet-50 for medical imaging'
    },
    'densenet121_3d': {
        'class': densenet121_3d,
        'params': {'in_channels': 1, 'num_classes': 3},
        'description': '3D DenseNet-121 for medical imaging'
    },
    'medical_cnn_3d': {
        'class': MedicalCNN3D,
        'params': {'in_channels': 1, 'base_filters': 32, 'use_attention': True},
        'description': 'Custom 3D CNN with attention for medical imaging'
    }
}

def get_model_config(model_name: str) -> Dict:
    """Get model configuration by name"""
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model: {model_name}. Available models: {list(MODEL_CONFIGS.keys())}")
    return MODEL_CONFIGS[model_name]

def create_model(model_name: str, **kwargs) -> nn.Module:
    """Create model instance"""
    config = get_model_config(model_name)
    model_class = config['class']
    model_params = {**config['params'], **kwargs}
    return model_class(**model_params)

# Example usage
if __name__ == "__main__":
    # Test different model architectures
    test_input = torch.randn(2, 1, 64, 64, 64)  # batch_size=2, channels=1, D=64, H=64, W=64
    
    models_to_test = ['resnet18_3d', 'resnet34_3d', 'medical_cnn_3d']
    
    for model_name in models_to_test:
        print(f"\nTesting {model_name}:")
        model = create_model(model_name)
        output = model(test_input)
        print(f"Input shape: {test_input.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
    print("\n3D CNN models initialized successfully")