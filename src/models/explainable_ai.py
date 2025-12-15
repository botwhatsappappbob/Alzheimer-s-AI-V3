"""
Explainable AI Components for Alzheimer's Disease Detection
Implements various XAI methods for medical AI interpretability
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Union
import logging
from pathlib import Path
import nibabel as nib
from captum.attr import GradientShap, IntegratedGradients, LayerGradCam
from captum.attr import visualization as viz
import seaborn as sns

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GradCAM3D:
    """
    3D GradCAM implementation for medical imaging
    Generates attention maps for 3D medical images
    """
    
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self._register_hooks()
        
    def _register_hooks(self):
        """Register forward and backward hooks"""
        def forward_hook(module, input, output):
            self.activations = output.detach()
            
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
            
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)
        
    def generate_cam(self, input_tensor: torch.Tensor, target_class: int = None) -> np.ndarray:
        """
        Generate CAM for input tensor
        
        Args:
            input_tensor: Input image tensor
            target_class: Target class for CAM generation
            
        Returns:
            3D CAM as numpy array
        """
        self.model.eval()
        
        # Forward pass
        output = self.model(input_tensor)
        
        # Get target class if not specified
        if target_class is None:
            target_class = output.argmax().item()
            
        # Backward pass
        self.model.zero_grad()
        output[0, target_class].backward()
        
        # Calculate weights (global average pooling of gradients)
        weights = torch.mean(self.gradients, dim=(2, 3, 4), keepdim=True)
        
        # Calculate CAM
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = F.relu(cam)
        
        # Resize to input size
        cam = F.interpolate(cam, size=input_tensor.shape[2:], mode='trilinear', align_corners=False)
        
        # Normalize
        cam = cam - cam.min()
        cam = cam / cam.max()
        
        return cam.squeeze().cpu().numpy()

class SHAPMedicalExplainer:
    """
    SHAP (SHapley Additive exPlanations) for medical AI
    Explains both imaging and clinical features
    """
    
    def __init__(self, model: nn.Module, background_data: torch.Tensor):
        self.model = model
        self.background_data = background_data
        self.explainer = None
        
    def explain_image(self, input_image: torch.Tensor, target_class: int = None,
                     n_samples: int = 50) -> Dict:
        """
        Explain image prediction using SHAP
        
        Args:
            input_image: Input image tensor
            target_class: Target class for explanation
            n_samples: Number of samples for SHAP calculation
            
        Returns:
            Dictionary with SHAP values and visualization
        """
        try:
            # Use GradientSHAP for images
            gradient_shap = GradientShap(self.model)
            
            # Generate random baseline
            baseline = torch.randn_like(input_image) * 0.001
            
            # Calculate SHAP values
            attributions, delta = gradient_shap.attribute(
                input_image,
                baselines=baseline,
                target=target_class,
                n_samples=n_samples,
                return_convergence_delta=True
            )
            
            # Convert to numpy for visualization
            shap_values = attributions.squeeze().cpu().numpy()
            
            return {
                'shap_values': shap_values,
                'convergence_delta': delta.item(),
                'visualization': self._create_shap_visualization(shap_values, input_image)
            }
            
        except Exception as e:
            logger.error(f"SHAP explanation failed: {e}")
            return {'error': str(e)}
            
    def explain_clinical(self, clinical_data: np.ndarray, 
                        feature_names: List[str], target_class: int = None) -> Dict:
        """
        Explain clinical features using SHAP
        
        Args:
            clinical_data: Clinical feature array
            feature_names: Names of clinical features
            target_class: Target class for explanation
            
        Returns:
            Dictionary with SHAP values and feature importance
        """
        try:
            # Use KernelSHAP for tabular data
            from captum.attr import KernelShap
            
            kernel_shap = KernelShap(self.model)
            
            # Calculate SHAP values
            attributions = kernel_shap.attribute(
                torch.FloatTensor(clinical_data).unsqueeze(0),
                target=target_class,
                n_samples=100
            )
            
            shap_values = attributions.squeeze().cpu().numpy()
            
            # Create feature importance
            feature_importance = list(zip(feature_names, shap_values))
            feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)
            
            return {
                'shap_values': shap_values,
                'feature_importance': feature_importance,
                'visualization': self._create_feature_importance_plot(feature_importance)
            }
            
        except Exception as e:
            logger.error(f"Clinical SHAP explanation failed: {e}")
            return {'error': str(e)}
            
    def _create_shap_visualization(self, shap_values: np.ndarray, 
                                  input_image: torch.Tensor) -> str:
        """Create SHAP visualization"""
        try:
            # Create overlay visualization
            input_np = input_image.squeeze().cpu().numpy()
            
            # Take middle slice for 2D visualization
            if len(input_np.shape) == 3:
                slice_idx = input_np.shape[0] // 2
                input_slice = input_np[slice_idx]
                shap_slice = shap_values[slice_idx]
            else:
                input_slice = input_np
                shap_slice = shap_values
                
            # Create plot
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Original image
            axes[0].imshow(input_slice, cmap='gray')
            axes[0].set_title('Original Image')
            axes[0].axis('off')
            
            # SHAP values
            im = axes[1].imshow(shap_slice, cmap='RdBu_r')
            axes[1].set_title('SHAP Values')
            axes[1].axis('off')
            plt.colorbar(im, ax=axes[1])
            
            # Overlay
            axes[2].imshow(input_slice, cmap='gray')
            axes[2].imshow(shap_slice, cmap='RdBu_r', alpha=0.5)
            axes[2].set_title('SHAP Overlay')
            axes[2].axis('off')
            
            # Save to temporary file
            temp_path = Path('/tmp/shap_visualization.png')
            plt.savefig(temp_path, bbox_inches='tight', dpi=150)
            plt.close()
            
            return str(temp_path)
            
        except Exception as e:
            logger.error(f"SHAP visualization failed: {e}")
            return None
            
    def _create_feature_importance_plot(self, feature_importance: List[Tuple[str, float]]) -> str:
        """Create feature importance plot"""
        try:
            features, values = zip(*feature_importance[:10])  # Top 10 features
            
            plt.figure(figsize=(12, 6))
            colors = ['red' if v < 0 else 'blue' for v in values]
            plt.barh(range(len(features)), values, color=colors)
            plt.yticks(range(len(features)), features)
            plt.xlabel('SHAP Value')
            plt.title('Top 10 Feature Importance (SHAP)')
            plt.tight_layout()
            
            # Save to temporary file
            temp_path = Path('/tmp/feature_importance.png')
            plt.savefig(temp_path, bbox_inches='tight', dpi=150)
            plt.close()
            
            return str(temp_path)
            
        except Exception as e:
            logger.error(f"Feature importance plot failed: {e}")
            return None

class AttentionVisualizer:
    """
    Visualize attention mechanisms in the model
    """
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.attention_weights = {}
        
    def register_attention_hooks(self, attention_layers: List[nn.Module]):
        """Register hooks to capture attention weights"""
        for i, layer in enumerate(attention_layers):
            def hook_fn(module, input, output):
                # Capture attention weights (implementation depends on attention mechanism)
                if hasattr(module, 'attention_weights'):
                    self.attention_weights[f'layer_{i}'] = module.attention_weights.detach().cpu()
                    
            layer.register_forward_hook(hook_fn)
            
    def visualize_attention_maps(self, input_tensor: torch.Tensor) -> Dict:
        """
        Visualize attention maps
        
        Args:
            input_tensor: Input tensor to the model
            
        Returns:
            Dictionary with attention visualizations
        """
        self.model.eval()
        
        with torch.no_grad():
            output = self.model(input_tensor)
            
        visualizations = {}
        
        for layer_name, attention_weights in self.attention_weights.items():
            # Create attention visualization
            # This is a simplified implementation - actual visualization depends on attention mechanism
            attention_map = attention_weights.mean(dim=1).squeeze().numpy()
            
            # Create heatmap
            plt.figure(figsize=(10, 8))
            sns.heatmap(attention_map, cmap='viridis', center=0)
            plt.title(f'Attention Map - {layer_name}')
            plt.tight_layout()
            
            # Save to temporary file
            temp_path = Path(f'/tmp/attention_{layer_name}.png')
            plt.savefig(temp_path, bbox_inches='tight', dpi=150)
            plt.close()
            
            visualizations[layer_name] = str(temp_path)
            
        return visualizations

class IntegratedGradients3D:
    """
    3D Integrated Gradients for medical imaging
    """
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.integrated_gradients = IntegratedGradients(model)
        
    def explain_prediction(self, input_tensor: torch.Tensor, target_class: int = None,
                          n_steps: int = 50) -> Dict:
        """
        Explain prediction using integrated gradients
        
        Args:
            input_tensor: Input image tensor
            target_class: Target class for explanation
            n_steps: Number of integration steps
            
        Returns:
            Dictionary with integrated gradients explanation
        """
        try:
            # Calculate integrated gradients
            attributions = self.integrated_gradients.attribute(
                input_tensor,
                target=target_class,
                n_steps=n_steps,
                method='gausslegendre'
            )
            
            # Convert to numpy
            ig_values = attributions.squeeze().cpu().numpy()
            
            # Create visualization
            visualization = self._create_ig_visualization(ig_values, input_tensor)
            
            return {
                'integrated_gradients': ig_values,
                'visualization': visualization,
                'attribution_sum': np.sum(ig_values)
            }
            
        except Exception as e:
            logger.error(f"Integrated gradients failed: {e}")
            return {'error': str(e)}
            
    def _create_ig_visualization(self, ig_values: np.ndarray, 
                                input_tensor: torch.Tensor) -> str:
        """Create integrated gradients visualization"""
        try:
            input_np = input_tensor.squeeze().cpu().numpy()
            
            # Take middle slice
            if len(input_np.shape) == 3:
                slice_idx = input_np.shape[0] // 2
                input_slice = input_np[slice_idx]
                ig_slice = ig_values[slice_idx]
            else:
                input_slice = input_np
                ig_slice = ig_values
                
            # Create plot
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            
            # Original image
            axes[0].imshow(input_slice, cmap='gray')
            axes[0].set_title('Original Image')
            axes[0].axis('off')
            
            # Integrated gradients
            im = axes[1].imshow(ig_slice, cmap='RdBu_r')
            axes[1].set_title('Integrated Gradients')
            axes[1].axis('off')
            plt.colorbar(im, ax=axes[1])
            
            # Save to temporary file
            temp_path = Path('/tmp/integrated_gradients.png')
            plt.savefig(temp_path, bbox_inches='tight', dpi=150)
            plt.close()
            
            return str(temp_path)
            
        except Exception as e:
            logger.error(f"IG visualization failed: {e}")
            return None

class MedicalExplainer:
    """
    Main class for medical AI explanations
    Combines multiple explanation methods
    """
    
    def __init__(self, model: nn.Module, background_data: Optional[torch.Tensor] = None):
        self.model = model
        self.background_data = background_data
        self.explainers = {}
        
        # Initialize explainers
        self.explainers['gradcam'] = None  # Will be set with target layer
        self.explainers['shap'] = SHAPMedicalExplainer(model, background_data) if background_data else None
        self.explainers['integrated_gradients'] = IntegratedGradients3D(model)
        
    def set_gradcam_target(self, target_layer: nn.Module):
        """Set target layer for GradCAM"""
        self.explainers['gradcam'] = GradCAM3D(self.model, target_layer)
        
    def explain_prediction(self, input_data: Dict, target_class: int = None) -> Dict:
        """
        Generate comprehensive explanation using multiple methods
        
        Args:
            input_data: Dictionary with 'image' and/or 'clinical' data
            target_class: Target class for explanation
            
        Returns:
            Dictionary with all explanations
        """
        explanations = {
            'target_class': target_class,
            'timestamp': datetime.now().isoformat(),
            'methods': {}
        }
        
        # Image explanations
        if 'image' in input_data and self.explainers['gradcam'] is not None:
            try:
                gradcam_result = self.explainers['gradcam'].generate_cam(
                    input_data['image'], target_class
                )
                explanations['methods']['gradcam'] = {
                    'attention_map': gradcam_result,
                    'description': 'Gradient-based attention visualization'
                }
            except Exception as e:
                explanations['methods']['gradcam'] = {'error': str(e)}
                
        if 'image' in input_data:
            try:
                ig_result = self.explainers['integrated_gradients'].explain_prediction(
                    input_data['image'], target_class
                )
                explanations['methods']['integrated_gradients'] = ig_result
            except Exception as e:
                explanations['methods']['integrated_gradients'] = {'error': str(e)}
                
        # Clinical explanations
        if 'clinical' in input_data and self.explainers['shap'] is not None:
            try:
                feature_names = input_data.get('feature_names', [f'feature_{i}' for i in range(len(input_data['clinical']))])
                shap_result = self.explainers['shap'].explain_clinical(
                    input_data['clinical'], feature_names, target_class
                )
                explanations['methods']['shap'] = shap_result
            except Exception as e:
                explanations['methods']['shap'] = {'error': str(e)}
                
        return explanations
        
    def create_explanation_report(self, explanations: Dict, output_path: Path) -> Path:
        """
        Create comprehensive explanation report
        
        Args:
            explanations: Dictionary with all explanations
            output_path: Path to save the report
            
        Returns:
            Path to the generated report
        """
        try:
            # Create report
            report = {
                'explanation_summary': {
                    'target_class': explanations['target_class'],
                    'timestamp': explanations['timestamp'],
                    'methods_used': list(explanations['methods'].keys())
                },
                'detailed_explanations': explanations['methods'],
                'interpretation_guidelines': {
                    'gradcam': 'Hotter regions indicate areas the model focused on',
                    'shap': 'Positive values increase prediction, negative values decrease',
                    'integrated_gradients': 'Shows contribution of each pixel to the prediction'
                }
            }
            
            # Save report
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
                
            return output_path
            
        except Exception as e:
            logger.error(f"Report creation failed: {e}")
            return None

# Example usage
if __name__ == "__main__":
    # Create a simple model for testing
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv3d(1, 16, 3, padding=1)
            self.conv2 = nn.Conv3d(16, 32, 3, padding=1)
            self.pool = nn.AdaptiveAvgPool3d(1)
            self.fc = nn.Linear(32, 3)
            
        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            return self.fc(x)
    
    model = SimpleModel()
    
    # Initialize explainer
    explainer = MedicalExplainer(model)
    explainer.set_gradcam_target(model.conv2)
    
    # Test with dummy data
    test_image = torch.randn(1, 1, 32, 32, 32)
    
    explanations = explainer.explain_prediction({
        'image': test_image
    }, target_class=1)
    
    print("Explanation generated successfully!")
    print(f"Methods used: {list(explanations['methods'].keys())}")
    
    # Create report
    report_path = Path("/tmp/explanation_report.json")
    explainer.create_explanation_report(explanations, report_path)
    print(f"Report saved to: {report_path}")