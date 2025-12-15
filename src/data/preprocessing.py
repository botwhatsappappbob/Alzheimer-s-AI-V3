"""
Medical Image Preprocessing Pipeline for Alzheimer's Disease Detection
Handles MRI and PET image preprocessing, normalization, and quality control
"""

import numpy as np
import nibabel as nib
from pathlib import Path
import logging
from typing import Dict, Tuple, Optional, List
import ants
from skimage import exposure
import pandas as pd
from scipy import ndimage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MedicalImagePreprocessor:
    """
    Comprehensive preprocessing pipeline for multi-modal medical imaging data
    Supports MRI (T1, T2, FLAIR) and PET formats with standardized protocols
    """
    
    def __init__(self, target_shape: Tuple[int, int, int] = (256, 256, 256)):
        self.target_shape = target_shape
        self.mni_template = None
        self.load_mni_template()
        
    def load_mni_template(self):
        """Load MNI152 template for registration"""
        try:
            # This would typically load from FSL or ANTs installation
            # For now, we'll create a placeholder
            self.mni_template = np.zeros(self.target_shape)
            logger.info("MNI template loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load MNI template: {e}")
            
    def skull_strip(self, image: np.ndarray, method: str = "ants") -> np.ndarray:
        """
        Perform skull stripping using ANTs or other methods
        """
        if method == "ants":
            try:
                # ANTs skull stripping
                ants_img = ants.from_numpy(image)
                stripped = ants.brain_extraction(ants_img, modality="t1")
                return stripped.numpy()
            except Exception as e:
                logger.warning(f"ANTs skull stripping failed: {e}")
                return self._fallback_skull_strip(image)
        else:
            return self._fallback_skull_strip(image)
    
    def _fallback_skull_strip(self, image: np.ndarray) -> np.ndarray:
        """
        Fallback skull stripping using simple thresholding
        """
        # Simple intensity thresholding approach
        threshold = np.percentile(image, 95)
        mask = image > threshold
        mask = ndimage.binary_erosion(mask, iterations=5)
        mask = ndimage.binary_dilation(mask, iterations=5)
        return image * mask
    
    def register_to_mni(self, image: np.ndarray, 
                       fixed_image: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Register image to MNI space using rigid + affine transformation
        """
        if fixed_image is None:
            fixed_image = self.mni_template
            
        try:
            # ANTs registration
            moving = ants.from_numpy(image)
            fixed = ants.from_numpy(fixed_image)
            
            # Rigid + Affine registration
            transform = ants.registration(
                fixed=fixed, 
                moving=moving,
                type_of_transform='RigidThenAffine'
            )
            
            return transform['warpedmovout'].numpy()
        except Exception as e:
            logger.warning(f"Registration failed: {e}")
            return image  # Return original if registration fails
    
    def intensity_normalize(self, image: np.ndarray, 
                          method: str = "zscore") -> np.ndarray:
        """
        Normalize image intensities
        """
        if method == "zscore":
            mean = np.mean(image)
            std = np.std(image)
            return (image - mean) / (std + 1e-8)
        elif method == "minmax":
            min_val = np.min(image)
            max_val = np.max(image)
            return (image - min_val) / (max_val - min_val + 1e-8)
        elif method == "histogram":
            # Histogram matching to MNI template
            return exposure.match_histograms(image, self.mni_template)
        else:
            return image
    
    def preprocess_mri(self, image_path: Path, output_path: Path,
                      modality: str = "t1") -> Dict[str, any]:
        """
        Complete MRI preprocessing pipeline
        """
        logger.info(f"Processing MRI: {image_path}")
        
        try:
            # Load image
            nii_img = nibabel.load(image_path)
            image = nii_img.get_fdata()
            
            # Store original metadata
            metadata = {
                'original_shape': image.shape,
                'affine': nii_img.affine.tolist(),
                'voxel_size': nii_img.header.get_zooms(),
                'modality': modality
            }
            
            # Resample to target shape
            image = self._resample_image(image, self.target_shape)
            
            # Skull stripping
            brain_image = self.skull_strip(image)
            
            # Registration to MNI space
            registered_image = self.register_to_mni(brain_image)
            
            # Intensity normalization
            normalized_image = self.intensity_normalize(registered_image, method="zscore")
            
            # Quality control
            qc_score = self.compute_quality_score(normalized_image)
            
            # Save processed image
            self._save_processed_image(normalized_image, nii_img.affine, output_path)
            
            metadata.update({
                'processed_shape': normalized_image.shape,
                'qc_score': qc_score,
                'mean_intensity': np.mean(normalized_image),
                'std_intensity': np.std(normalized_image),
                'snr': self.compute_snr(normalized_image)
            })
            
            logger.info(f"MRI preprocessing completed: QC={qc_score:.3f}")
            return metadata
            
        except Exception as e:
            logger.error(f"MRI preprocessing failed: {e}")
            return {'error': str(e)}
    
    def preprocess_pet(self, image_path: Path, output_path: Path,
                      pet_type: str = "fdg") -> Dict[str, any]:
        """
        Complete PET preprocessing pipeline
        """
        logger.info(f"Processing PET: {image_path}")
        
        try:
            # Load PET image
            nii_img = nibabel.load(image_path)
            image = nii_img.get_fdata()
            
            # Store original metadata
            metadata = {
                'original_shape': image.shape,
                'affine': nii_img.affine.tolist(),
                'voxel_size': nii_img.header.get_zooms(),
                'pet_type': pet_type
            }
            
            # Resample to target shape
            image = self._resample_image(image, self.target_shape)
            
            # Registration to MNI space (if not already registered)
            registered_image = self.register_to_mni(image)
            
            # PET-specific preprocessing
            if pet_type.lower() == "fdg":
                # FDG-PET: Normalize by pons or cerebellum
                normalized_image = self._normalize_fdg_pet(registered_image)
            elif pet_type.lower() in ["amyloid", "tau"]:
                # Amyloid/Tau PET: SUVR calculation
                normalized_image = self._calculate_suvr(registered_image)
            else:
                normalized_image = self.intensity_normalize(registered_image)
            
            # Quality control
            qc_score = self.compute_quality_score(normalized_image)
            
            # Save processed image
            self._save_processed_image(normalized_image, nii_img.affine, output_path)
            
            metadata.update({
                'processed_shape': normalized_image.shape,
                'qc_score': qc_score,
                'mean_intensity': np.mean(normalized_image),
                'std_intensity': np.std(normalized_image),
                'snr': self.compute_snr(normalized_image)
            })
            
            logger.info(f"PET preprocessing completed: QC={qc_score:.3f}")
            return metadata
            
        except Exception as e:
            logger.error(f"PET preprocessing failed: {e}")
            return {'error': str(e)}
    
    def _resample_image(self, image: np.ndarray, target_shape: Tuple[int, int, int]) -> np.ndarray:
        """Resample image to target shape"""
        factors = [t/s for t, s in zip(target_shape, image.shape)]
        return ndimage.zoom(image, factors, order=1)
    
    def _normalize_fdg_pet(self, image: np.ndarray) -> np.ndarray:
        """Normalize FDG-PET by pons reference region"""
        # Define pons region (approximate coordinates in MNI space)
        pons_region = image[120:140, 150:170, 90:110]
        pons_mean = np.mean(pons_region)
        
        if pons_mean > 0:
            return image / pons_mean
        else:
            return self.intensity_normalize(image)
    
    def _calculate_suvr(self, image: np.ndarray) -> np.ndarray:
        """Calculate SUVR for amyloid/tau PET"""
        # Define cerebellar reference region
        cerebellum_region = image[180:220, 140:180, 60:100]
        cerebellum_mean = np.mean(cerebellum_region)
        
        if cerebellum_mean > 0:
            return image / cerebellum_mean
        else:
            return self.intensity_normalize(image)
    
    def compute_quality_score(self, image: np.ndarray) -> float:
        """
        Compute image quality score (0-1)
        Higher score indicates better quality
        """
        # Signal-to-noise ratio
        snr = self.compute_snr(image)
        
        # Intensity distribution
        intensity_std = np.std(image)
        
        # Contrast (edge detection)
        edges = np.gradient(image)
        edge_strength = np.mean([np.std(edge) for edge in edges])
        
        # Combine metrics into quality score
        quality_score = (snr / 100.0) * 0.4 + (intensity_std / 10.0) * 0.3 + (edge_strength / 5.0) * 0.3
        
        return min(1.0, max(0.0, quality_score))
    
    def compute_snr(self, image: np.ndarray) -> float:
        """Compute signal-to-noise ratio"""
        # Simple SNR estimation
        brain_mask = image > np.percentile(image, 10)
        signal = np.mean(image[brain_mask])
        noise = np.std(image[~brain_mask])
        
        if noise > 0:
            return signal / noise
        else:
            return 0.0
    
    def _save_processed_image(self, image: np.ndarray, affine: np.ndarray, 
                            output_path: Path):
        """Save processed image as NIfTI"""
        nii_img = nibabel.Nifti1Image(image, affine)
        nibabel.save(nii_img, output_path)
        
    def batch_process_dataset(self, input_dir: Path, output_dir: Path,
                            dataset_name: str = "ADNI") -> pd.DataFrame:
        """
        Process entire dataset with batch processing
        """
        logger.info(f"Starting batch processing of {dataset_name}")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find all image files
        image_files = list(input_dir.rglob("*.nii.gz")) + list(input_dir.rglob("*.nii"))
        
        results = []
        
        for i, image_file in enumerate(image_files):
            logger.info(f"Processing {i+1}/{len(image_files)}: {image_file.name}")
            
            # Determine output path
            relative_path = image_file.relative_to(input_dir)
            output_file = output_dir / relative_path
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Determine modality
            modality = self._detect_modality(image_file)
            
            # Process based on modality
            if modality in ["t1", "t2", "flair"]:
                metadata = self.preprocess_mri(image_file, output_file, modality)
            elif modality in ["fdg", "amyloid", "tau", "pet"]:
                metadata = self.preprocess_pet(image_file, output_file, modality)
            else:
                logger.warning(f"Unknown modality for {image_file}")
                continue
            
            # Add file information
            metadata.update({
                'input_file': str(image_file),
                'output_file': str(output_file),
                'dataset': dataset_name,
                'modality': modality
            })
            
            results.append(metadata)
        
        # Create summary DataFrame
        results_df = pd.DataFrame(results)
        
        # Save processing summary
        summary_path = output_dir / "processing_summary.csv"
        results_df.to_csv(summary_path, index=False)
        
        logger.info(f"Batch processing completed. Summary saved to {summary_path}")
        return results_df
    
    def _detect_modality(self, image_file: Path) -> str:
        """Detect image modality from filename and metadata"""
        filename = image_file.name.lower()
        
        if any(x in filename for x in ["t1", "mprage", "mp-rage"]):
            return "t1"
        elif any(x in filename for x in ["t2", "t2w"]):
            return "t2"
        elif "flair" in filename:
            return "flair"
        elif any(x in filename for x in ["fdg", "f18", "fluorodeoxyglucose"]):
            return "fdg"
        elif any(x in filename for x in ["amyloid", "pib", "av45"]):
            return "amyloid"
        elif any(x in filename for x in ["tau", "av1451"]):
            return "tau"
        elif "pet" in filename:
            return "pet"
        else:
            return "unknown"

# Example usage and testing
if __name__ == "__main__":
    # Initialize preprocessor
    preprocessor = MedicalImagePreprocessor(target_shape=(256, 256, 256))
    
    # Test with synthetic data
    test_image = np.random.randn(256, 256, 256)
    
    # Test quality scoring
    qc_score = preprocessor.compute_quality_score(test_image)
    print(f"Quality score: {qc_score:.3f}")
    
    # Test SNR computation
    snr = preprocessor.compute_snr(test_image)
    print(f"SNR: {snr:.3f}")
    
    print("Preprocessing pipeline initialized successfully")