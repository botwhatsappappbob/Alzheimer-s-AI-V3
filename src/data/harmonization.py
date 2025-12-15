"""
Cross-Site Data Harmonization for Alzheimer's Disease Detection
Handles harmonization of data from different imaging centers and protocols
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
import nibabel as nib
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataHarmonizer:
    """
    Comprehensive harmonization pipeline for multi-site medical data
    Handles scanner differences, protocol variations, and demographic differences
    """
    
    def __init__(self):
        self.scalers = {}
        self.imputers = {}
        self.site_stats = {}
        self.harmonization_params = {}
        
    def harmonize_imaging_data(self, image_path: Path, site_id: str, 
                              target_site: str = 'ADNI') -> np.ndarray:
        """
        Harmonize imaging data across different sites and scanners
        
        Args:
            image_path: Path to the image file
            site_id: Source site identifier (e.g., 'ADNI', 'OASIS', 'AIBL')
            target_site: Target site for harmonization (default: ADNI)
            
        Returns:
            Harmonized image data as numpy array
        """
        logger.info(f"Harmonizing image from {site_id} to {target_site} standards")
        
        try:
            # Load image
            nii_img = nib.load(image_path)
            image_data = nii_img.get_fdata()
            
            # Site-specific harmonization
            if site_id == 'OASIS':
                harmonized_data = self._harmonize_oasis_to_adni(image_data)
            elif site_id == 'AIBL':
                harmonized_data = self._harmonize_aibl_to_adni(image_data)
            elif site_id == 'ADNI':
                harmonized_data = image_data  # Already ADNI standard
            else:
                logger.warning(f"Unknown site {site_id}, applying generic harmonization")
                harmonized_data = self._generic_harmonization(image_data)
                
            # Apply intensity standardization
            harmonized_data = self._standardize_intensity(harmonized_data, site_id)
            
            return harmonized_data
            
        except Exception as e:
            logger.error(f"Image harmonization failed: {e}")
            raise
            
    def harmonize_clinical_data(self, clinical_df: pd.DataFrame, 
                               site_id: str) -> pd.DataFrame:
        """
        Harmonize clinical and cognitive assessment data across sites
        
        Args:
            clinical_df: DataFrame with clinical data
            site_id: Source site identifier
            
        Returns:
            Harmonized clinical DataFrame
        """
        logger.info(f"Harmonizing clinical data from {site_id}")
        
        harmonized_df = clinical_df.copy()
        
        # Site-specific clinical harmonization
        if site_id == 'OASIS':
            harmonized_df = self._harmonize_oasis_clinical(harmonized_df)
        elif site_id == 'AIBL':
            harmonized_df = self._harmonize_aibl_clinical(harmonized_df)
        elif site_id == 'ADNI':
            harmonized_df = self._harmonize_adni_clinical(harmonized_df)
            
        # Apply demographic harmonization
        harmonized_df = self._harmonize_demographics(harmonized_df)
        
        # Handle missing values
        harmonized_df = self._harmonize_missing_values(harmonized_df)
        
        return harmonized_df
        
    def _harmonize_oasis_to_adni(self, image_data: np.ndarray) -> np.ndarray:
        """Harmonize OASIS imaging data to ADNI standards"""
        # OASIS-specific adjustments
        # 1. Intensity normalization (OASIS may have different intensity ranges)
        normalized = self._normalize_intensity_range(image_data, source='OASIS', target='ADNI')
        
        # 2. Spatial resolution harmonization if needed
        # OASIS and ADNI have similar resolutions, so minimal adjustment needed
        
        return normalized
        
    def _harmonize_aibl_to_adni(self, image_data: np.ndarray) -> np.ndarray:
        """Harmonize AIBL imaging data to ADNI standards"""
        # AIBL-specific adjustments
        # AIBL follows ADNI protocols closely, so minimal harmonization needed
        normalized = self._normalize_intensity_range(image_data, source='AIBL', target='ADNI')
        
        return normalized
        
    def _generic_harmonization(self, image_data: np.ndarray) -> np.ndarray:
        """Generic harmonization for unknown sites"""
        # Apply standard intensity normalization
        normalized = (image_data - np.mean(image_data)) / np.std(image_data)
        
        # Clip extreme values
        normalized = np.clip(normalized, -3, 3)
        
        return normalized
        
    def _standardize_intensity(self, image_data: np.ndarray, site_id: str) -> np.ndarray:
        """Standardize image intensity across sites"""
        if site_id not in self.site_stats:
            # Calculate site statistics if not available
            self.site_stats[site_id] = {
                'mean': np.mean(image_data),
                'std': np.std(image_data),
                'percentiles': np.percentile(image_data, [1, 99])
            }
            
        # Apply robust scaling
        p1, p99 = self.site_stats[site_id]['percentiles']
        standardized = np.clip(image_data, p1, p99)
        standardized = (standardized - self.site_stats[site_id]['mean']) / self.site_stats[site_id]['std']
        
        return standardized
        
    def _normalize_intensity_range(self, image_data: np.ndarray, source: str, target: str) -> np.ndarray:
        """Normalize intensity ranges between different sites"""
        # Define typical intensity ranges for each site
        intensity_ranges = {
            'OASIS': {'min': 0, 'max': 4000},  # Typical OASIS range
            'AIBL': {'min': 0, 'max': 3000},   # Typical AIBL range  
            'ADNI': {'min': 0, 'max': 2500}    # Typical ADNI range
        }
        
        if source in intensity_ranges and target in intensity_ranges:
            source_range = intensity_ranges[source]['max'] - intensity_ranges[source]['min']
            target_range = intensity_ranges[target]['max'] - intensity_ranges[target]['min']
            
            # Scale to target range
            normalized = (image_data - intensity_ranges[source]['min']) / source_range
            normalized = normalized * target_range + intensity_ranges[target]['min']
            
            return normalized
        else:
            return image_data
            
    def _harmonize_oasis_clinical(self, clinical_df: pd.DataFrame) -> pd.DataFrame:
        """Harmonize OASIS clinical data"""
        harmonized = clinical_df.copy()
        
        # OASIS-specific adjustments
        # 1. Convert OASIS CDR to standard format if needed
        if 'cdr' in harmonized.columns:
            harmonized['cdr'] = harmonized['cdr'].astype(float)
            
        # 2. Handle OASIS-specific missing value indicators
        missing_indicators = [-1, -999, -9999]
        for col in harmonized.select_dtypes(include=[np.number]).columns:
            harmonized[col] = harmonized[col].replace(missing_indicators, np.nan)
            
        return harmonized
        
    def _harmonize_aibl_clinical(self, clinical_df: pd.DataFrame) -> pd.DataFrame:
        """Harmonize AIBL clinical data"""
        harmonized = clinical_df.copy()
        
        # AIBL-specific adjustments
        # AIBL follows similar protocols to ADNI, so minimal adjustment needed
        
        return harmonized
        
    def _harmonize_adni_clinical(self, clinical_df: pd.DataFrame) -> pd.DataFrame:
        """Harmonize ADNI clinical data (already standard)"""
        return clinical_df
        
    def _harmonize_demographics(self, clinical_df: pd.DataFrame) -> pd.DataFrame:
        """Harmonize demographic variables across sites"""
        harmonized = clinical_df.copy()
        
        # Standardize age ranges (remove outliers)
        if 'age' in harmonized.columns:
            # Remove ages outside reasonable range (50-100 years)
            harmonized = harmonized[(harmonized['age'] >= 50) & (harmonized['age'] <= 100)]
            
        # Standardize education years (cap at reasonable maximum)
        if 'education' in harmonized.columns:
            harmonized['education'] = harmonized['education'].clip(upper=20)
            
        return harmonized
        
    def _harmonize_missing_values(self, clinical_df: pd.DataFrame) -> pd.DataFrame:
        """Harmonize missing value handling across sites"""
        harmonized = clinical_df.copy()
        
        # Initialize imputers if not already done
        if 'default' not in self.imputers:
            self.imputers['default'] = SimpleImputer(strategy='median')
            
        # Apply imputation to numeric columns
        numeric_cols = harmonized.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            harmonized[numeric_cols] = self.imputers['default'].fit_transform(harmonized[numeric_cols])
            
        return harmonized
        
    def create_harmonization_report(self, original_data: pd.DataFrame, 
                                   harmonized_data: pd.DataFrame) -> Dict:
        """
        Create a report of the harmonization process
        
        Args:
            original_data: Data before harmonization
            harmonized_data: Data after harmonization
            
        Returns:
            Dictionary with harmonization statistics
        """
        report = {
            'original_shape': original_data.shape,
            'harmonized_shape': harmonized_data.shape,
            'rows_removed': original_data.shape[0] - harmonized_data.shape[0],
            'columns_changed': {},
            'missing_values_before': original_data.isnull().sum().to_dict(),
            'missing_values_after': harmonized_data.isnull().sum().to_dict(),
            'summary_statistics': {
                'before': original_data.describe().to_dict(),
                'after': harmonized_data.describe().to_dict()
            }
        }
        
        return report
        
# Example usage and testing
if __name__ == "__main__":
    # Initialize harmonizer
    harmonizer = DataHarmonizer()
    
    # Test with synthetic data
    test_image = np.random.randn(256, 256, 256)
    harmonized_image = harmonizer.harmonize_imaging_data(
        test_image, 'OASIS', 'ADNI'
    )
    
    print(f"Original shape: {test_image.shape}")
    print(f"Harmonized shape: {harmonized_image.shape}")
    print(f"Intensity range before: [{test_image.min():.2f}, {test_image.max():.2f}]")
    print(f"Intensity range after: [{harmonized_image.min():.2f}, {harmonized_image.max():.2f}]")
    
    # Test clinical data harmonization
    test_clinical = pd.DataFrame({
        'age': [65, 70, 75, 80],
        'mmse': [28, 26, 24, 22],
        'cdr': [0.0, 0.5, 1.0, 2.0],
        'education': [12, 16, 18, 20]
    })
    
    harmonized_clinical = harmonizer.harmonize_clinical_data(test_clinical, 'OASIS')
    print("\nClinical Data Harmonization:")
    print("Before:")
    print(test_clinical.describe())
    print("\nAfter:")
    print(harmonized_clinical.describe())
    
    print("\nData harmonization pipeline initialized successfully")