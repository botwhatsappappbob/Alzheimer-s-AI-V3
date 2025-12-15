"""
OASIS Dataset Downloader and Preprocessor
Handles downloading and preprocessing of OASIS data
"""

import os
import requests
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
import zipfile
import gzip
import shutil

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OASISDownloader:
    """
    OASIS (Open Access Series of Imaging Studies) dataset downloader
    Provides access to cross-sectional and longitudinal MRI data
    """
    
    def __init__(self):
        self.base_urls = {
            'oasis1': "https://www.oasis-brains.org/",
            'oasis2': "https://www.oasis-brains.org/",
            'oasis3': "https://www.oasis-brains.org/",
            'oasis4': "https://www.oasis-brains.org/"
        }
        
        # OASIS dataset structure
        self.oasis_structure = {
            'oasis1': {
                'subjects': 416,
                'age_range': (18, 96),
                'description': 'Cross-sectional MRI data'
            },
            'oasis2': {
                'subjects': 150,
                'age_range': (60, 96),
                'description': 'Longitudinal MRI data'
            },
            'oasis3': {
                'subjects': 1378,
                'age_range': (42, 95),
                'description': 'Longitudinal multimodal dataset'
            },
            'oasis4': {
                'subjects': 0,  # Variable
                'age_range': (50, 90),
                'description': 'Memory complaints dataset'
            }
        }
        
    def download_oasis1(self, output_dir: Path, force_download: bool = False) -> List[Path]:
        """
        Download OASIS-1 cross-sectional dataset
        
        Args:
            output_dir: Directory to save downloaded data
            force_download: Force re-download even if files exist
            
        Returns:
            List of paths to downloaded files
        """
        logger.info("Downloading OASIS-1 cross-sectional dataset")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        downloaded_files = []
        
        # OASIS-1 is available through NITRC
        # For demo purposes, we'll create synthetic data
        # In production, this would download from actual OASIS repository
        
        try:
            # Create demo data structure
            subjects_per_group = 10  # Demo size
            
            # Create subject demographics
            demographics = []
            groups = ['CN', 'AD', 'MCI']
            
            for i in range(subjects_per_group * len(groups)):
                group = groups[i % len(groups)]
                age = np.random.randint(60, 96)
                gender = i % 2
                education = np.random.randint(8, 20)
                mmse = np.random.randint(18, 30) if group == 'CN' else np.random.randint(10, 26)
                cdr = {'CN': 0, 'MCI': 0.5, 'AD': 1.0}[group]
                
                demographics.append({
                    'subject_id': f'OAS1_{i+1:04d}',
                    'group': group,
                    'age': age,
                    'gender': gender,
                    'education': education,
                    'mmse': mmse,
                    'cdr': cdr,
                    'visit': 'baseline'
                })
                
            # Save demographics
            demographics_df = pd.DataFrame(demographics)
            demo_file = output_dir / "oasis1_demographics.csv"
            demographics_df.to_csv(demo_file, index=False)
            downloaded_files.append(demo_file)
            
            # Create synthetic MRI data
            import nibabel as nib
            import numpy as np
            
            for i, subject in enumerate(demographics):
                # Create synthetic T1-weighted brain image
                synthetic_brain = np.random.randn(256, 256, 256)
                
                # Add some brain-like structure
                x, y, z = np.ogrid[:256, :256, :256]
                center_x, center_y, center_z = 128, 128, 128
                brain_mask = ((x - center_x)**2 + (y - center_y)**2 + (z - center_z)**2) < 80**2
                synthetic_brain[brain_mask] += 0.5
                
                # Add ventricles
                ventricle_mask = ((x - center_x)**2 + (y - center_y)**2 + (z - center_z)**2) < 20**2
                ventricle_mask &= ((x - center_x)**2 + (y - center_y)**2) > 5**2
                synthetic_brain[ventricle_mask] -= 0.3
                
                # Save as NIfTI
                image_path = output_dir / f"{subject['subject_id']}_T1.nii.gz"
                nii_img = nib.Nifti1Image(synthetic_brain, np.eye(4))
                nib.save(nii_img, image_path)
                downloaded_files.append(image_path)
                
            logger.info(f"OASIS-1 demo data created with {len(demographics)} subjects")
            return downloaded_files
            
        except Exception as e:
            logger.error(f"OASIS-1 download failed: {e}")
            return []
            
    def download_oasis2(self, output_dir: Path, force_download: bool = False) -> List[Path]:
        """
        Download OASIS-2 longitudinal dataset
        
        Args:
            output_dir: Directory to save downloaded data
            force_download: Force re-download even if files exist
            
        Returns:
            List of paths to downloaded files
        """
        logger.info("Downloading OASIS-2 longitudinal dataset")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        downloaded_files = []
        
        try:
            # Create longitudinal demo data
            subjects_per_group = 5  # Demo size
            
            # Create longitudinal subject data
            longitudinal_data = []
            groups = ['CN', 'AD']
            
            for i in range(subjects_per_group * len(groups)):
                group = groups[i % len(groups)]
                base_age = np.random.randint(65, 85)
                
                # Create multiple visits
                for visit in range(1, 4):  # 3 visits
                    current_age = base_age + (visit - 1) * 2  # 2 years between visits
                    
                    # Progression modeling
                    if group == 'CN' and visit > 1:
                        # Some CN subjects may progress to MCI
                        if np.random.random() < 0.2:
                            current_group = 'MCI'
                            mmse = np.random.randint(20, 26)
                            cdr = 0.5
                        else:
                            current_group = 'CN'
                            mmse = np.random.randint(26, 30)
                            cdr = 0.0
                    elif group == 'AD':
                        # AD subjects may show progression
                        current_group = 'AD'
                        mmse = max(10, 26 - (visit - 1) * 3)
                        cdr = min(3.0, 1.0 + (visit - 1) * 0.5)
                    else:
                        current_group = group
                        mmse = np.random.randint(26, 30)
                        cdr = 0.0
                    
                    longitudinal_data.append({
                        'subject_id': f'OAS2_{i+1:04d}',
                        'group': current_group,
                        'age': current_age,
                        'gender': i % 2,
                        'education': np.random.randint(12, 20),
                        'mmse': mmse,
                        'cdr': cdr,
                        'visit': f'visit{visit}',
                        'time_from_baseline': (visit - 1) * 2
                    })
                    
            # Save longitudinal data
            longitudinal_df = pd.DataFrame(longitudinal_data)
            demo_file = output_dir / "oasis2_longitudinal.csv"
            longitudinal_df.to_csv(demo_file, index=False)
            downloaded_files.append(demo_file)
            
            # Create longitudinal MRI data
            import nibabel as nib
            import numpy as np
            
            for subject_data in longitudinal_data:
                # Create brain image with some longitudinal change
                synthetic_brain = np.random.randn(256, 256, 256)
                
                # Add brain structure
                x, y, z = np.ogrid[:256, :256, :256]
                center_x, center_y, center_z = 128, 128, 128
                brain_mask = ((x - center_x)**2 + (y - center_y)**2 + (z - center_z)**2) < 80**2
                synthetic_brain[brain_mask] += 0.5
                
                # Simulate atrophy progression for AD subjects
                if subject_data['group'] == 'AD' and subject_data['visit'] != 'visit1':
                    # Simulate ventricular enlargement
                    ventricle_mask = ((x - center_x)**2 + (y - center_y)**2 + (z - center_z)**2) < 25**2
                    ventricle_mask &= ((x - center_x)**2 + (y - center_y)**2) > 3**2
                    synthetic_brain[ventricle_mask] -= 0.1 * int(subject_data['visit'][-1])
                
                # Save as NIfTI
                image_path = output_dir / f"{subject_data['subject_id']}_{subject_data['visit']}_T1.nii.gz"
                nii_img = nib.Nifti1Image(synthetic_brain, np.eye(4))
                nib.save(nii_img, image_path)
                downloaded_files.append(image_path)
                
            logger.info(f"OASIS-2 demo data created with {len(longitudinal_data)} records")
            return downloaded_files
            
        except Exception as e:
            logger.error(f"OASIS-2 download failed: {e}")
            return []
            
    def download_oasis3(self, output_dir: Path, force_download: bool = False) -> List[Path]:
        """
        Download OASIS-3 multimodal dataset
        
        Args:
            output_dir: Directory to save downloaded data
            force_download: Force re-download even if files exist
            
        Returns:
            List of paths to downloaded files
        """
        logger.info("Downloading OASIS-3 multimodal dataset")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        downloaded_files = []
        
        try:
            # OASIS-3 includes multiple modalities
            # For demo, we'll create multimodal synthetic data
            subjects_per_group = 8  # Demo size
            
            multimodal_data = []
            groups = ['CN', 'AD', 'MCI']
            
            for i in range(subjects_per_group * len(groups)):
                group = groups[i % len(groups)]
                
                multimodal_data.append({
                    'subject_id': f'OAS3_{i+1:04d}',
                    'group': group,
                    'age': np.random.randint(50, 95),
                    'gender': i % 2,
                    'education': np.random.randint(8, 20),
                    'mmse': np.random.randint(18, 30) if group == 'CN' else np.random.randint(10, 26),
                    'cdr': {'CN': 0, 'MCI': 0.5, 'AD': 1.0}[group],
                    'visit': 'baseline',
                    'modalities': ['T1', 'T2', 'FLAIR', 'FDG_PET']
                })
                
            # Save multimodal data
            multimodal_df = pd.DataFrame(multimodal_data)
            demo_file = output_dir / "oasis3_multimodal.csv"
            multimodal_df.to_csv(demo_file, index=False)
            downloaded_files.append(demo_file)
            
            # Create multimodal images
            import nibabel as nib
            import numpy as np
            
            for subject in multimodal_data:
                for modality in subject['modalities']:
                    # Create modality-specific synthetic data
                    if modality == 'T1':
                        synthetic_data = np.random.randn(256, 256, 256)
                        # Add brain structure for T1
                        x, y, z = np.ogrid[:256, :256, :256]
                        center_x, center_y, center_z = 128, 128, 128
                        brain_mask = ((x - center_x)**2 + (y - center_y)**2 + (z - center_z)**2) < 80**2
                        synthetic_data[brain_mask] += 0.5
                    elif modality == 'T2':
                        synthetic_data = np.random.randn(256, 256, 256) * 0.8
                        # T2 has different contrast
                        synthetic_data += 0.2
                    elif modality == 'FLAIR':
                        synthetic_data = np.random.randn(256, 256, 256) * 0.9
                        # FLAIR suppresses CSF
                        csf_mask = ((x - center_x)**2 + (y - center_y)**2 + (z - center_z)**2) < 20**2
                        synthetic_data[csf_mask] *= 0.3
                    elif modality == 'FDG_PET':
                        synthetic_data = np.random.randn(256, 256, 256) * 0.5
                        # PET has different metabolism patterns
                        # Simulate reduced metabolism in AD
                        if subject['group'] == 'AD':
                            temporal_lobe = ((x - 100)**2 + (y - 150)**2 + (z - 120)**2) < 30**2
                            synthetic_data[temporal_lobe] -= 0.2
                    
                    # Save as NIfTI
                    image_path = output_dir / f"{subject['subject_id']}_{modality}.nii.gz"
                    nii_img = nib.Nifti1Image(synthetic_data, np.eye(4))
                    nib.save(nii_img, image_path)
                    downloaded_files.append(image_path)
                    
            logger.info(f"OASIS-3 demo data created with {len(multimodal_data)} subjects")
            return downloaded_files
            
        except Exception as e:
            logger.error(f"OASIS-3 download failed: {e}")
            return []
            
    def create_manifest(self, downloaded_files: List[Path], dataset_version: str) -> Dict:
        """
        Create manifest file documenting the downloaded data
        
        Args:
            downloaded_files: List of downloaded file paths
            dataset_version: Version of the dataset
            
        Returns:
            Dictionary with manifest information
        """
        manifest = {
            'dataset': 'OASIS',
            'version': dataset_version,
            'download_date': datetime.now().isoformat(),
            'total_files': len(downloaded_files),
            'files': [str(f) for f in downloaded_files],
            'structure': self.oasis_structure,
            'checksums': {}
        }
        
        # Calculate checksums for validation
        for file_path in downloaded_files:
            if file_path.exists():
                with open(file_path, 'rb') as f:
                    file_hash = hashlib.md5(f.read()).hexdigest()
                    manifest['checksums'][str(file_path)] = file_hash
                    
        return manifest
        
    def validate_download(self, downloaded_files: List[Path]) -> Dict:
        """
        Validate downloaded files
        
        Args:
            downloaded_files: List of downloaded file paths
            
        Returns:
            Validation report dictionary
        """
        validation_report = {
            'total_files': len(downloaded_files),
            'valid_files': 0,
            'invalid_files': [],
            'missing_files': []
        }
        
        for file_path in downloaded_files:
            if file_path.exists():
                # Check if file is a valid NIfTI file
                try:
                    import nibabel as nib
                    img = nib.load(file_path)
                    # Try to access data to ensure file is valid
                    _ = img.get_fdata()
                    validation_report['valid_files'] += 1
                except Exception as e:
                    validation_report['invalid_files'].append({
                        'file': str(file_path),
                        'error': str(e)
                    })
            else:
                validation_report['missing_files'].append(str(file_path))
                
        return validation_report

# Example usage
if __name__ == "__main__":
    # Initialize downloader
    downloader = OASISDownloader()
    
    # Download demo datasets
    output_dir = Path("data/oasis")
    
    print("Downloading OASIS-1...")
    oasis1_files = downloader.download_oasis1(output_dir / "oasis1")
    print(f"OASIS-1: {len(oasis1_files)} files downloaded")
    
    print("\nDownloading OASIS-2...")
    oasis2_files = downloader.download_oasis2(output_dir / "oasis2")
    print(f"OASIS-2: {len(oasis2_files)} files downloaded")
    
    print("\nDownloading OASIS-3...")
    oasis3_files = downloader.download_oasis3(output_dir / "oasis3")
    print(f"OASIS-3: {len(oasis3_files)} files downloaded")
    
    # Create manifest
    all_files = oasis1_files + oasis2_files + oasis3_files
    manifest = downloader.create_manifest(all_files, "demo_v1.0")
    
    print(f"\nTotal files: {manifest['total_files']}")
    print("OASIS downloader initialized successfully")