"""
AIBL Dataset Downloader and Preprocessor
Handles downloading and preprocessing of Australian Imaging Biomarker and Lifestyle data
"""

import os
import requests
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
import json
from datetime import datetime
import hashlib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AIBLDownloader:
    """
    AIBL (Australian Imaging Biomarker and Lifestyle) dataset downloader
    Provides access to multimodal Australian dementia research data
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('AIBL_API_KEY')
        self.base_url = "https://aibl.csiro.au"
        self.session = requests.Session()
        
        # AIBL dataset structure
        self.aibl_structure = {
            'total_subjects': 3000,
            'age_range': (50, 100),
            'modalities': ['MRI', 'PET', 'Clinical', 'Biomarkers', 'Lifestyle'],
            'pet_tracers': ['PIB', 'FDG', 'AV45'],
            'follow_up_period': 18,  # months
            'study_duration': '15+ years'
        }
        
    def authenticate(self) -> bool:
        """
        Authenticate with AIBL data repository
        
        Returns:
            True if authentication successful, False otherwise
        """
        # AIBL access typically requires application and approval
        # For demo purposes, we'll simulate authentication
        logger.info("AIBL authentication requires application and approval")
        logger.info("Contact aibl.csiro.au for data access")
        return True  # Simulate successful authentication for demo
        
    def download_subject_data(self, output_dir: Path, criteria: Dict = None) -> pd.DataFrame:
        """
        Download AIBL subject demographics and clinical data
        
        Args:
            output_dir: Directory to save downloaded data
            criteria: Dictionary of selection criteria
            
        Returns:
            DataFrame with subject information
        """
        logger.info("Downloading AIBL subject data")
        
        # Default criteria
        if criteria is None:
            criteria = {
                'min_age': 50,
                'max_age': 100,
                'groups': ['CN', 'MCI', 'AD'],
                'modalities': ['MRI', 'PET']
            }
            
        try:
            # Create demo AIBL data
            subjects_per_group = 15  # Demo size
            
            subjects = []
            groups = criteria.get('groups', ['CN', 'MCI', 'AD'])
            
            for i in range(subjects_per_group * len(groups)):
                group = groups[i % len(groups)]
                
                # AIBL-specific characteristics
                subject_data = {
                    'subject_id': f'AIBL{i+1:04d}',
                    'group': group,
                    'age': np.random.randint(50, 100),
                    'gender': i % 2,
                    'education': np.random.randint(8, 20),
                    'mmse': np.random.randint(20, 30) if group == 'CN' else np.random.randint(10, 26),
                    'cdr': {'CN': 0, 'MCI': 0.5, 'AD': 1.0}[group],
                    'adas_cog': np.random.randint(5, 40) if group == 'CN' else np.random.randint(20, 70),
                    'apoe_genotype': np.random.choice(['E2/E2', 'E2/E3', 'E3/E3', 'E3/E4', 'E4/E4']),
                    'visit': 'baseline',
                    'site': np.random.choice(['Melbourne', 'Perth']),
                    'recruitment_year': np.random.randint(2006, 2021)
                }
                
                subjects.append(subject_data)
                
            # Create DataFrame
            subjects_df = pd.DataFrame(subjects)
            
            # Save to file
            subjects_file = output_dir / "aibl_subjects.csv"
            subjects_df.to_csv(subjects_file, index=False)
            
            logger.info(f"AIBL subject data created with {len(subjects)} subjects")
            return subjects_df
            
        except Exception as e:
            logger.error(f"AIBL subject data download failed: {e}")
            return pd.DataFrame()
            
    def download_imaging_data(self, subjects_df: pd.DataFrame, output_dir: Path,
                             modalities: List[str] = None) -> List[Path]:
        """
        Download AIBL imaging data
        
        Args:
            subjects_df: DataFrame with subject information
            output_dir: Directory to save imaging data
            modalities: List of modalities to download
            
        Returns:
            List of paths to downloaded images
        """
        logger.info("Downloading AIBL imaging data")
        
        if modalities is None:
            modalities = ['T1', 'FDG_PET', 'PIB_PET']
            
        output_dir.mkdir(parents=True, exist_ok=True)
        downloaded_files = []
        
        try:
            # Create synthetic imaging data for demo
            import nibabel as nib
            import numpy as np
            
            for _, subject in subjects_df.iterrows():
                for modality in modalities:
                    # Create modality-specific synthetic data
                    if modality == 'T1':
                        # T1-weighted MRI
                        image_data = np.random.randn(256, 256, 256)
                        x, y, z = np.ogrid[:256, :256, :256]
                        center_x, center_y, center_z = 128, 128, 128
                        brain_mask = ((x - center_x)**2 + (y - center_y)**2 + (z - center_z)**2) < 80**2
                        image_data[brain_mask] += 0.5
                        
                    elif modality == 'FDG_PET':
                        # FDG PET - metabolism patterns
                        image_data = np.random.randn(256, 256, 256) * 0.5
                        # Simulate reduced metabolism in temporal lobe for AD
                        if subject['group'] == 'AD':
                            temporal_lobe = ((x - 100)**2 + (y - 150)**2 + (z - 120)**2) < 30**2
                            image_data[temporal_lobe] -= 0.2
                            
                    elif modality == 'PIB_PET':
                        # PIB PET - amyloid deposition
                        image_data = np.random.randn(256, 256, 256) * 0.3
                        # Simulate amyloid deposition in AD
                        if subject['group'] == 'AD':
                            cortex_mask = ((x - center_x)**2 + (y - center_y)**2 + (z - center_z)**2) < 75**2
                            cortex_mask &= ((x - center_x)**2 + (y - center_y)**2 + (z - center_z)**2) > 70**2
                            image_data[cortex_mask] += 0.4
                            
                    # Save as NIfTI
                    image_path = output_dir / f"{subject['subject_id']}_{modality}.nii.gz"
                    nii_img = nib.Nifti1Image(image_data, np.eye(4))
                    nib.save(nii_img, image_path)
                    downloaded_files.append(image_path)
                    
            logger.info(f"AIBL imaging data created with {len(downloaded_files)} images")
            return downloaded_files
            
        except Exception as e:
            logger.error(f"AIBL imaging data download failed: {e}")
            return []
            
    def download_biomarker_data(self, subjects_df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
        """
        Download AIBL biomarker data
        
        Args:
            subjects_df: DataFrame with subject information
            output_dir: Directory to save biomarker data
            
        Returns:
            DataFrame with biomarker data
        """
        logger.info("Downloading AIBL biomarker data")
        
        try:
            # Create synthetic biomarker data
            biomarker_data = []
            
            for _, subject in subjects_df.iterrows():
                # CSF biomarkers
                csf_data = {
                    'subject_id': subject['subject_id'],
                    'visit': subject['visit'],
                    'csf_abeta42': np.random.normal(800, 200) if subject['group'] == 'CN' else np.random.normal(500, 150),
                    'csf_ptau': np.random.normal(25, 10) if subject['group'] == 'CN' else np.random.normal(45, 15),
                    'csf_ttau': np.random.normal(300, 100) if subject['group'] == 'CN' else np.random.normal(500, 150),
                    'plasma_abeta42': np.random.normal(50, 15),
                    'plasma_ptau': np.random.normal(5, 2),
                    'plasma_nfl': np.random.normal(15, 5)
                }
                biomarker_data.append(csf_data)
                
            # Create DataFrame
            biomarker_df = pd.DataFrame(biomarker_data)
            
            # Save to file
            biomarker_file = output_dir / "aibl_biomarkers.csv"
            biomarker_df.to_csv(biomarker_file, index=False)
            
            logger.info(f"AIBL biomarker data created with {len(biomarker_data)} records")
            return biomarker_df
            
        except Exception as e:
            logger.error(f"AIBL biomarker data download failed: {e}")
            return pd.DataFrame()
            
    def download_lifestyle_data(self, subjects_df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
        """
        Download AIBL lifestyle and cognitive data
        
        Args:
            subjects_df: DataFrame with subject information
            output_dir: Directory to save lifestyle data
            
        Returns:
            DataFrame with lifestyle data
        """
        logger.info("Downloading AIBL lifestyle data")
        
        try:
            # Create synthetic lifestyle data
            lifestyle_data = []
            
            for _, subject in subjects_df.iterrows():
                lifestyle_info = {
                    'subject_id': subject['subject_id'],
                    'visit': subject['visit'],
                    # Physical activity
                    'physical_activity_hours': np.random.randint(0, 10),
                    'exercise_frequency': np.random.randint(0, 7),
                    # Diet
                    'mediterranean_diet_score': np.random.randint(0, 10),
                    'alcohol_consumption': np.random.choice(['none', 'light', 'moderate', 'heavy']),
                    # Cognitive activity
                    'cognitive_activity_score': np.random.randint(0, 10),
                    'social_activity_score': np.random.randint(0, 10),
                    # Medical history
                    'cardiovascular_disease': np.random.choice([0, 1], p=[0.8, 0.2]),
                    'diabetes': np.random.choice([0, 1], p=[0.9, 0.1]),
                    'hypertension': np.random.choice([0, 1], p=[0.7, 0.3]),
                    'depression_history': np.random.choice([0, 1], p=[0.85, 0.15]),
                    # Sleep
                    'sleep_quality_score': np.random.randint(0, 10),
                    'sleep_duration': np.random.randint(4, 10)
                }
                lifestyle_data.append(lifestyle_info)
                
            # Create DataFrame
            lifestyle_df = pd.DataFrame(lifestyle_data)
            
            # Save to file
            lifestyle_file = output_dir / "aibl_lifestyle.csv"
            lifestyle_df.to_csv(lifestyle_file, index=False)
            
            logger.info(f"AIBL lifestyle data created with {len(lifestyle_data)} records")
            return lifestyle_df
            
        except Exception as e:
            logger.error(f"AIBL lifestyle data download failed: {e}")
            return pd.DataFrame()
            
    def create_longitudinal_dataset(self, subjects_df: pd.DataFrame,
                                   biomarker_df: pd.DataFrame,
                                   lifestyle_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create longitudinal dataset by merging all AIBL data
        
        Args:
            subjects_df: Subject demographics
            biomarker_df: Biomarker data
            lifestyle_df: Lifestyle data
            
        Returns:
            Merged longitudinal DataFrame
        """
        logger.info("Creating AIBL longitudinal dataset")
        
        try:
            # Merge all datasets
            longitudinal_df = subjects_df.copy()
            
            # Merge biomarker data
            if not biomarker_df.empty:
                longitudinal_df = longitudinal_df.merge(
                    biomarker_df, on=['subject_id', 'visit'], how='left'
                )
                
            # Merge lifestyle data
            if not lifestyle_df.empty:
                longitudinal_df = longitudinal_df.merge(
                    lifestyle_df, on=['subject_id', 'visit'], how='left'
                )
                
            # Sort by subject and visit
            longitudinal_df = longitudinal_df.sort_values(['subject_id', 'visit'])
            
            logger.info(f"Longitudinal dataset created with {len(longitudinal_df)} records")
            return longitudinal_df
            
        except Exception as e:
            logger.error(f"Longitudinal dataset creation failed: {e}")
            return pd.DataFrame()
            
    def create_manifest(self, downloaded_files: List[Path], dataset_info: Dict) -> Dict:
        """
        Create manifest file documenting the downloaded data
        
        Args:
            downloaded_files: List of downloaded file paths
            dataset_info: Dictionary with dataset information
            
        Returns:
            Dictionary with manifest information
        """
        manifest = {
            'dataset': 'AIBL',
            'version': 'demo_v1.0',
            'download_date': datetime.now().isoformat(),
            'total_files': len(downloaded_files),
            'dataset_info': dataset_info,
            'files': [str(f) for f in downloaded_files],
            'checksums': {}
        }
        
        # Calculate checksums
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
                # Check if file is a valid NIfTI or CSV file
                try:
                    if file_path.suffix == '.nii.gz':
                        import nibabel as nib
                        img = nib.load(file_path)
                        _ = img.get_fdata()
                    elif file_path.suffix == '.csv':
                        df = pd.read_csv(file_path)
                        # Check if DataFrame is not empty
                        if len(df) > 0:
                            validation_report['valid_files'] += 1
                        else:
                            validation_report['invalid_files'].append(str(file_path))
                    else:
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
    downloader = AIBLDownloader()
    
    # Download demo data
    output_dir = Path("data/aibl")
    
    print("Downloading AIBL data...")
    
    # Subject data
    subjects_df = downloader.download_subject_data(output_dir)
    print(f"Subjects: {len(subjects_df)} records")
    
    # Imaging data
    imaging_files = downloader.download_imaging_data(subjects_df, output_dir / "images")
    print(f"Images: {len(imaging_files)} files")
    
    # Biomarker data
    biomarker_df = downloader.download_biomarker_data(subjects_df, output_dir)
    print(f"Biomarkers: {len(biomarker_df)} records")
    
    # Lifestyle data
    lifestyle_df = downloader.download_lifestyle_data(subjects_df, output_dir)
    print(f"Lifestyle: {len(lifestyle_df)} records")
    
    # Create longitudinal dataset
    longitudinal_df = downloader.create_longitudinal_dataset(subjects_df, biomarker_df, lifestyle_df)
    print(f"Longitudinal: {len(longitudinal_df)} records")
    
    # Create manifest
    all_files = imaging_files + [output_dir / "aibl_subjects.csv", 
                                output_dir / "aibl_biomarkers.csv",
                                output_dir / "aibl_lifestyle.csv"]
    manifest = downloader.create_manifest(all_files, downloader.aibl_structure)
    
    print(f"\nTotal files: {manifest['total_files']}")
    print("AIBL downloader initialized successfully")