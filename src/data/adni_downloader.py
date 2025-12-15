"""
ADNI Dataset Downloader and Preprocessor
Handles downloading and preprocessing of ADNI data with proper authentication
"""

import os
import requests
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
from urllib.parse import urljoin
import json
from datetime import datetime
import hashlib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ADNIDownloader:
    """
    ADNI dataset downloader with authentication and preprocessing capabilities
    Requires ADNI credentials for data access
    """
    
    def __init__(self, username: Optional[str] = None, password: Optional[str] = None):
        self.username = username or os.getenv('ADNI_USERNAME')
        self.password = password or os.getenv('ADNI_PASSWORD')
        self.session = requests.Session()
        self.base_url = "https://ida.loni.usc.edu"
        self.authenticated = False
        
        # ADNI data structure
        self.adni_structure = {
            'projects': ['ADNI1', 'ADNI2', 'ADNIGO', 'ADNI3'],
            'modalities': ['MRI', 'PET', 'fMRI', 'DTI'],
            'image_types': ['T1', 'T2', 'FLAIR'],
            'pet_types': ['FDG', 'AV45', 'PIB', 'Tau']
        }
        
    def authenticate(self) -> bool:
        """
        Authenticate with ADNI data archive
        
        Returns:
            True if authentication successful, False otherwise
        """
        if not self.username or not self.password:
            logger.error("ADNI credentials not provided")
            return False
            
        try:
            # ADNI authentication process
            auth_url = f"{self.base_url}/login.jsp"
            
            # Step 1: Get login page
            login_page = self.session.get(auth_url)
            
            # Step 2: Submit credentials
            auth_data = {
                'user_email': self.username,
                'user_pass': self.password,
                'project': 'ADNI'
            }
            
            response = self.session.post(auth_url, data=auth_data)
            
            # Check if authentication successful
            if response.status_code == 200 and 'ADNI' in response.text:
                self.authenticated = True
                logger.info("Successfully authenticated with ADNI")
                return True
            else:
                logger.error("ADNI authentication failed")
                return False
                
        except Exception as e:
            logger.error(f"Authentication error: {e}")
            return False
            
    def download_subject_list(self, criteria: Dict = None) -> pd.DataFrame:
        """
        Download list of subjects matching specified criteria
        
        Args:
            criteria: Dictionary of search criteria
            
        Returns:
            DataFrame with subject information
        """
        if not self.authenticated:
            logger.error("Not authenticated with ADNI")
            return pd.DataFrame()
            
        # Default criteria
        if criteria is None:
            criteria = {
                'project': 'ADNI1',
                'modality': 'MRI',
                'image_type': 'T1',
                'group': ['CN', 'AD', 'MCI']
            }
            
        try:
            # Search for subjects
            search_url = f"{self.base_url}/cgi-bin/doSearch"
            
            search_params = {
                'project': criteria.get('project', 'ADNI1'),
                'modality': criteria.get('modality', 'MRI'),
                'image_type': criteria.get('image_type', 'T1'),
                'group[]': criteria.get('group', ['CN', 'AD']),
                'format': 'csv'
            }
            
            response = self.session.get(search_url, params=search_params)
            
            if response.status_code == 200:
                # Parse CSV response
                subjects_df = pd.read_csv(io.StringIO(response.text))
                logger.info(f"Found {len(subjects_df)} subjects matching criteria")
                return subjects_df
            else:
                logger.error(f"Search failed: {response.status_code}")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Subject list download failed: {e}")
            return pd.DataFrame()
            
    def download_images(self, subject_list: pd.DataFrame, 
                       output_dir: Path, max_subjects: int = 100) -> List[Path]:
        """
        Download images for specified subjects
        
        Args:
            subject_list: DataFrame with subject information
            output_dir: Directory to save downloaded images
            max_subjects: Maximum number of subjects to download
            
        Returns:
            List of paths to downloaded images
        """
        if not self.authenticated:
            logger.error("Not authenticated with ADNI")
            return []
            
        output_dir.mkdir(parents=True, exist_ok=True)
        downloaded_files = []
        
        # Limit number of subjects
        subjects_to_download = subject_list.head(max_subjects)
        
        for idx, subject in subjects_to_download.iterrows():
            try:
                subject_id = subject.get('subject_id', f"subject_{idx}")
                image_id = subject.get('image_id', f"image_{idx}")
                
                # Create filename
                filename = f"{subject_id}_{image_id}.nii.gz"
                file_path = output_dir / filename
                
                # Skip if already downloaded
                if file_path.exists():
                    logger.info(f"Skipping {filename}, already exists")
                    downloaded_files.append(file_path)
                    continue
                    
                # Download image
                download_url = f"{self.base_url}/cgi-bin/doDownload"
                download_params = {
                    'image_id': image_id,
                    'format': 'nifti'
                }
                
                response = self.session.get(download_url, params=download_params)
                
                if response.status_code == 200:
                    with open(file_path, 'wb') as f:
                        f.write(response.content)
                    logger.info(f"Downloaded {filename}")
                    downloaded_files.append(file_path)
                else:
                    logger.warning(f"Failed to download {filename}: {response.status_code}")
                    
            except Exception as e:
                logger.error(f"Download failed for subject {subject_id}: {e}")
                
        logger.info(f"Downloaded {len(downloaded_files)} images")
        return downloaded_files
        
    def download_clinical_data(self, subject_list: pd.DataFrame) -> pd.DataFrame:
        """
        Download clinical and cognitive assessment data
        
        Args:
            subject_list: DataFrame with subject information
            
        Returns:
            DataFrame with clinical data
        """
        if not self.authenticated:
            logger.error("Not authenticated with ADNI")
            return pd.DataFrame()
            
        try:
            # Download clinical data CSV
            clinical_url = f"{self.base_url}/cgi-bin/getClinicalData"
            
            # Get unique subject IDs
            subject_ids = subject_list['subject_id'].unique().tolist()
            
            clinical_params = {
                'subject_ids': ','.join(subject_ids),
                'format': 'csv'
            }
            
            response = self.session.get(clinical_url, params=clinical_params)
            
            if response.status_code == 200:
                clinical_df = pd.read_csv(io.StringIO(response.text))
                logger.info(f"Downloaded clinical data for {len(clinical_df)} subjects")
                return clinical_df
            else:
                logger.error(f"Clinical data download failed: {response.status_code}")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Clinical data download failed: {e}")
            return pd.DataFrame()
            
    def create_download_manifest(self, subject_list: pd.DataFrame, 
                                clinical_data: pd.DataFrame) -> Dict:
        """
        Create a manifest file documenting the downloaded data
        
        Args:
            subject_list: DataFrame with subject information
            clinical_data: DataFrame with clinical data
            
        Returns:
            Dictionary with download manifest
        """
        manifest = {
            'download_date': datetime.now().isoformat(),
            'total_subjects': len(subject_list),
            'total_clinical_records': len(clinical_data),
            'subjects': [],
            'clinical_summary': {},
            'data_checksums': {}
        }
        
        # Add subject information
        for _, subject in subject_list.iterrows():
            manifest['subjects'].append({
                'subject_id': subject.get('subject_id'),
                'group': subject.get('group'),
                'age': subject.get('age'),
                'gender': subject.get('gender'),
                'visit': subject.get('visit')
            })
            
        # Add clinical summary
        if not clinical_data.empty:
            manifest['clinical_summary'] = {
                'mmse_stats': clinical_data['mmse'].describe().to_dict() if 'mmse' in clinical_data.columns else {},
                'cdr_stats': clinical_data['cdr'].describe().to_dict() if 'cdr' in clinical_data.columns else {},
                'groups': clinical_data['group'].value_counts().to_dict() if 'group' in clinical_data.columns else {}
            }
            
        return manifest
        
    def validate_download(self, downloaded_files: List[Path], 
                         expected_files: List[str]) -> Dict:
        """
        Validate downloaded files
        
        Args:
            downloaded_files: List of downloaded file paths
            expected_files: List of expected filenames
            
        Returns:
            Validation report dictionary
        """
        validation_report = {
            'total_expected': len(expected_files),
            'total_downloaded': len(downloaded_files),
            'missing_files': [],
            'corrupted_files': [],
            'checksums': {}
        }
        
        downloaded_names = [f.name for f in downloaded_files]
        
        # Check for missing files
        for expected_file in expected_files:
            if expected_file not in downloaded_names:
                validation_report['missing_files'].append(expected_file)
                
        # Validate file integrity
        for file_path in downloaded_files:
            if file_path.exists():
                # Calculate checksum
                with open(file_path, 'rb') as f:
                    file_hash = hashlib.md5(f.read()).hexdigest()
                    validation_report['checksums'][file_path.name] = file_hash
                    
                # Check if file is corrupted (too small or too large)
                file_size = file_path.stat().st_size
                if file_size < 1000:  # Less than 1KB likely corrupted
                    validation_report['corrupted_files'].append(file_path.name)
            else:
                validation_report['missing_files'].append(file_path.name)
                
        return validation_report
        
    def download_demo_data(self, output_dir: Path, num_subjects: int = 10) -> Dict:
        """
        Download demo data for testing purposes
        
        Args:
            output_dir: Directory to save demo data
            num_subjects: Number of demo subjects to create
            
        Returns:
            Dictionary with demo data information
        """
        logger.info(f"Creating demo data with {num_subjects} subjects")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create demo subject list
        demo_subjects = []
        groups = ['CN', 'MCI', 'AD']
        
        for i in range(num_subjects):
            subject_data = {
                'subject_id': f'DEMO_{i:03d}',
                'group': groups[i % len(groups)],
                'age': 65 + (i * 2),
                'gender': i % 2,
                'education': 12 + (i % 9),
                'mmse': 30 - (i * 2),
                'cdr': (i % 4) * 0.5,
                'visit': 'baseline'
            }
            demo_subjects.append(subject_data)
            
        # Create demo images (synthetic data)
        import nibabel as nib
        import numpy as np
        
        demo_images = []
        for subject in demo_subjects:
            # Create synthetic brain image
            synthetic_brain = np.random.randn(256, 256, 256)
            
            # Add some structure to make it brain-like
            x, y, z = np.ogrid[:256, :256, :256]
            center_x, center_y, center_z = 128, 128, 128
            mask = ((x - center_x)**2 + (y - center_y)**2 + (z - center_z)**2) < 80**2
            synthetic_brain[mask] += 0.5
            
            # Save as NIfTI
            image_path = output_dir / f"{subject['subject_id']}_T1.nii.gz"
            nii_img = nib.Nifti1Image(synthetic_brain, np.eye(4))
            nib.save(nii_img, image_path)
            demo_images.append(image_path)
            
        # Create clinical data CSV
        clinical_df = pd.DataFrame(demo_subjects)
        clinical_path = output_dir / "demo_clinical_data.csv"
        clinical_df.to_csv(clinical_path, index=False)
        
        # Create manifest
        manifest = {
            'type': 'demo_data',
            'created_date': datetime.now().isoformat(),
            'num_subjects': len(demo_subjects),
            'subjects': demo_subjects,
            'files_created': {
                'images': [str(p) for p in demo_images],
                'clinical_data': str(clinical_path)
            }
        }
        
        manifest_path = output_dir / "demo_manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
            
        logger.info(f"Demo data created in {output_dir}")
        return manifest

# Example usage
if __name__ == "__main__":
    # Initialize downloader (credentials would be provided in production)
    downloader = ADNIDownloader()
    
    # Create demo data for testing
    demo_dir = Path("data/demo")
    demo_manifest = downloader.download_demo_data(demo_dir, num_subjects=5)
    
    print("Demo data created successfully!")
    print(f"Subjects: {demo_manifest['num_subjects']}")
    print(f"Files created: {len(demo_manifest['files_created']['images'])} images")
    
    # In production, you would use:
    # if downloader.authenticate():
    #     subjects = downloader.download_subject_list()
    #     images = downloader.download_images(subjects, Path("data/adni"))
    #     clinical = downloader.download_clinical_data(subjects)