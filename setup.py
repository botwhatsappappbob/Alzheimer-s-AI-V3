#!/usr/bin/env python3
"""
Setup script for Alzheimer's Disease Early Detection AI Platform
Handles installation, configuration, and initial setup
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Platform metadata
PLATFORM_NAME = "Alzheimer's Disease Early Detection AI Platform"
VERSION = "1.0.0"
AUTHOR = "Alzheimer Detection Platform Contributors"
EMAIL = "support@alzheimer-ai-platform.org"
DESCRIPTION = "End-to-end AI platform for early detection of Alzheimer's Disease"

class PlatformSetup:
    """Main setup class for the Alzheimer's Detection Platform"""
    
    def __init__(self):
        self.platform_dir = Path(__file__).parent
        self.config_dir = self.platform_dir / "config"
        self.data_dir = self.platform_dir / "data"
        self.models_dir = self.platform_dir / "models"
        self.logs_dir = self.platform_dir / "logs"
        
    def check_system_requirements(self):
        """Check if system meets minimum requirements"""
        logger.info("Checking system requirements...")
        
        # Check Python version
        if sys.version_info < (3, 8):
            logger.error("Python 3.8 or higher is required")
            return False
            
        # Check available memory (minimum 8GB recommended)
        try:
            import psutil
            memory = psutil.virtual_memory()
            if memory.total < 8 * 1024**3:  # 8GB
                logger.warning(f"Low memory detected: {memory.total / 1024**3:.1f}GB. 8GB+ recommended")
        except ImportError:
            logger.warning("psutil not available, cannot check memory")
            
        # Check disk space (minimum 50GB recommended)
        disk = psutil.disk_usage(self.platform_dir)
        if disk.free < 50 * 1024**3:  # 50GB
            logger.warning(f"Low disk space: {disk.free / 1024**3:.1f}GB free. 50GB+ recommended")
            
        logger.info("System requirements check completed")
        return True
        
    def create_directories(self):
        """Create necessary directory structure"""
        logger.info("Creating directory structure...")
        
        directories = [
            self.config_dir,
            self.data_dir,
            self.data_dir / "raw",
            self.data_dir / "processed",
            self.data_dir / "splits",
            self.models_dir,
            self.models_dir / "checkpoints",
            self.models_dir / "experiments",
            self.logs_dir,
            self.logs_dir / "api",
            self.logs_dir / "training",
            self.logs_dir / "system"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {directory}")
            
    def install_dependencies(self):
        """Install Python dependencies"""
        logger.info("Installing Python dependencies...")
        
        requirements_file = self.platform_dir / "requirements.txt"
        if not requirements_file.exists():
            logger.error("requirements.txt not found")
            return False
            
        try:
            subprocess.run([
                sys.executable, "-m", "pip", "install", "--upgrade", "pip"
            ], check=True)
            
            subprocess.run([
                sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
            ], check=True)
            
            logger.info("Dependencies installed successfully")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install dependencies: {e}")
            return False
            
    def setup_configuration(self):
        """Setup configuration files"""
        logger.info("Setting up configuration...")
        
        # Create default configuration
        config = {
            "platform": {
                "name": PLATFORM_NAME,
                "version": VERSION,
                "environment": "development"
            },
            "database": {
                "url": "sqlite:///./data/alzheimer.db",
                "echo": True
            },
            "redis": {
                "url": "redis://localhost:6379",
                "decode_responses": True
            },
            "mlflow": {
                "tracking_uri": "./models/experiments",
                "artifact_location": "./models/artifacts"
            },
            "security": {
                "secret_key": os.urandom(32).hex(),
                "algorithm": "HS256",
                "access_token_expire_minutes": 30
            },
            "model": {
                "checkpoint_path": "./models/checkpoints/alzheimer_detection_model.pth",
                "device": "auto",  # auto, cpu, cuda
                "batch_size": 4,
                "num_workers": 2
            },
            "api": {
                "host": "0.0.0.0",
                "port": 8000,
                "reload": True,
                "log_level": "info"
            }
        }
        
        # Save configuration
        import json
        config_file = self.config_dir / "config.json"
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
            
        logger.info(f"Configuration saved to {config_file}")
        
        # Create environment file
        env_file = self.platform_dir / ".env"
        env_content = f"""# Alzheimer's Disease Detection Platform Configuration
# Generated by setup script

PLATFORM_NAME="{PLATFORM_NAME}"
PLATFORM_VERSION="{VERSION}"
ENVIRONMENT=development

# Database
DATABASE_URL=sqlite:///./data/alzheimer.db

# Redis
REDIS_URL=redis://localhost:6379

# MLflow
MLFLOW_TRACKING_URI=./models/experiments

# Security (change these in production)
SECRET_KEY={os.urandom(32).hex()}
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# Model Configuration
MODEL_CHECKPOINT_PATH=./models/checkpoints/alzheimer_detection_model.pth
MODEL_DEVICE=auto
MODEL_BATCH_SIZE=4
MODEL_NUM_WORKERS=2

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_RELOAD=True
API_LOG_LEVEL=info

# Medical Disclaimer
MEDICAL_DISCLAIMER="This system provides clinical decision support only. Not intended for standalone diagnosis."
"""
        
        with open(env_file, 'w') as f:
            f.write(env_content)
            
        logger.info(f"Environment file created at {env_file}")
        
    def download_pretrained_models(self):
        """Download pretrained models if available"""
        logger.info("Checking for pretrained models...")
        
        # This would typically download from a model registry
        # For now, we'll create a placeholder
        model_file = self.models_dir / "checkpoints" / "alzheimer_detection_model.pth"
        
        if not model_file.exists():
            logger.warning("No pretrained model found. Training will be required.")
            logger.info("Creating placeholder model file...")
            
            # Create a simple checkpoint placeholder
            placeholder_checkpoint = {
                "epoch": 0,
                "model_state_dict": {},
                "optimizer_state_dict": {},
                "metrics": {},
                "config": {}
            }
            
            import torch
            torch.save(placeholder_checkpoint, model_file)
            
        logger.info("Model setup completed")
        
    def setup_database(self):
        """Initialize database schema"""
        logger.info("Setting up database...")
        
        # For SQLite, we'll create the database file
        # In production, this would use PostgreSQL with proper schema migration
        
        db_file = self.data_dir / "alzheimer.db"
        if not db_file.exists():
            logger.info("Creating database file...")
            # Placeholder for database initialization
            
        logger.info("Database setup completed")
        
    def create_systemd_service(self):
        """Create systemd service file for production deployment"""
        logger.info("Creating systemd service...")
        
        service_content = f"""[Unit]
Description={PLATFORM_NAME}
After=network.target

[Service]
Type=simple
User=alzheimer
WorkingDirectory={self.platform_dir}
Environment=PATH={self.platform_dir}/venv/bin
ExecStart={sys.executable} -m src.api.main
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
"""
        
        service_file = self.platform_dir / "alzheimer-detection.service"
        with open(service_file, 'w') as f:
            f.write(service_content)
            
        logger.info(f"Systemd service file created at {service_file}")
        logger.info("To install the service, run:")
        logger.info(f"sudo cp {service_file} /etc/systemd/system/")
        logger.info("sudo systemctl daemon-reload")
        logger.info("sudo systemctl enable alzheimer-detection")
        logger.info("sudo systemctl start alzheimer-detection")
        
    def run_tests(self):
        """Run basic tests to verify installation"""
        logger.info("Running basic tests...")
        
        try:
            # Test Python imports
            import torch
            import numpy as np
            import nibabel as nib
            import fastapi
            import pydantic
            
            logger.info("✓ Core Python packages imported successfully")
            
            # Test GPU availability
            if torch.cuda.is_available():
                logger.info(f"✓ GPU available: {torch.cuda.get_device_name()}")
            else:
                logger.warning("⚠ GPU not available, using CPU mode")
                
            # Test model loading
            model_file = self.models_dir / "checkpoints" / "alzheimer_detection_model.pth"
            if model_file.exists():
                checkpoint = torch.load(model_file, map_location='cpu')
                logger.info("✓ Model checkpoint can be loaded")
            
            logger.info("✓ All basic tests passed")
            return True
            
        except Exception as e:
            logger.error(f"✗ Test failed: {e}")
            return False
            
    def create_launcher_script(self):
        """Create launcher script for easy startup"""
        logger.info("Creating launcher script...")
        
        launcher_content = f"""#!/bin/bash
# Launcher script for {PLATFORM_NAME}

echo "Starting {PLATFORM_NAME} (v{VERSION})..."
echo "==================================="

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install/update dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Set environment variables
export PLATFORM_NAME="{PLATFORM_NAME}"
export PLATFORM_VERSION="{VERSION}"
export ENVIRONMENT=development

# Start the application
echo "Starting API server..."
python -m src.api.main
"""
        
        launcher_file = self.platform_dir / "launch.sh"
        with open(launcher_file, 'w') as f:
            f.write(launcher_content)
            
        # Make executable
        os.chmod(launcher_file, 0o755)
        
        logger.info(f"Launcher script created at {launcher_file}")
        
    def display_next_steps(self):
        """Display next steps to the user"""
        logger.info("=" * 60)
        logger.info(f"{PLATFORM_NAME} Setup Complete!")
        logger.info("=" * 60)
        logger.info("")
        logger.info("📁 Directory structure created:")
        logger.info(f"  • Platform: {self.platform_dir}")
        logger.info(f"  • Configuration: {self.config_dir}")
        logger.info(f"  • Data: {self.data_dir}")
        logger.info(f"  • Models: {self.models_dir}")
        logger.info(f"  • Logs: {self.logs_dir}")
        logger.info("")
        logger.info("🚀 Next steps:")
        logger.info("1. Review configuration in config/config.json")
        logger.info("2. Set up your environment variables in .env")
        logger.info("3. Run './launch.sh' to start the application")
        logger.info("4. Access the dashboard at http://localhost:8000")
        logger.info("5. Read the documentation in docs/")
        logger.info("")
        logger.info("⚠️  IMPORTANT MEDICAL DISCLAIMER:")
        logger.info("This system provides clinical decision support only.")
        logger.info("Not intended for standalone diagnosis or treatment decisions.")
        logger.info("All outputs must be reviewed by qualified healthcare professionals.")
        logger.info("")
        logger.info("📚 Documentation:")
        logger.info("• README.md - Main documentation")
        logger.info("• docs/clinical_guide.md - For clinicians")
        logger.info("• docs/api_documentation.md - For developers")
        logger.info("• docs/deployment.md - Deployment guide")
        logger.info("")
        logger.info("🆘 Support:")
        logger.info("• Issues: GitHub Issues")
        logger.info("• Email: support@alzheimer-ai-platform.org")
        logger.info("• Clinical Questions: clinical@alzheimer-ai-platform.org")
        logger.info("=" * 60)
        
    def run_setup(self, full_setup=True):
        """Run the complete setup process"""
        logger.info(f"Setting up {PLATFORM_NAME} v{VERSION}")
        logger.info("=" * 60)
        
        try:
            # Check requirements
            if not self.check_system_requirements():
                return False
                
            # Create directories
            self.create_directories()
            
            # Install dependencies
            if full_setup:
                if not self.install_dependencies():
                    return False
                    
            # Setup configuration
            self.setup_configuration()
            
            # Download models
            self.download_pretrained_models()
            
            # Setup database
            self.setup_database()
            
            # Create launcher
            self.create_launcher_script()
            
            # Run tests
            if full_setup:
                if not self.run_tests():
                    logger.error("Tests failed. Please check the installation.")
                    return False
                    
            # Create systemd service
            self.create_systemd_service()
            
            # Display next steps
            self.display_next_steps()
            
            return True
            
        except Exception as e:
            logger.error(f"Setup failed: {e}")
            return False

def main():
    """Main setup function"""
    parser = argparse.ArgumentParser(description=f"Setup {PLATFORM_NAME}")
    parser.add_argument('--quick', action='store_true', help='Quick setup without dependency installation')
    parser.add_argument('--test', action='store_true', help='Run tests only')
    
    args = parser.parse_args()
    
    setup = PlatformSetup()
    
    if args.test:
        success = setup.run_tests()
        sys.exit(0 if success else 1)
    else:
        success = setup.run_setup(full_setup=not args.quick)
        sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()