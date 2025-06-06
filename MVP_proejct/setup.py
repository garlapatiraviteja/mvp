# setup.py - Installation and setup script for Factory AI MVP

import os
import sys
import subprocess
import json
from pathlib import Path
import requests
import zipfile
import shutil

def create_directory_structure():
    """Create required directory structure"""
    directories = [
        "models/base",
        "models/sectors", 
        "config",
        "data/images",
        "data/videos",
        "logs",
        "exports",
        "temp"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created directory: {directory}")

def install_requirements():
    """Install required Python packages"""
    requirements = [
        "streamlit>=1.28.0",
        "opencv-python>=4.8.0",
        "torch>=2.0.0",
        "torchvision>=0.15.0", 
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "plotly>=5.15.0",
        "Pillow>=10.0.0",
        "scikit-learn>=1.3.0",
        "requests>=2.31.0",
        "python-dotenv>=1.0.0"
    ]
    
    print("Installing required packages...")
    for package in requirements:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✓ Installed: {package}")
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to install {package}: {e}")
            return False
    return True

def download_base_models():
    """Download base pre-trained models"""
    models_info = {
        "efficientnet_b0.pth": {
            "url": "https://download.pytorch.org/models/efficientnet_b0_rwightman-3dd342df.pth",
            "path": "models/base/efficientnet_b0.pth"
        }
    }
    
    print("Downloading base models...")
    for model_name, info in models_info.items():
        model_path = Path(info["path"])
        if not model_path.exists():
            try:
                print(f"Downloading {model_name}...")
                response = requests.get(info["url"], stream=True)
                response.raise_for_status()
                
                with open(model_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                print(f"✓ Downloaded: {model_name}")
            except Exception as e:
                print(f"✗ Failed to download {model_name}: {e}")
        else:
            print(f"✓ Model already exists: {model_name}")

def create_config_files():
    """Create default configuration files"""
    # Model configuration
    model_config = {
        "models": {
            "electric_cable": {
                "type": "efficientdet",
                "confidence_threshold": 0.7,
                "nms_threshold": 0.4,
                "input_size": 512,
                "defect_classes": {
                    "0": "Cable Deformation",
                    "1": "Insulation Damage",
                    "2": "Length Mismatch", 
                    "3": "Connector Defects",
                    "4": "Surface Scratches"
                }
            },
            "seed_packaging": {
                "type": "efficientdet", 
                "confidence_threshold": 0.75,
                "nms_threshold": 0.4,
                "input_size": 512,
                "defect_classes": {
                    "0": "Torn Packets",
                    "1": "Missing Labels",
                    "2": "Improper Sealing",
                    "3": "Contamination", 
                    "4": "Under Filled Packets"
                }
            }
        },
        "system": {
            "device": "auto",
            "max_detections": 100,
            "save_detection_images": True,
            "detection_history_limit": 10000
        },
        "alerts": {
            "email_enabled": False,
            "high_defect_threshold": 10,
            "confidence_drop_threshold": 0.6
        }
    }
    
    with open("config/model_config.json", "w") as f:
        json.dump(model_config, f, indent=2)
    print("✓ Created model configuration")
    
    # Camera configuration
    camera_config = {
        "default_camera": 0,
        "resolution": [1920, 1080],
        "fps": 30,
        "auto_discovery": True,
        "supported_formats": ["RTSP", "USB", "IP"]
    }
    
    with open("config/camera_config.json", "w") as f:
        json.dump(camera_config, f, indent=2)
    print("✓ Created camera configuration")
    
    # Environment configuration
    env_content = """# Factory AI Environment Configuration
STREAMLIT_PORT=8501
STREAMLIT_HOST=0.0.0.0
LOG_LEVEL=INFO
DETECTION_SAVE_PATH=exports/
EMAIL_ENABLED=false
SLACK_ENABLED=false
"""
    
    with open(".env", "w") as f:
        f.write(env_content)
    print("✓ Created environment configuration")

def create_startup_scripts():
    """Create startup scripts for different platforms"""
    
    # Windows batch script
    windows_script = """@echo off
echo Starting Factory AI Quality Control System...
echo.
echo Checking Python installation...
python --version
if errorlevel 1 (
    echo Python is not installed or not in PATH
    pause
    exit /b 1
)

echo.
echo Starting Streamlit application...
streamlit run main.py --server.port=8501 --server.headless=true

pause
"""
    
    with open("start_windows.bat", "w") as f:
        f.write(windows_script)
    print("✓ Created Windows startup script")
    
    # Linux/Mac shell script
    unix_script = """#!/bin/bash
echo "Starting Factory AI Quality Control System..."
echo ""

# Check Python installation
if ! command -v python3 &> /dev/null; then
    echo "Python3 is not installed"
    exit 1
fi

echo "Python version:"
python3 --version
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install/update requirements
echo "Installing requirements..."
pip install -r requirements.txt

echo ""
echo "Starting Streamlit application..."
streamlit run main.py --server.port=8501 --server.headless=true
"""
    
    with open("start_unix.sh", "w") as f:
        f.write(unix_script)
    
    # Make executable
    os.chmod("start_unix.sh", 0o755)
    print("✓ Created Unix startup script")

def create_requirements_file():
    """Create requirements.txt file"""
    requirements_content = """streamlit>=1.28.0
opencv-python>=4.8.0
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
pandas>=2.0.0
plotly>=5.15.0
Pillow>=10.0.0
scikit-learn>=1.3.0
requests>=2.31.0
python-dotenv>=1.0.0
"""
    
    with open("requirements.txt", "w") as f:
        f.write(requirements_content)
    print("✓ Created requirements.txt")

def create_docker_files():
    """Create Docker configuration files"""
    
    # Dockerfile
    dockerfile_content = """FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    libglib2.0-0 \\
    libsm6 \\
    libxext6 \\
    libxrender-dev \\
    libgomp1 \\
    libglib2.0-0 \\
    libgtk-3-0 \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p models/base models/sectors config data/images data/videos logs exports temp

# Expose port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Start application
CMD ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]
"""
    
    with open("Dockerfile", "w") as f:
        f.write(dockerfile_content)
    print("✓ Created Dockerfile")
    
    # Docker Compose
    docker_compose_content = """version: '3.8'

services:
  factory-ai:
    build: .
    ports:
      - "8501:8501"
    volumes:
      - ./data:/app/data
      - ./models:/app/models
      - ./config:/app/config
      - ./logs:/app/logs
      - ./exports:/app/exports
    environment:
      - STREAMLIT_PORT=8501
      - STREAMLIT_HOST=0.0.0.0
      - LOG_LEVEL=INFO
    restart: unless-stopped
    
  # Optional: Add database for production
  # postgres:
  #   image: postgres:13
  #   environment:
  #     POSTGRES_DB: factory_ai
  #     POSTGRES_USER: factory_user
  #     POSTGRES_PASSWORD: secure_password
  #   volumes:
  #     - postgres_data:/var/lib/postgresql/data
  #   ports:
  #     - "5432:5432"

# volumes:
#   postgres_data:
"""
    
    with open("docker-compose.yml", "w") as f:
        f.write(docker_compose_content)
    print("✓ Created docker-compose.yml")

def create_readme():
    """Create comprehensive README file"""
    readme_content = """# Factory AI - Quality Control MVP

🏭 **Plug-and-Play AI Quality Control for Manufacturing**

Transform your factory quality control from manual inspection to AI-powered automation with zero technical setup required.

## 🚀 Quick Start

### Option 1: Direct Python Installation

1. **Clone or download** this repository
2. **Run setup script:**
   ```bash
   python setup.py
   ```
3. **Start the application:**
   - Windows: Double-click `start_windows.bat`
   - Linux/Mac: Run `./start_unix.sh`
4. **Open browser** to `http://localhost:8501`

### Option 2: Docker Installation

```bash
# Build and run with Docker Compose
docker-compose up --build

# Access at http://localhost:8501
```

## 📋 Supported Manufacturing Sectors

### Phase 1 (Current MVP)
- **Electric Cable Manufacturing**: Cable deformation, insulation damage, length mismatch, connector defects
- **Seed Packaging**: Torn packets, missing labels, improper sealing, contamination

### Phase 2 (Coming Soon)
- Automotive parts manufacturing
- Electronics assembly
- Textile manufacturing
- Food packaging

## 🎯 Features

- **Real-time Detection**: Live camera feed processing
- **Multi-Sector Support**: Switch between different manufacturing types
- **Smart Analytics**: Defect trends and performance metrics
- **Alert System**: Immediate notifications for quality issues
- **Easy Setup**: Plug-and-play with existing CCTV cameras

## 📊 System Requirements

### Minimum Requirements
- **OS**: Windows 10/11, Linux, macOS
- **RAM**: 4GB (8GB recommended)
- **CPU**: Intel i5 or equivalent
- **Camera**: USB camera or IP camera with RTSP support

### Recommended Requirements
- **RAM**: 8GB or more
- **GPU**: NVIDIA GPU with CUDA support (optional, for faster processing)
- **Storage**: 10GB free space

## 🔧 Configuration

### Camera Setup
1. Connect USB camera or configure IP camera
2. Update `config/camera_config.json` with your camera settings
3. Test camera connection in the dashboard

### Model Configuration
- Edit `config/model_config.json` to adjust detection thresholds
- Customize defect types for your specific needs
- Set alert thresholds and notification preferences

## 📱 Usage Guide

### 1. Sector Selection
- Choose your manufacturing sector from the sidebar
- System automatically loads appropriate detection models

### 2. Camera Activation
- Click "Start Camera" to begin live detection
- View real-time defect detection results
- Monitor detection confidence scores

### 3. Analytics Dashboard
- View defect distribution charts
- Track defect trends over time
- Export reports for quality control documentation

### 4. Alert Management
- Configure alert thresholds
- Set up email notifications (optional)
- Review detection history

## 🛠️ Troubleshooting

### Common Issues

**Camera not detected:**
- Check USB connection or IP camera network settings
- Verify camera permissions in system settings
- Try different camera sources (0, 1, 2, etc.)

**Low detection accuracy:**
- Ensure proper lighting conditions
- Clean camera lens
- Adjust detection sensitivity in settings

**Performance issues:**
- Close other applications using camera
- Reduce video resolution in camera settings
- Consider using GPU acceleration if available

## 📞 Support

### Getting Help
- Check the troubleshooting section above
- Review system logs in the `logs/` directory
- Contact support with system configuration details

### Reporting Issues
When reporting issues, please include:
- Operating system and version
- Camera model and connection type
- Error messages from logs
- Steps to reproduce the issue

## 🔄 Updates and Expansion

### Automatic Updates
- System checks for model updates automatically
- New sector support added through configuration updates

### Custom Sector Development
- Contact support for custom manufacturing sector development
- Provide sample images and defect specifications
- Training typically takes 1-2 weeks for new sectors

## 📈 Roadmap

### Short Term (Next 3 months)
- Additional manufacturing sectors
- Mobile app for remote monitoring
- Enhanced analytics and reporting

### Long Term (6-12 months)
- Predictive maintenance features
- ERP/MES system integration
- Advanced AI features and optimization

---

**Factory AI MVP v1.0**  
*Transforming Manufacturing Quality Control with AI*
"""
    
    with open("README.md", "w") as f:
        f.write(readme_content)
    print("✓ Created README.md")

def main():
    """Main setup function"""
    print("=" * 60)
    print("    Factory AI - Quality Control MVP Setup")
    print("=" * 60)
    print("")
    
    try:
        # Create directory structure
        print("1. Creating directory structure...")
        create_directory_structure()
        print("")
        
        # Create configuration files
        print("2. Creating configuration files...")
        create_config_files()
        print("")
        
        # Create requirements file
        print("3. Creating requirements file...")
        create_requirements_file()
        print("")
        
        # Create startup scripts
        print("4. Creating startup scripts...")
        create_startup_scripts()
        print("")
        
        # Create Docker files
        print("5. Creating Docker configuration...")
        create_docker_files()
        print("")
        
        # Create README
        print("6. Creating documentation...")
        create_readme()
        print("")
        
        # Optional: Install requirements
        install_choice = input("Install Python requirements now? (y/n): ").lower().strip()
        if install_choice == 'y':
            print("\n7. Installing Python requirements...")
            if install_requirements():
                print("✓ All requirements installed successfully")
            else:
                print("✗ Some requirements failed to install")
        
        # Optional: Download base models
        download_choice = input("\nDownload base models now? (y/n): ").lower().strip()
        if download_choice == 'y':
            print("\n8. Downloading base models...")
            download_base_models()
        
        print("\n" + "=" * 60)
        print("    Setup Complete!")
        print("=" * 60)
        print("")
        print("Next steps:")
        print("1. Run startup script:")
        print("   - Windows: start_windows.bat")
        print("   - Linux/Mac: ./start_unix.sh")
        print("   - Docker: docker-compose up")
        print("")
        print("2. Open browser to: http://localhost:8501")
        print("")
        print("3. Connect camera and select manufacturing sector")
        print("")
        print("🎉 Your Factory AI system is ready to use!")
        
    except Exception as e:
        print(f"\n✗ Setup failed: {e}")
        return False
        
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)