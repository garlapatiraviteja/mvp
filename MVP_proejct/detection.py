# detection_models.py - Core detection models for Factory AI MVP

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import efficientnet_b0, efficientnet_b3
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
import json
from pathlib import Path
import logging
from abc import ABC, abstractmethod
import time

logger = logging.getLogger(__name__)

class BaseDetectionModel(ABC):
    """Base class for all detection models"""
    
    def __init__(self, model_path: str = None, device: str = "cpu"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = None
        self.transforms = None
        self.is_loaded = False
        
    @abstractmethod
    def load_model(self, model_path: str):
        """Load the detection model"""
        pass
    
    @abstractmethod
    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image for model input"""
        pass
    
    @abstractmethod
    def detect(self, image: np.ndarray) -> List[Dict]:
        """Run detection on image"""
        pass

class EfficientDetModel(BaseDetectionModel):
    """EfficientDet-based detection model"""
    
    def __init__(self, model_type: str = "d0", num_classes: int = 5, **kwargs):
        super().__init__(**kwargs)
        self.model_type = model_type
        self.num_classes = num_classes
        self.confidence_threshold = 0.5
        self.nms_threshold = 0.4
        
        # Initialize transforms
        self.transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
    def load_model(self, model_path: str = None):
        """Load EfficientDet model"""
        try:
            if model_path and Path(model_path).exists():
                # Load custom trained model
                checkpoint = torch.load(model_path, map_location=self.device)
                self.model = self._create_model()
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                # Use mock model for MVP
                self.model = self._create_mock_model()
                
            self.model.to(self.device)
            self.model.eval()
            self.is_loaded = True
            logger.info(f"EfficientDet model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading EfficientDet model: {e}")
            self.is_loaded = False
            
    def _create_mock_model(self):
        """Create mock model for MVP demonstration"""
        # Use EfficientNet backbone as mock detector
        backbone = efficientnet_b0(pretrained=True)
        backbone.classifier = nn.Linear(backbone.classifier.in_features, self.num_classes)
        return backbone
        
    def _create_model(self):
        """Create actual EfficientDet model architecture"""
        # This would be the actual EfficientDet implementation
        # For MVP, using simplified version
        return self._create_mock_model()
        
    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image for EfficientDet"""
        if len(image.shape) == 3 and image.shape[2] == 3:
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        tensor = self.transforms(image)
        return tensor.unsqueeze(0)  # Add batch dimension
        
    def detect(self, image: np.ndarray) -> List[Dict]:
        """Run detection on image"""
        if not self.is_loaded:
            logger.warning("Model not loaded, loading mock model")
            self.load_model()
            
        try:
            # Preprocess
            input_tensor = self.preprocess_image(image).to(self.device)
            
            # Mock detection for MVP
            detections = self._mock_detection(image)
            return detections
            
        except Exception as e:
            logger.error(f"Detection error: {e}")
            return []
            
    def _mock_detection(self, image: np.ndarray) -> List[Dict]:
        """Mock detection results for MVP"""
        detections = []
        
        # Simulate random detections
        if np.random.random() > 0.6:  # 40% chance of detection
            h, w = image.shape[:2]
            
            # Random bounding box
            x1 = np.random.randint(0, w//2)
            y1 = np.random.randint(0, h//2)
            x2 = min(x1 + np.random.randint(50, w//3), w)
            y2 = min(y1 + np.random.randint(50, h//3), h)
            
            detection = {
                'bbox': [x1, y1, x2, y2],
                'confidence': np.random.uniform(0.5, 0.95),
                'class_id': np.random.randint(0, self.num_classes),
                'class_name': f'defect_{np.random.randint(0, self.num_classes)}'
            }
            detections.append(detection)
            
        return detections

class SectorSpecificDetector:
    """Sector-specific detection logic"""
    
    def __init__(self):
        self.sectors = {
            'electric_cable': ElectricCableDetector(),
            'seed_packaging': SeedPackagingDetector()
        }
        
    def get_detector(self, sector: str):
        """Get detector for specific sector"""
        return self.sectors.get(sector)

class ElectricCableDetector:
    """Electric cable manufacturing defect detection"""
    
    def __init__(self):
        self.defect_types = {
            0: "Cable Deformation",
            1: "Insulation Damage", 
            2: "Length Mismatch",
            3: "Connector Defects",
            4: "Surface Scratches"
        }
        self.model = EfficientDetModel(num_classes=len(self.defect_types))
        
    def detect_defects(self, image: np.ndarray) -> List[Dict]:
        """Detect cable manufacturing defects"""
        detections = self.model.detect(image)
        
        # Process detections specific to cable manufacturing
        processed_detections = []
        for det in detections:
            processed_det = {
                'defect_type': self.defect_types.get(det['class_id'], 'Unknown'),
                'confidence': det['confidence'],
                'bbox': det['bbox'],
                'severity': self._assess_severity(det),
                'recommendation': self._get_recommendation(det['class_id'])
            }
            processed_detections.append(processed_det)
            
        return processed_detections
        
    def _assess_severity(self, detection: Dict) -> str:
        """Assess defect severity"""
        confidence = detection['confidence']
        if confidence > 0.8:
            return "High"
        elif confidence > 0.6:
            return "Medium"
        else:
            return "Low"
            
    def _get_recommendation(self, class_id: int) -> str:
        """Get recommendation for defect type"""
        recommendations = {
            0: "Check cable formation process parameters",
            1: "Inspect insulation material quality and application",
            2: "Calibrate length cutting mechanism",
            3: "Verify connector assembly process",
            4: "Review handling and transport procedures"
        }
        return recommendations.get(class_id, "Manual inspection required")

class SeedPackagingDetector:
    """Seed packaging defect detection"""
    
    def __init__(self):
        self.defect_types = {
            0: "Torn Packets",
            1: "Missing Labels",
            2: "Improper Sealing", 
            3: "Contamination",
            4: "Under Filled Packets"
        }
        self.model = EfficientDetModel(num_classes=len(self.defect_types))
        
    def detect_defects(self, image: np.ndarray) -> List[Dict]:
        """Detect seed packaging defects"""
        detections = self.model.detect(image)
        
        # Process detections specific to seed packaging
        processed_detections = []
        for det in detections:
            processed_det = {
                'defect_type': self.defect_types.get(det['class_id'], 'Unknown'),
                'confidence': det['confidence'],
                'bbox': det['bbox'],
                'severity': self._assess_severity(det),
                'recommendation': self._get_recommendation(det['class_id'])
            }
            processed_detections.append(processed_det)
            
        return processed_detections
        
    def _assess_severity(self, detection: Dict) -> str:
        """Assess defect severity for packaging"""
        confidence = detection['confidence']
        class_id = detection.get('class_id', 0)
        
        # Higher severity for contamination and sealing issues
        if class_id in [2, 3]:  # Sealing, Contamination
            return "High" if confidence > 0.6 else "Medium"
        else:
            return "High" if confidence > 0.8 else "Medium" if confidence > 0.6 else "Low"
            
    def _get_recommendation(self, class_id: int) -> str:
        """Get recommendation for packaging defect"""
        recommendations = {
            0: "Check packaging material quality and handling",
            1: "Verify label application process and adhesive",
            2: "Inspect sealing temperature and pressure settings",
            3: "Review cleaning protocols and material sources",
            4: "Calibrate filling mechanism and weight sensors"
        }
        return recommendations.get(class_id, "Quality control inspection required")

class AnomalyDetectionEngine:
    """Rule-based and statistical anomaly detection"""
    
    def __init__(self):
        self.detection_history = []
        self.alert_thresholds = {
            'electric_cable': 3,  # Alert after 3 defects
            'seed_packaging': 5   # Alert after 5 defects
        }
        
    def add_detection(self, detection: Dict, sector: str):
        """Add new detection to history"""
        detection['sector'] = sector
        detection['timestamp'] = np.datetime64('now')
        self.detection_history.append(detection)
        
        # Keep only last 1000 detections
        if len(self.detection_history) > 1000:
            self.detection_history = self.detection_history[-1000:]
            
    def check_anomalies(self, sector: str) -> List[Dict]:
        """Check for anomalous patterns"""
        anomalies = []
        
        # Get recent detections for sector
        recent_detections = [d for d in self.detection_history[-50:] 
                           if d.get('sector') == sector]
        
        if len(recent_detections) >= self.alert_thresholds[sector]:
            anomalies.append({
                'type': 'high_defect_rate',
                'message': f'High defect rate detected in {sector}',
                'count': len(recent_detections),
                'severity': 'high'
            })
            
        # Check for confidence drops
        if recent_detections:
            avg_confidence = np.mean([d['confidence'] for d in recent_detections])
            if avg_confidence < 0.6:
                anomalies.append({
                    'type': 'low_confidence',
                    'message': 'Detection confidence dropping - possible environmental changes',
                    'avg_confidence': avg_confidence,
                    'severity': 'medium'
                })
                
        return anomalies

class ModelManager:
    """Manage multiple detection models"""
    
    def __init__(self):
        self.models = {}
        self.sector_detector = SectorSpecificDetector()
        self.anomaly_engine = AnomalyDetectionEngine()
        
    def load_sector_models(self):
        """Load all sector-specific models"""
        try:
            for sector in ['electric_cable', 'seed_packaging']:
                detector = self.sector_detector.get_detector(sector)
                if detector:
                    detector.model.load_model()
                    self.models[sector] = detector
                    logger.info(f"Loaded model for {sector}")
        except Exception as e:
            logger.error(f"Error loading sector models: {e}")
            
    def detect_defects(self, image: np.ndarray, sector: str) -> Dict:
        """Run defect detection for specific sector"""
        if sector not in self.models:
            logger.warning(f"Model for sector {sector} not loaded")
            return {'detections': [], 'anomalies': []}
            
        try:
            # Run detection
            detector = self.models[sector]
            detections = detector.detect_defects(image)
            
            # Add to anomaly detection
            for detection in detections:
                self.anomaly_engine.add_detection(detection, sector)
                
            # Check for anomalies
            anomalies = self.anomaly_engine.check_anomalies(sector)
            
            return {
                'detections': detections,
                'anomalies': anomalies,
                'sector': sector,
                'timestamp': str(np.datetime64('now'))
            }
            
        except Exception as e:
            logger.error(f"Detection error for {sector}: {e}")
            return {'detections': [], 'anomalies': [], 'error': str(e)}
            
    def get_model_stats(self) -> Dict:
        """Get statistics about loaded models"""
        stats = {
            'loaded_models': list(self.models.keys()),
            'total_detections': len(self.anomaly_engine.detection_history),
            'model_status': {}
        }
        
        for sector, model in self.models.items():
            stats['model_status'][sector] = {
                'loaded': model.model.is_loaded if hasattr(model.model, 'is_loaded') else True,
                'device': str(getattr(model.model, 'device', 'cpu'))
            }
            
        return stats

# Configuration and utility functions
def create_model_config():
    """Create default model configuration"""
    config = {
        'models': {
            'electric_cable': {
                'type': 'efficientdet',
                'model_path': 'models/sectors/electric_cable_v1.pt',
                'confidence_threshold': 0.7,
                'nms_threshold': 0.4,
                'input_size': 512,
                'defect_classes': {
                    0: "Cable Deformation",
                    1: "Insulation Damage", 
                    2: "Length Mismatch",
                    3: "Connector Defects",
                    4: "Surface Scratches"
                }
            },
            'seed_packaging': {
                'type': 'efficientdet',
                'model_path': 'models/sectors/seed_packaging_v1.pt',
                'confidence_threshold': 0.75,
                'nms_threshold': 0.4,
                'input_size': 512,
                'defect_classes': {
                    0: "Torn Packets",
                    1: "Missing Labels",
                    2: "Improper Sealing", 
                    3: "Contamination",
                    4: "Under Filled Packets"
                }
            }
        },
        'system': {
            'device': 'auto',  # auto, cpu, cuda
            'batch_size': 1,
            'max_detections': 100,
            'save_detection_images': True,
            'detection_history_limit': 10000
        },
        'alerts': {
            'email_enabled': True,
            'slack_enabled': False,
            'high_defect_threshold': 10,
            'confidence_drop_threshold': 0.6
        }
    }
    return config

def save_config(config: Dict, config_path: str = "config/model_config.json"):
    """Save model configuration"""
    Path(config_path).parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

def load_config(config_path: str = "config/model_config.json") -> Dict:
    """Load model configuration"""
    if Path(config_path).exists():
        with open(config_path, 'r') as f:
            return json.load(f)
    else:
        # Create default config
        config = create_model_config()
        save_config(config, config_path)
        return config

# Image processing utilities
class ImageProcessor:
    """Image processing utilities for quality control"""
    
    @staticmethod
    def enhance_image(image: np.ndarray) -> np.ndarray:
        """Enhance image quality for better detection"""
        # Convert to LAB color space for better enhancement
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        
        # Merge channels and convert back
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        return enhanced
    
    @staticmethod
    def detect_edges(image: np.ndarray) -> np.ndarray:
        """Detect edges for defect analysis"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        return edges
    
    @staticmethod
    def calculate_image_quality(image: np.ndarray) -> Dict:
        """Calculate image quality metrics"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Calculate sharpness (Laplacian variance)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Calculate brightness
        brightness = np.mean(gray)
        
        # Calculate contrast (standard deviation)
        contrast = np.std(gray)
        
        return {
            'sharpness': float(laplacian_var),
            'brightness': float(brightness),
            'contrast': float(contrast),
            'quality_score': min(100, (laplacian_var / 100) * 50 + (contrast / 128) * 50)
        }

# Training utilities for continuous learning
class ContinuousLearning:
    """Handle continuous learning and model updates"""
    
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
        self.feedback_data = []
        self.training_queue = []
        
    def add_feedback(self, image: np.ndarray, detection_result: Dict, 
                    user_feedback: Dict, sector: str):
        """Add user feedback for model improvement"""
        feedback_entry = {
            'image_hash': hash(image.tobytes()),
            'detection': detection_result,
            'user_feedback': user_feedback,
            'sector': sector,
            'timestamp': str(np.datetime64('now'))
        }
        
        self.feedback_data.append(feedback_entry)
        
        # Add to training queue if correction needed
        if user_feedback.get('correction_needed', False):
            self.training_queue.append(feedback_entry)
            
    def should_retrain(self, sector: str) -> bool:
        """Determine if model should be retrained"""
        sector_feedback = [f for f in self.feedback_data if f['sector'] == sector]
        
        if len(sector_feedback) < 10:
            return False
            
        # Check error rate
        corrections_needed = sum(1 for f in sector_feedback[-50:] 
                               if f['user_feedback'].get('correction_needed', False))
        
        error_rate = corrections_needed / min(50, len(sector_feedback))
        
        return error_rate > 0.2  # Retrain if error rate > 20%
    
    def prepare_training_data(self, sector: str) -> Dict:
        """Prepare data for model retraining"""
        sector_data = [f for f in self.training_queue if f['sector'] == sector]
        
        training_data = {
            'images': [],
            'annotations': [],
            'feedback_count': len(sector_data)
        }
        
        for data in sector_data:
            # Process feedback into training format
            training_data['annotations'].append({
                'detection': data['detection'],
                'correction': data['user_feedback']
            })
            
        return training_data

# Performance monitoring
class PerformanceMonitor:
    """Monitor system performance and model accuracy"""
    
    def __init__(self):
        self.metrics = {
            'detection_times': [],
            'accuracy_scores': [],
            'false_positives': 0,
            'false_negatives': 0,
            'true_positives': 0,
            'processing_fps': []
        }
        
    def log_detection_time(self, detection_time: float):
        """Log detection processing time"""
        self.metrics['detection_times'].append(detection_time)
        
        # Keep only last 1000 measurements
        if len(self.metrics['detection_times']) > 1000:
            self.metrics['detection_times'] = self.metrics['detection_times'][-1000:]
            
    def log_accuracy_feedback(self, predicted: bool, actual: bool):
        """Log accuracy feedback from user"""
        if predicted and actual:
            self.metrics['true_positives'] += 1
        elif predicted and not actual:
            self.metrics['false_positives'] += 1
        elif not predicted and actual:
            self.metrics['false_negatives'] += 1
            
    def get_performance_stats(self) -> Dict:
        """Get current performance statistics"""
        detection_times = self.metrics['detection_times']
        
        if not detection_times:
            return {'status': 'no_data'}
            
        # Calculate metrics
        avg_detection_time = np.mean(detection_times)
        fps = 1.0 / avg_detection_time if avg_detection_time > 0 else 0
        
        tp = self.metrics['true_positives']
        fp = self.metrics['false_positives']
        fn = self.metrics['false_negatives']
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'avg_detection_time': float(avg_detection_time),
            'fps': float(fps),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1_score),
            'total_detections': tp + fp,
            'accuracy_rate': float(tp / (tp + fp)) if (tp + fp) > 0 else 0
        }

# Main detection service
class DetectionService:
    """Main service for coordinating all detection components"""
    
    def __init__(self, config_path: str = "config/model_config.json"):
        self.config = load_config(config_path)
        self.model_manager = ModelManager()
        self.image_processor = ImageProcessor()
        self.continuous_learning = ContinuousLearning(self.model_manager)
        self.performance_monitor = PerformanceMonitor()
        self.initialize()
        
    def initialize(self):
        """Initialize detection service"""
        try:
            # Load models
            self.model_manager.load_sector_models()
            logger.info("Detection service initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize detection service: {e}")
            
    def process_frame(self, image: np.ndarray, sector: str) -> Dict:
        """Process single frame for defect detection"""
        start_time = time.time()
        
        try:
            # Enhance image quality
            enhanced_image = self.image_processor.enhance_image(image)
            
            # Calculate image quality
            quality_metrics = self.image_processor.calculate_image_quality(enhanced_image)
            
            # Run detection
            detection_result = self.model_manager.detect_defects(enhanced_image, sector)
            
            # Add processing time
            processing_time = time.time() - start_time
            self.performance_monitor.log_detection_time(processing_time)
            
            # Combine results
            result = {
                **detection_result,
                'image_quality': quality_metrics,
                'processing_time': processing_time,
                'timestamp': str(np.datetime64('now'))
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            return {
                'detections': [],
                'anomalies': [],
                'error': str(e),
                'processing_time': time.time() - start_time
            }
            
    def add_user_feedback(self, image: np.ndarray, detection_result: Dict, 
                         feedback: Dict, sector: str):
        """Add user feedback for continuous learning"""
        self.continuous_learning.add_feedback(image, detection_result, feedback, sector)
        
        # Log accuracy feedback
        predicted = len(detection_result.get('detections', [])) > 0
        actual = feedback.get('defect_present', False)
        self.performance_monitor.log_accuracy_feedback(predicted, actual)
        
    def get_system_status(self) -> Dict:
        """Get overall system status"""
        return {
            'model_stats': self.model_manager.get_model_stats(),
            'performance_stats': self.performance_monitor.get_performance_stats(),
            'config': self.config,
            'timestamp': str(np.datetime64('now'))
        }

# Export main classes
__all__ = [
    'DetectionService',
    'ModelManager', 
    'SectorSpecificDetector',
    'ElectricCableDetector',
    'SeedPackagingDetector',
    'ImageProcessor',
    'PerformanceMonitor'
]