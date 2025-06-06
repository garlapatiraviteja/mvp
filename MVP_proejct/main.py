
# main.py - Factory AI MVP Main Application
# Complete integration of all components for Stage 1 MVP
import streamlit as st
import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import threading
import queue
import json
import logging
from pathlib import Path
import sys
import os
from typing import Dict, List, Optional, Tuple
import asyncio
from PIL import Image
import base64
from io import BytesIO

# Add current directory to path for imports
sys.path.append(os.path.dirname(__file__))

# Import our detection modules
try:
    from detection import (
        DetectionService, 
        ModelManager, 
        SectorSpecificDetector,
        ImageProcessor,
        PerformanceMonitor
    )
    from backend_architecture import (
        DataManager,
        AlertSystem,
        ReportGenerator,
        ConfigManager
    )
    MODULES_AVAILABLE = True
except ImportError:
    # Fallback for development
    print("Warning: Detection models not found, using mock implementations")
    MODULES_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('factory_ai.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Factory AI - Industrial Vision System",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #2a5298;
    }
    .alert-critical {
        background: #ff4444;
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .alert-warning {
        background: #ff8800;
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .alert-info {
        background: #0088cc;
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .status-good {
        color: #00aa00;
        font-weight: bold;
    }
    .status-warning {
        color: #ff8800;
        font-weight: bold;
    }
    .status-critical {
        color: #ff4444;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

class MockDetectionService:
    """Mock detection service for development/demo purposes"""
    def __init__(self):
        self.is_running = False
        self.sectors = ["Assembly Line", "Quality Control", "Packaging", "Warehouse"]
        
    def start_detection(self):
        self.is_running = True
        return True
        
    def stop_detection(self):
        self.is_running = False
        return True
        
    def get_latest_detections(self):
        """Generate mock detection data"""
        import random
        detections = []
        for sector in self.sectors:
            detection = {
                'timestamp': datetime.now(),
                'sector': sector,
                'defect_type': random.choice(['crack', 'scratch', 'dent', 'misalignment', 'contamination']),
                'confidence': random.uniform(0.7, 0.99),
                'severity': random.choice(['low', 'medium', 'high']),
                'coordinates': [random.randint(50, 200), random.randint(50, 200), 
                              random.randint(100, 150), random.randint(100, 150)],
                'image_path': f"mock_image_{sector.lower().replace(' ', '_')}.jpg"
            }
            detections.append(detection)
        return detections
        
    def get_performance_metrics(self):
        """Generate mock performance metrics"""
        import random
        return {
            'total_detections': random.randint(100, 500),
            'accuracy': random.uniform(0.85, 0.99),
            'processing_time': random.uniform(50, 200),
            'false_positives': random.randint(5, 20),
            'false_negatives': random.randint(2, 10)
        }

class FactoryAIApp:
    """Main application class for Factory AI MVP"""
    
    def __init__(self):
        self.initialize_session_state()
        self.setup_services()
        
    def initialize_session_state(self):
        """Initialize Streamlit session state variables"""
        if 'detection_active' not in st.session_state:
            st.session_state.detection_active = False
        if 'alerts' not in st.session_state:
            st.session_state.alerts = []
        if 'detection_history' not in st.session_state:
            st.session_state.detection_history = []
        if 'current_sector' not in st.session_state:
            st.session_state.current_sector = "Assembly Line"
        if 'last_update' not in st.session_state:
            st.session_state.last_update = datetime.now()
            
    def setup_services(self):
        """Initialize all services"""
        try:
            if MODULES_AVAILABLE:
                self.detection_service = DetectionService()
                self.model_manager = ModelManager()
                self.performance_monitor = PerformanceMonitor()
                self.data_manager = DataManager()
                self.alert_system = AlertSystem()
                self.report_generator = ReportGenerator()
                logger.info("All services initialized successfully")
            else:
                self.detection_service = MockDetectionService()
                logger.info("Using mock services for development")
        except Exception as e:
            logger.error(f"Error initializing services: {e}")
            self.detection_service = MockDetectionService()
            
    def render_header(self):
        """Render the main header"""
        st.markdown("""
        <div class="main-header">
            <h1>🏭 Factory AI - Industrial Vision System</h1>
            <p>Real-time defect detection and quality monitoring across manufacturing sectors</p>
        </div>
        """, unsafe_allow_html=True)
        
    def render_sidebar(self):
        """Render the sidebar with controls and settings"""
        st.sidebar.title("🔧 Control Panel")
        
        # System Status
        st.sidebar.subheader("System Status")
        if st.session_state.detection_active:
            st.sidebar.markdown('<p class="status-good">🟢 Active</p>', unsafe_allow_html=True)
        else:
            st.sidebar.markdown('<p class="status-warning">🟡 Inactive</p>', unsafe_allow_html=True)
            
        # Detection Controls
        st.sidebar.subheader("Detection Controls")
        col1, col2 = st.sidebar.columns(2)
        
        with col1:
            if st.button("▶️ Start", key="start_detection"):
                self.start_detection()
                
        with col2:
            if st.button("⏹️ Stop", key="stop_detection"):
                self.stop_detection()
                
        # Sector Selection
        st.sidebar.subheader("Sector Selection")
        sectors = ["Assembly Line", "Quality Control", "Packaging", "Warehouse"]
        st.session_state.current_sector = st.sidebar.selectbox(
            "Select Sector",
            sectors,
            index=sectors.index(st.session_state.current_sector)
        )
        
        # Model Settings
        st.sidebar.subheader("Model Settings")
        confidence_threshold = st.sidebar.slider(
            "Confidence Threshold",
            min_value=0.1,
            max_value=1.0,
            value=0.7,
            step=0.05
        )
        
        # Alert Settings
        st.sidebar.subheader("Alert Settings")
        alert_enabled = st.sidebar.checkbox("Enable Alerts", value=True)
        alert_threshold = st.sidebar.selectbox(
            "Alert Threshold",
            ["Low", "Medium", "High"],
            index=1
        )
        
        return {
            'confidence_threshold': confidence_threshold,
            'alert_enabled': alert_enabled,
            'alert_threshold': alert_threshold.lower()
        }
        
    def start_detection(self):
        """Start the detection process"""
        try:
            if self.detection_service.start_detection():
                st.session_state.detection_active = True
                st.success("Detection started successfully!")
                logger.info("Detection process started")
            else:
                st.error("Failed to start detection")
        except Exception as e:
            st.error(f"Error starting detection: {e}")
            logger.error(f"Error starting detection: {e}")
            
    def stop_detection(self):
        """Stop the detection process"""
        try:
            if self.detection_service.stop_detection():
                st.session_state.detection_active = False
                st.success("Detection stopped successfully!")
                logger.info("Detection process stopped")
            else:
                st.error("Failed to stop detection")
        except Exception as e:
            st.error(f"Error stopping detection: {e}")
            logger.error(f"Error stopping detection: {e}")
            
    def render_metrics_dashboard(self):
        """Render the main metrics dashboard"""
        st.subheader("📊 Real-time Metrics")
        
        # Get latest metrics
        try:
            metrics = self.detection_service.get_performance_metrics()
        except:
            metrics = {
                'total_detections': 0,
                'accuracy': 0.0,
                'processing_time': 0.0,
                'false_positives': 0,
                'false_negatives': 0
            }
            
        # Display metrics in columns
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Detections",
                metrics['total_detections'],
                delta=f"+{np.random.randint(1, 10)}" if st.session_state.detection_active else None
            )
            
        with col2:
            st.metric(
                "Accuracy",
                f"{metrics['accuracy']:.2%}",
                delta=f"{np.random.uniform(-0.02, 0.02):.2%}" if st.session_state.detection_active else None
            )
            
        with col3:
            st.metric(
                "Avg Processing Time",
                f"{metrics['processing_time']:.1f}ms",
                delta=f"{np.random.uniform(-10, 10):.1f}ms" if st.session_state.detection_active else None
            )
            
        with col4:
            st.metric(
                "False Positives",
                metrics['false_positives'],
                delta=f"+{np.random.randint(0, 3)}" if st.session_state.detection_active else None
            )
            
    def render_live_detection(self):
        """Render the live detection view"""
        st.subheader("🎯 Live Detection")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Camera feed placeholder
            st.subheader(f"Camera Feed - {st.session_state.current_sector}")
            
            if st.session_state.detection_active:
                # Create a placeholder for the video feed
                video_placeholder = st.empty()
                
                # Simulate camera feed with sample image
                sample_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
                
                # Add some mock detection boxes
                if np.random.random() > 0.7:  # 30% chance of detection
                    cv2.rectangle(sample_image, (100, 100), (200, 200), (0, 255, 0), 2)
                    cv2.putText(sample_image, 'Defect Detected', (100, 90), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                video_placeholder.image(sample_image, channels="RGB", use_column_width=True)
            else:
                st.info("Start detection to view live camera feed")
                
        with col2:
            # Detection results
            st.subheader("Detection Results")
            
            if st.session_state.detection_active:
                # Get latest detections
                try:
                    detections = self.detection_service.get_latest_detections()
                    
                    if detections:
                        for detection in detections[-5:]:  # Show last 5 detections
                            severity_color = {
                                'low': 'info',
                                'medium': 'warning', 
                                'high': 'critical'
                            }.get(detection['severity'], 'info')
                            
                            st.markdown(f"""
                            <div class="alert-{severity_color}">
                                <strong>{detection['defect_type'].title()}</strong><br>
                                Confidence: {detection['confidence']:.2%}<br>
                                Severity: {detection['severity'].title()}<br>
                                Time: {detection['timestamp'].strftime('%H:%M:%S')}
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.info("No detections yet")
                        
                except Exception as e:
                    st.error(f"Error getting detections: {e}")
            else:
                st.info("Start detection to view results")
                
    def render_alerts_panel(self):
        """Render the alerts panel"""
        st.subheader("🚨 Alerts & Notifications")
        
        # Generate mock alerts if detection is active
        if st.session_state.detection_active and np.random.random() > 0.8:
            new_alert = {
                'timestamp': datetime.now(),
                'type': np.random.choice(['Critical Defect', 'System Alert', 'Quality Issue']),
                'message': np.random.choice([
                    'Critical defect detected in Assembly Line',
                    'Quality threshold exceeded in Packaging',
                    'System performance degradation detected',
                    'Unusual pattern detected in Warehouse'
                ]),
                'severity': np.random.choice(['high', 'medium', 'low'])
            }
            st.session_state.alerts.insert(0, new_alert)
            
        # Display alerts
        if st.session_state.alerts:
            for i, alert in enumerate(st.session_state.alerts[:10]):  # Show last 10 alerts
                severity_color = {
                    'low': 'info',
                    'medium': 'warning',
                    'high': 'critical'
                }.get(alert['severity'], 'info')
                
                st.markdown(f"""
                <div class="alert-{severity_color}">
                    <strong>{alert['type']}</strong><br>
                    {alert['message']}<br>
                    <small>{alert['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}</small>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No alerts to display")
            
    def render_analytics_dashboard(self):
        """Render analytics and reporting dashboard"""
        st.subheader("📈 Analytics Dashboard")
        
        # Generate sample data for charts
        dates = pd.date_range(start=datetime.now()-timedelta(days=30), end=datetime.now(), freq='D')
        
        # Defect trend chart
        defect_data = pd.DataFrame({
            'date': dates,
            'defects': np.random.poisson(5, len(dates)),
            'sector': np.random.choice(['Assembly Line', 'Quality Control', 'Packaging', 'Warehouse'], len(dates))
        })
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Defect Trends (Last 30 Days)")
            fig_line = px.line(defect_data, x='date', y='defects', 
                              title='Daily Defect Count', 
                              labels={'defects': 'Number of Defects', 'date': 'Date'})
            st.plotly_chart(fig_line, use_container_width=True)
            
        with col2:
            st.subheader("Defects by Sector")
            sector_summary = defect_data.groupby('sector')['defects'].sum().reset_index()
            fig_bar = px.bar(sector_summary, x='sector', y='defects',
                           title='Total Defects by Sector',
                           labels={'defects': 'Total Defects', 'sector': 'Sector'})
            st.plotly_chart(fig_bar, use_container_width=True)
            
        # Performance metrics over time
        st.subheader("System Performance Metrics")
        
        perf_data = pd.DataFrame({
            'timestamp': pd.date_range(start=datetime.now()-timedelta(hours=24), 
                                     end=datetime.now(), freq='H'),
            'accuracy': np.random.uniform(0.85, 0.99, 24),
            'processing_time': np.random.uniform(50, 200, 24),
            'throughput': np.random.uniform(80, 120, 24)
        })
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            fig_acc = px.line(perf_data, x='timestamp', y='accuracy',
                            title='Model Accuracy (24h)',
                            labels={'accuracy': 'Accuracy (%)', 'timestamp': 'Time'})
            st.plotly_chart(fig_acc, use_container_width=True)
            
        with col2:
            fig_time = px.line(perf_data, x='timestamp', y='processing_time',
                             title='Processing Time (24h)',
                             labels={'processing_time': 'Time (ms)', 'timestamp': 'Time'})
            st.plotly_chart(fig_time, use_container_width=True)
            
        with col3:
            fig_throughput = px.line(perf_data, x='timestamp', y='throughput',
                                   title='Throughput (24h)',
                                   labels={'throughput': 'Items/min', 'timestamp': 'Time'})
            st.plotly_chart(fig_throughput, use_container_width=True)
            
    def render_settings_panel(self):
        """Render the settings and configuration panel"""
        st.subheader("⚙️ Settings & Configuration")
        
        tab1, tab2, tab3 = st.tabs(["Model Settings", "Alert Configuration", "System Settings"])
        
        with tab1:
            st.subheader("Model Configuration")
            
            col1, col2 = st.columns(2)
            with col1:
                st.selectbox("Primary Model", ["YOLOv8", "ResNet-50", "EfficientNet"])
                st.slider("Confidence Threshold", 0.1, 1.0, 0.7)
                st.slider("NMS Threshold", 0.1, 1.0, 0.5)
                
            with col2:
                st.selectbox("Backup Model", ["YOLOv5", "SSD", "RCNN"])
                st.number_input("Max Detections per Image", 1, 100, 10)
                st.checkbox("Enable Model Ensemble", value=False)
                
        with tab2:
            st.subheader("Alert Configuration")
            
            st.checkbox("Enable Email Alerts", value=True)
            st.checkbox("Enable SMS Alerts", value=False)
            st.checkbox("Enable Push Notifications", value=True)
            
            st.selectbox("Alert Frequency", ["Immediate", "Every 5 minutes", "Every 15 minutes", "Hourly"])
            st.multiselect("Alert Recipients", ["supervisor@factory.com", "quality@factory.com", "maintenance@factory.com"])
            
        with tab3:
            st.subheader("System Configuration")
            
            st.number_input("Video Frame Rate", 1, 60, 30)
            st.number_input("Detection Interval (seconds)", 1, 300, 5)
            st.selectbox("Log Level", ["DEBUG", "INFO", "WARNING", "ERROR"])
            
            st.checkbox("Enable Performance Monitoring", value=True)
            st.checkbox("Enable Data Logging", value=True)
            st.checkbox("Enable Automatic Reporting", value=True)
            
    def render_reports_section(self):
        """Render the reports section"""
        st.subheader("📋 Reports & Export")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Generate Reports")
            
            report_type = st.selectbox("Report Type", [
                "Daily Summary",
                "Weekly Analysis", 
                "Monthly Report",
                "Custom Range"
            ])
            
            if report_type == "Custom Range":
                start_date = st.date_input("Start Date")
                end_date = st.date_input("End Date")
                
            sectors = st.multiselect("Include Sectors", 
                                   ["Assembly Line", "Quality Control", "Packaging", "Warehouse"],
                                   default=["Assembly Line", "Quality Control"])
                                   
            if st.button("Generate Report"):
                st.success("Report generated successfully!")
                
                # Create sample report data
                report_data = {
                    'Total Detections': 156,
                    'Critical Issues': 12,
                    'Average Accuracy': '94.2%',
                    'System Uptime': '99.1%',
                    'Most Common Defect': 'Surface Scratch',
                    'Peak Detection Time': '14:30 - 15:00'
                }
                
                st.json(report_data)
                
        with col2:
            st.subheader("Export Options")
            
            st.button("📊 Export to Excel", key="export_excel")
            st.button("📄 Export to PDF", key="export_pdf")
            st.button("📈 Export Charts", key="export_charts")
            st.button("💾 Export Raw Data", key="export_raw")
            
            st.subheader("Scheduled Reports")
            st.checkbox("Daily Email Report", value=True)
            st.checkbox("Weekly Summary", value=True)
            st.checkbox("Monthly Analysis", value=False)
            
    def run(self):
        """Main application runner"""
        self.render_header()
        
        # Sidebar controls
        settings = self.render_sidebar()
        
        # Main content tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🎯 Live Detection", 
            "📊 Dashboard", 
            "🚨 Alerts", 
            "📈 Analytics",
            "⚙️ Settings"
        ])
        
        with tab1:
            self.render_live_detection()
            
        with tab2:
            self.render_metrics_dashboard()
            
        with tab3:
            self.render_alerts_panel()
            
        with tab4:
            self.render_analytics_dashboard()
            
        with tab5:
            self.render_settings_panel()
            self.render_reports_section()
            
        # Auto-refresh every 5 seconds when detection is active
        if st.session_state.detection_active:
            time.sleep(5)
            st.rerun()

def main():
    """Main function to run the Factory AI application"""
    try:
        app = FactoryAIApp()
        app.run()
    except Exception as e:
        st.error(f"Application Error: {e}")
        logger.error(f"Application error: {e}")
        
        # Show error details in expander
        with st.expander("Error Details"):
            st.code(str(e))
            
if __name__ == "__main__":
    main()