# app.py
import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
import tempfile
import os
import time
from torchvision import transforms

# Import your model (make sure this path is correct)
from models.hybrid_model import SimpleCNNModel

# Set page configuration
st.set_page_config(
    page_title="Deepfake Detection System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

class StreamlitDeepfakeDetector:
    def __init__(self, model_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        st.info(f"Using device: {self.device}")
        
        # Load model
        self.model = SimpleCNNModel().to(self.device)
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
            st.success("Model loaded successfully!")
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return
        
        # Initialize face detector
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Transform
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Buffer for recent predictions
        self.predictions = []
        self.buffer_size = 5

    def detect_face(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) > 0:
            x, y, w, h = faces[0]
            x = max(0, x - 10)
            y = max(0, y - 10)
            w = min(w + 20, frame.shape[1] - x)
            h = min(h + 20, frame.shape[0] - y)
            return frame[y:y+h, x:x+w], (x, y, w, h)
        return None, None

    def process_image(self, image):
        """Process a single image for deepfake detection"""
        # Convert PIL to OpenCV format
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        face, bbox = self.detect_face(frame)
        
        if face is not None:
            # Convert to PIL Image and apply transform
            face_pil = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
            face_tensor = self.transform(face_pil).unsqueeze(0).to(self.device)
            
            # Make prediction
            with torch.no_grad():
                logits = self.model(face_tensor)
                confidence = torch.sigmoid(logits).item()
            
            # Add to prediction buffer for smoothing
            self.predictions.append(confidence)
            if len(self.predictions) > self.buffer_size:
                self.predictions.pop(0)
            
            smoothed_confidence = np.mean(self.predictions) if self.predictions else confidence
            
            # Draw results on frame
            x, y, w, h = bbox
            label = "FAKE" if smoothed_confidence > 0.5 else "REAL"
            color = (0, 0, 255) if smoothed_confidence > 0.5 else (0, 255, 0)
            
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"{label}: {smoothed_confidence:.3f}", 
                       (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Convert back to RGB for display
            result_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return result_frame, smoothed_confidence, label
        else:
            st.warning("No face detected in the image!")
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), 0.5, "No Face"

    def process_video_frame(self, frame):
        """Process a single video frame"""
        face, bbox = self.detect_face(frame)
        
        if face is not None:
            face_pil = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
            face_tensor = self.transform(face_pil).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                logits = self.model(face_tensor)
                confidence = torch.sigmoid(logits).item()
            
            self.predictions.append(confidence)
            if len(self.predictions) > self.buffer_size:
                self.predictions.pop(0)
            
            smoothed_confidence = np.mean(self.predictions) if self.predictions else confidence
            
            # Draw results
            x, y, w, h = bbox
            label = "FAKE" if smoothed_confidence > 0.5 else "REAL"
            color = (0, 0, 255) if smoothed_confidence > 0.5 else (0, 255, 0)
            
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"{label}: {smoothed_confidence:.3f}", 
                       (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            return frame, smoothed_confidence, label
        
        return frame, 0.5, "No Face"

def main():
    st.title("🔍 Deepfake Detection System")
    st.markdown("---")
    
    # Sidebar
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox(
        "Choose the app mode",
        ["Home", "Video Detection", "Webcam Detection", "About"]
    )
    
    # Model path
    base_dir = r"C:\Users\Sowmya\Downloads\Deep_fake"
    model_path = os.path.join(base_dir, "best_model.pth")
    
    # Check if model exists
    if not os.path.exists(model_path):
        st.error(f"Model file not found at {model_path}")
        st.info("Please train the model first using 02_train_model.py")
        return
    
    # Initialize detector
    detector = StreamlitDeepfakeDetector(model_path)
    
    if app_mode == "Home":
        st.header("Welcome to Deepfake Detection System")
        st.markdown("""
        This application uses deep learning to detect deepfake images and videos.
        
        ### Features:
        
        - 🎥 **Video Detection**: Upload and analyze video files
        - 🌐 **Real-time Webcam**: Real-time detection using your webcam
        
        ### How to use:
        1. Select a mode from the sidebar
        2. Upload an video or start webcam
        3. View the detection results in real-time
        
        ### Model Information:
        - Architecture: Custom CNN
        - Training Data: DFD (Deepfake Detection) dataset
        - Input: Face images (224x224)
        - Output: Real/Fake classification with confidence score
        """)
        
        # Display training history if available
        history_path = os.path.join(base_dir, "training_history.png")
        if os.path.exists(history_path):
            st.subheader("Model Training History")
            st.image(history_path, use_column_width=True)
    
    elif app_mode == "Image Detection":
        st.header("📷 Image Deepfake Detection")
        st.markdown("Upload an image to check if it's real or AI-generated.")
        
        uploaded_file = st.file_uploader(
            "Choose an image...", 
            type=['jpg', 'jpeg', 'png'],
            help="Upload a face image for deepfake detection"
        )
        
        if uploaded_file is not None:
            # Display original image
            image = Image.open(uploaded_file)
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Original Image")
                st.image(image, use_column_width=True)
            
            with col2:
                st.subheader("Analysis Result")
                
                # Add analyze button
                if st.button("Analyze Image", type="primary"):
                    with st.spinner("Analyzing image..."):
                        # Process image
                        result_image, confidence, label = detector.process_image(image)
                        
                        # Display result
                        st.image(result_image, use_column_width=True)
                        
                        # Display confidence
                        if label != "No Face":
                            st.metric(
                                label="Detection Result",
                                value=label,
                                delta=f"{confidence:.3f} confidence"
                            )
                            
                            # Confidence gauge
                            st.progress(int(confidence * 100))
                            st.write(f"Confidence Score: {confidence:.3f}")
                            
                            # Interpretation
                            if confidence > 0.7:
                                st.error("High confidence of being DEEPFAKE")
                            elif confidence > 0.5:
                                st.warning("Moderate confidence of being DEEPFAKE")
                            elif confidence > 0.3:
                                st.success("Moderate confidence of being REAL")
                            else:
                                st.success("High confidence of being REAL")
    
    elif app_mode == "Video Detection":
        st.header("🎥 Video Deepfake Detection")
        st.markdown("Upload a video file to analyze for deepfake content.")
        
        uploaded_file = st.file_uploader(
            "Choose a video...", 
            type=['mp4', 'avi', 'mov', 'mkv'],
            help="Upload a video file for deepfake detection"
        )
        
        if uploaded_file is not None:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                tmp_file.write(uploaded_file.read())
                video_path = tmp_file.name
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Original Video")
                st.video(uploaded_file)
            
            with col2:
                st.subheader("Analysis")
                if st.button("Analyze Video", type="primary"):
                    # Process video
                    cap = cv2.VideoCapture(video_path)
                    
                    if not cap.isOpened():
                        st.error("Error opening video file")
                        return
                    
                    # Video info
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    
                    st.info(f"Video Info: {total_frames} frames, {fps:.1f} FPS")
                    
                    # Process video
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    results = []
                    
                    frame_placeholder = st.empty()
                    
                    for frame_idx in range(total_frames):
                        ret, frame = cap.read()
                        if not ret:
                            break
                        
                        # Process every 5th frame to speed up
                        if frame_idx % 5 == 0:
                            processed_frame, confidence, label = detector.process_video_frame(frame)
                            
                            if label != "No Face":
                                results.append(confidence)
                            
                            # Display processed frame
                            frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                            frame_placeholder.image(frame_rgb, use_column_width=True)
                        
                        # Update progress
                        progress = (frame_idx + 1) / total_frames
                        progress_bar.progress(progress)
                        status_text.text(f"Processing frame {frame_idx + 1}/{total_frames}")
                    
                    cap.release()
                    
                    # Display summary
                    if results:
                        avg_confidence = np.mean(results)
                        final_label = "FAKE" if avg_confidence > 0.5 else "REAL"
                        
                        st.success("Analysis Complete!")
                        st.metric(
                            label="Overall Video Analysis",
                            value=final_label,
                            delta=f"{avg_confidence:.3f} average confidence"
                        )
                        
                        # Confidence distribution
                        st.subheader("Confidence Distribution")
                        st.bar_chart({f"Frame {i}": conf for i, conf in enumerate(results[:20])})
                    
                    # Clean up
                    os.unlink(video_path)
    
    elif app_mode == "Webcam Detection":
        st.header("🌐 Real-time Webcam Detection")
        st.markdown("Use your webcam for real-time deepfake detection.")
        
        if st.button("Start Webcam", type="primary"):
            st.info("Starting webcam... Press 'Stop' to end the session.")
            
            # Initialize webcam
            cap = cv2.VideoCapture(0)
            stop_button = st.button("Stop Webcam")
            
            frame_placeholder = st.empty()
            results_placeholder = st.empty()
            
            confidence_history = []
            
            while not stop_button:
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to access webcam")
                    break
                
                # Process frame
                processed_frame, confidence, label = detector.process_video_frame(frame)
                
                if label != "No Face":
                    confidence_history.append(confidence)
                
                # Convert to RGB for display
                frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(frame_rgb, use_column_width=True)
                
                # Display current results
                with results_placeholder.container():
                    if confidence_history:
                        current_conf = confidence_history[-1]
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Current Result", label, f"{current_conf:.3f}")
                        with col2:
                            st.metric("Average Confidence", f"{np.mean(confidence_history):.3f}")
                        with col3:
                            fake_percentage = len([c for c in confidence_history if c > 0.5]) / len(confidence_history) * 100
                            st.metric("Fake Frames", f"{fake_percentage:.1f}%")
                
                # Check for stop
                if stop_button:
                    break
            
            cap.release()
            st.success("Webcam session ended")
    
    elif app_mode == "About":
        st.header("About This Project")
        st.markdown("""
        ### Deepfake Detection System
        
        This application demonstrates a practical implementation of deepfake detection
        using deep learning techniques.
        
        ### Technical Details:
        - **Model**: Custom Convolutional Neural Network (CNN)
        - **Framework**: PyTorch
        - **Face Detection**: OpenCV Haar Cascades
        - **Web Interface**: Streamlit
        
        ### Dataset:
        The model was trained on the **DFD (Deepfake Detection)** dataset, which contains
        both original and manipulated video sequences.
        
        ### Features:
        - Real-time face detection and analysis
        - Confidence-based predictions
        - Multiple input modalities (image, video, webcam)
        - Smoothing of predictions for stable results
        
        ### Limitations:
        - Performance depends on face detection quality
        - Model accuracy varies with different deepfake generation methods
        - Requires clear frontal face images for best results
        
        For more information about the technical implementation, check the source code.
        """)

if __name__ == "__main__":
    main()