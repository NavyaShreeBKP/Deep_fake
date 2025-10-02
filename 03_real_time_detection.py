import torch
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image
import time
from models.hybrid_model import SimpleCNNModel
import os

class RealTimeDeepfakeDetector:
    def __init__(self, model_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Load simple model
        self.model = SimpleCNNModel().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
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

    def process_frame(self, frame):
        face, bbox = self.detect_face(frame)
        
        if face is not None:
            # Convert to PIL Image and apply transform
            face_pil = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
            face_tensor = self.transform(face_pil).unsqueeze(0).to(self.device)
            
            # Make prediction
            with torch.no_grad():
                logits = self.model(face_tensor)
                # Apply sigmoid manually to get probability
                confidence = torch.sigmoid(logits).item()
            
            # Add to prediction buffer for smoothing
            self.predictions.append(confidence)
            if len(self.predictions) > self.buffer_size:
                self.predictions.pop(0)
            
            smoothed_confidence = np.mean(self.predictions) if self.predictions else 0.5
            
            # Draw results on frame
            x, y, w, h = bbox
            label = "FAKE" if smoothed_confidence > 0.5 else "REAL"
            color = (0, 0, 255) if smoothed_confidence > 0.5 else (0, 255, 0)
            
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"{label}: {smoothed_confidence:.3f}", 
                       (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        return frame

    def run(self):
        cap = cv2.VideoCapture(0)
        
        print("Starting real-time deepfake detection...")
        print("Press 'q' to quit")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            processed_frame = self.process_frame(frame)
            
            # Display frame
            cv2.imshow('Deepfake Detection', processed_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    base_dir = r"C:\Users\Sowmya\Downloads\Deep_fake"
    model_path = os.path.join(base_dir, "best_model.pth")
    
    detector = RealTimeDeepfakeDetector(model_path)
    detector.run()