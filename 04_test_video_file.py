import torch
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image
import os
from models.hybrid_model import SimpleCNNModel

class VideoFileTester:
    def __init__(self, model_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = SimpleCNNModel().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def test_video(self, video_path, output_path=None):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return
        
        frame_count = 0
        total_confidence = 0
        faces_detected = 0
        
        print(f"Testing video: {os.path.basename(video_path)}")
        print("Press 'q' to stop early")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect face
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
            
            if len(faces) > 0:
                x, y, w, h = faces[0]
                face = frame[y:y+h, x:x+w]
                
                if face.size > 0:
                    # Process face
                    face_pil = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
                    face_tensor = self.transform(face_pil).unsqueeze(0).to(self.device)
                    
                    # Make prediction
                    with torch.no_grad():
                        logits = self.model(face_tensor)
                        confidence = torch.sigmoid(logits).item()
                    
                    total_confidence += confidence
                    faces_detected += 1
                    
                    print(f"Frame {frame_count}: Confidence = {confidence:.3f} ({'FAKE' if confidence > 0.5 else 'REAL'})")
            
            frame_count += 1
            
            # Process every 10th frame to speed up testing
            if frame_count % 10 != 0:
                continue
        
        cap.release()
        
        if faces_detected > 0:
            avg_confidence = total_confidence / faces_detected
            result = "FAKE" if avg_confidence > 0.5 else "REAL"
            print(f"\n=== FINAL RESULT ===")
            print(f"Average confidence: {avg_confidence:.3f}")
            print(f"Prediction: {result}")
            print(f"Frames processed: {frame_count}")
            print(f"Faces detected: {faces_detected}")
        else:
            print("No faces detected in the video")

if __name__ == "__main__":
    base_dir = r"C:\Users\Sowmya\Downloads\Deep_fake"
    model_path = os.path.join(base_dir, "best_model.pth")
    
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        print("Please train the model first using 02_train_model.py")
    else:
        tester = VideoFileTester(model_path)
        
        # Test on a video file - replace with your video path
        test_video_path = r"C:\Users\Sowmya\Downloads\Deep_fake\DFD_manipulated_sequences\01_11__outside_talking_pan_laughing__LL4RUKZA.mp4"  # Change this to your video file
        
        if os.path.exists(test_video_path):
            tester.test_video(test_video_path)
        else:
            print(f"Test video not found: {test_video_path}")
            print("Please create a test video or download one to test with.")