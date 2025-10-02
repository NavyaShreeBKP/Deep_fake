import os
import cv2
from mtcnn import MTCNN
import numpy as np
from tqdm import tqdm
import random
from sklearn.model_selection import train_test_split
import time

# Initialize the MTCNN detector with optimized settings
detector = MTCNN(
    min_face_size=20,
    scale_factor=0.709,
    steps_threshold=[0.6, 0.7, 0.7]
)

def extract_faces_from_video(video_path, output_dir, frame_interval=30, label='real'):
    """
    Extracts faces from a video file and saves them to an output directory.
    Optimized for better performance.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return 0
            
        frame_count = 0
        saved_count = 0
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        
        # Get total frames for progress bar
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % frame_interval == 0 and frame is not None:
                # Resize frame for faster processing (optional)
                small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
                rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                
                try:
                    detections = detector.detect_faces(rgb_frame)
                    
                    if detections:
                        best_detection = max(detections, key=lambda x: x['confidence'])
                        if best_detection['confidence'] > 0.9:
                            # Scale coordinates back to original size
                            x, y, width, height = [coord * 2 for coord in best_detection['box']]
                            x, y = max(0, x), max(0, y)
                            
                            # Extract face with boundary checking
                            face = frame[y:y+height, x:x+width]
                            
                            if face.size > 0 and face.shape[0] > 50 and face.shape[1] > 50:
                                face_resized = cv2.resize(face, (224, 224))
                                output_path = os.path.join(output_dir, f"{video_name}_f{frame_count:06d}.jpg")
                                cv2.imwrite(output_path, face_resized)
                                saved_count += 1
                                
                except Exception as e:
                    print(f"Error processing frame {frame_count} in {video_path}: {e}")
                    continue
                    
            frame_count += 1
            
            # Progress update
            if frame_count % 100 == 0:
                print(f"Processed {frame_count}/{total_frames} frames from {video_name}")
            
        cap.release()
        return saved_count
        
    except Exception as e:
        print(f"Error processing video {video_path}: {e}")
        return 0

def extract_faces_faster(video_path, output_dir, frame_interval=30, label='real'):
    """
    Alternative faster method using OpenCV's built-in face detector
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load OpenCV face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    saved_count = 0
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_count % frame_interval == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            if len(faces) > 0:
                # Take the largest face
                faces = sorted(faces, key=lambda x: x[2]*x[3], reverse=True)
                x, y, w, h = faces[0]
                
                face = frame[y:y+h, x:x+w]
                if face.size > 0:
                    face_resized = cv2.resize(face, (224, 224))
                    output_path = os.path.join(output_dir, f"{video_name}_f{frame_count:06d}.jpg")
                    cv2.imwrite(output_path, face_resized)
                    saved_count += 1
                    
        frame_count += 1
        
    cap.release()
    return saved_count

def create_train_val_split(input_dir, output_base_dir, test_size=0.2):
    """
    Creates train/validation split from processed images
    """
    real_dir = os.path.join(input_dir, 'real')
    fake_dir = os.path.join(input_dir, 'fake')
    
    real_images = [os.path.join(real_dir, f) for f in os.listdir(real_dir) if f.endswith('.jpg')]
    fake_images = [os.path.join(fake_dir, f) for f in os.listdir(fake_dir) if f.endswith('.jpg')]
    
    print(f"Found {len(real_images)} real images and {len(fake_images)} fake images")
    
    # Split real images
    real_train, real_val = train_test_split(real_images, test_size=test_size, random_state=42)
    fake_train, fake_val = train_test_split(fake_images, test_size=test_size, random_state=42)
    
    # Create directories
    train_real_dir = os.path.join(output_base_dir, 'train', 'real')
    train_fake_dir = os.path.join(output_base_dir, 'train', 'fake')
    val_real_dir = os.path.join(output_base_dir, 'val', 'real')
    val_fake_dir = os.path.join(output_base_dir, 'val', 'fake')
    
    for dir_path in [train_real_dir, train_fake_dir, val_real_dir, val_fake_dir]:
        os.makedirs(dir_path, exist_ok=True)
    
    # Copy files to appropriate directories (use copy instead of move for safety)
    import shutil
    
    def copy_files(file_list, destination_dir):
        for file_path in file_list:
            shutil.copy2(file_path, os.path.join(destination_dir, os.path.basename(file_path)))
    
    copy_files(real_train, train_real_dir)
    copy_files(real_val, val_real_dir)
    copy_files(fake_train, train_fake_dir)
    copy_files(fake_val, val_fake_dir)
    
    print(f"Train/Validation split created:")
    print(f"Train Real: {len(real_train)}, Train Fake: {len(fake_train)}")
    print(f"Val Real: {len(real_val)}, Val Fake: {len(fake_val)}")

def check_video_files(video_list):
    """Check if video files exist and are accessible"""
    valid_videos = []
    for video_path in video_list:
        if os.path.exists(video_path):
            cap = cv2.VideoCapture(video_path)
            if cap.isOpened():
                valid_videos.append(video_path)
                cap.release()
            else:
                print(f"Warning: Could not open video file: {video_path}")
        else:
            print(f"Warning: Video file not found: {video_path}")
    return valid_videos

if __name__ == "__main__":
    base_dir = r"C:\Users\Sowmya\Downloads\Deep_fake"
    
    # Input directories
    original_videos_root = os.path.join(base_dir, "DFD_original sequences")
    manipulated_videos_root = os.path.join(base_dir, "DFD_manipulated_sequences")
    
    # Output directories
    processed_dir = os.path.join(base_dir, "processed_data")
    output_real = os.path.join(processed_dir, "real")
    output_fake = os.path.join(processed_dir, "fake")
    
    # Find all video files
    video_extensions = ('.avi', '.mp4', '.mov', '.mkv')
    
    print("Searching for video files...")
    original_videos = []
    for root, dirs, files in os.walk(original_videos_root):
        for file in files:
            if file.lower().endswith(video_extensions):
                original_videos.append(os.path.join(root, file))
                
    manipulated_videos = []
    for root, dirs, files in os.walk(manipulated_videos_root):
        for file in files:
            if file.lower().endswith(video_extensions):
                manipulated_videos.append(os.path.join(root, file))
    
    print(f"Found {len(original_videos)} original videos")
    print(f"Found {len(manipulated_videos)} manipulated videos")
    
    # Validate video files
    print("Validating video files...")
    original_videos = check_video_files(original_videos)
    manipulated_videos = check_video_files(manipulated_videos)
    
    print(f"Valid original videos: {len(original_videos)}")
    print(f"Valid manipulated videos: {len(manipulated_videos)}")
    
    # Ask user which detector to use
    print("\nChoose face detection method:")
    print("1. MTCNN (More accurate but slower)")
    print("2. OpenCV Haar Cascade (Faster but less accurate)")
    choice = input("Enter choice (1 or 2): ").strip()
    
    use_fast_method = (choice == "2")
    
    # Process Real Videos
    print("\nProcessing REAL videos...")
    total_real_faces = 0
    for i, video_path in enumerate(tqdm(original_videos, desc="Processing REAL videos")):
        print(f"Processing real video {i+1}/{len(original_videos)}: {os.path.basename(video_path)}")
        if use_fast_method:
            total_real_faces += extract_faces_faster(video_path, output_real, frame_interval=30, label='real')
        else:
            total_real_faces += extract_faces_from_video(video_path, output_real, frame_interval=30, label='real')
    
    # Process Fake Videos
    print("\nProcessing FAKE videos...")
    total_fake_faces = 0
    for i, video_path in enumerate(tqdm(manipulated_videos, desc="Processing FAKE videos")):
        print(f"Processing fake video {i+1}/{len(manipulated_videos)}: {os.path.basename(video_path)}")
        if use_fast_method:
            total_fake_faces += extract_faces_faster(video_path, output_fake, frame_interval=30, label='fake')
        else:
            total_fake_faces += extract_faces_from_video(video_path, output_fake, frame_interval=30, label='fake')
    
    print(f"\nExtraction Complete:")
    print(f"Total Real faces extracted: {total_real_faces}")
    print(f"Total Fake faces extracted: {total_fake_faces}")
    
    # Create train/validation split
    print("\nCreating train/validation split...")
    create_train_val_split(processed_dir, os.path.join(base_dir, "dataset"))
    
    print("Data preprocessing complete!")