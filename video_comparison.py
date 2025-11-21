"""
Video Comparison Module
Creates side-by-side comparison videos of original and HIT-refined tracking.
"""

import os
import cv2
import numpy as np
from collections import defaultdict


def load_tracks_by_frame(filepath):
    """Load tracking results organized by frame"""
    tracks_by_frame = defaultdict(list)
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            frame = int(parts[0])
            track_id = int(parts[1])
            x, y, w, h = map(float, parts[2:6])
            conf = float(parts[6])
            tracks_by_frame[frame].append({
                'id': track_id,
                'bbox': [x, y, w, h],
                'conf': conf
            })
    return tracks_by_frame


def draw_tracks_on_image(image, tracks, title, use_hit_colors=False, original_detections_set=None):
    """Draw bounding boxes and track IDs on image"""
    img = image.copy()
    
    cv2.putText(img, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
               1.0, (0, 0, 255), 2, cv2.LINE_AA)
    
    for track in tracks:
        track_id = track['id']
        x, y, w, h = track['bbox']
        
        color = (0, 0, 255)  # Red by default
        
        # For HIT side, color interpolated detections pink
        if use_hit_colors and original_detections_set is not None:
            frame_key = (track.get('frame', 0), int(x), int(y), int(w), int(h))
            if frame_key not in original_detections_set:
                color = (255, 37, 255)  # Pink for interpolated
        
        cv2.rectangle(img, 
                     (int(x), int(y)), 
                     (int(x + w), int(y + h)), 
                     color, 2)
        
        # Draw ID label with background
        label = f'ID:{track_id}'
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img,
                     (int(x), int(y) - label_size[1] - 10),
                     (int(x) + label_size[0], int(y)),
                     color, -1)
        cv2.putText(img, label, 
                   (int(x), int(y) - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    count_text = f'Tracks: {len(set(t["id"] for t in tracks))}'
    cv2.putText(img, count_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX,
               0.8, (0, 0, 255), 2, cv2.LINE_AA)
    
    return img


def create_comparison_video(original_results_file, hit_results_file, frames_folder, 
                           output_dir, tracker_name, video_name, verbose=True):
    """Create side-by-side comparison images and video"""
    
    if verbose:
        print("\nCreating comparison video...")
    
    original_tracks = load_tracks_by_frame(original_results_file)
    hit_tracks = load_tracks_by_frame(hit_results_file)
    
    # Build set of original detections for comparison
    original_detections_set = set()
    with open(original_results_file, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            frame = int(parts[0])
            x, y, w, h = map(float, parts[2:6])
            original_detections_set.add((frame, int(x), int(y), int(w), int(h)))
    
    all_frames = sorted(set(list(original_tracks.keys()) + list(hit_tracks.keys())))
    os.makedirs(output_dir, exist_ok=True)
    
    for frame_num in all_frames:
        # Load original frame image - try different naming formats
        frame_path = f"{frames_folder}/frame-{frame_num:04d}.jpg"
        if not os.path.exists(frame_path):
            frame_path = f"{frames_folder}/frame-{frame_num:04d}.png"
        if not os.path.exists(frame_path):
            frame_path = f"{frames_folder}/{frame_num:06d}.jpg"
        if not os.path.exists(frame_path):
            frame_path = f"{frames_folder}/{frame_num:06d}.png"
        
        if not os.path.exists(frame_path):
            if verbose:
                print(f"Warning: Frame {frame_num} not found, skipping...")
            continue
        
        img = cv2.imread(frame_path)
        if img is None:
            continue
        
        # Add frame number for detection checking
        for track in hit_tracks.get(frame_num, []):
            track['frame'] = frame_num
        
        img_original = draw_tracks_on_image(
            img, 
            original_tracks.get(frame_num, []),
            f"{tracker_name.upper()}"
        )
        
        img_hit = draw_tracks_on_image(
            img,
            hit_tracks.get(frame_num, []),
            f"{tracker_name.upper()} + HIT",
            use_hit_colors=True,
            original_detections_set=original_detections_set
        )
        
        comparison = np.hstack([img_original, img_hit])
        cv2.imwrite(f"{output_dir}/frame_{frame_num:06d}.jpg", comparison)
        
        if verbose and frame_num % 50 == 0:
            print(f"Processed {frame_num}/{max(all_frames)} frames")
    
    # Create video from comparison frames
    frame_files = sorted([f for f in os.listdir(output_dir) if f.startswith('frame_') and f.endswith('.jpg')])
    
    if frame_files:
        first_frame_path = os.path.join(output_dir, frame_files[0])
        first_frame = cv2.imread(first_frame_path)
        
        if first_frame is not None:
            height, width = first_frame.shape[:2]
            video_path = os.path.join(output_dir, "comparison_video.mp4")
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(video_path, fourcc, 15.0, (width, height))
            
            for frame_file in frame_files:
                frame_path = os.path.join(output_dir, frame_file)
                frame = cv2.imread(frame_path)
                if frame is not None:
                    video_writer.write(frame)
            
            video_writer.release()
            if verbose:
                print(f"Video saved: {video_path}")
            return video_path
    
    if verbose:
        print("No frames found to create video.")
    return None
