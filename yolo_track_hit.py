from ultralytics import YOLO
import os
import numpy as np
from scipy.optimize import linear_sum_assignment
from collections import defaultdict
import cv2
from pathlib import Path
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import sys

# Add trackeval to path
sys.path.insert(0, os.path.abspath('.'))

# ==================== CONFIGURATION ====================
YOLO_MODEL_PATH = "yolo_v5n_best.pt"
VIDEO = "V01"
FRAMES_FOLDER = f"./videos/{VIDEO}/images"
IMAGE_SIZE = 1920

# TRACKER CONFIGURATION
TRACKER_CONFIG = "botsort-config-v4.yaml"
TRACKER_FOLDER_NAME = 'botsort_v4'
RESULTS_FILE = f'data/trackers/moranget/moranget-test/{TRACKER_FOLDER_NAME}/data/{VIDEO}.txt'

# HIT POST-PROCESSING CONFIGURATION
SKIP_YOLO_TRACKING = True  # Set to False to run YOLO tracking (set True to use existing results)
APPLY_HIT = True  # Set to False to skip HIT post-processing
CREATE_COMPARISON_VIDEO = True  # Set to False to skip comparison visualization
HIT_CONFIG = {
    'high_conf_thr': 0.55,      # High confidence threshold for motion matching
    'low_conf_thr': 0.40,       # Low confidence threshold
    'delta_t_scales': [1, 2, 4, 8, 16],  # Hierarchical matching scales
    'motion_iou_thr': 0.2,      # IoU threshold for motion-based matching (1-IoU cost)
    'iou_thr': 0.2,             # IoU threshold for second stage matching
    'interpolate': True,        # Fill gaps between matched tracklets
    'min_tracklet_len': 1,      # Minimum tracklet length to keep (1 = keep all detections)
}

# ==================== YOLO TRACKING ====================
directory = os.path.dirname(RESULTS_FILE)
os.makedirs(directory, exist_ok=True)

if SKIP_YOLO_TRACKING and os.path.exists(RESULTS_FILE):
    print(f"Skipping YOLO tracking - using existing results from: {RESULTS_FILE}")
else:
    print(f"Running YOLO tracking on {VIDEO}...")
    model = YOLO(YOLO_MODEL_PATH)
    
    results = model.track(
        source=FRAMES_FOLDER,
        save=True,
        save_txt=False,
        imgsz=IMAGE_SIZE,
        conf=0.40,
        iou=0.40,
        tracker=TRACKER_CONFIG,
        persist=True,
        verbose=True
    )
    
    # Save initial tracking results
    with open(RESULTS_FILE, 'w') as f:
        for frame_id, result in enumerate(results, start=1):
            if result.boxes is not None and len(result.boxes) > 0:
                for box in result.boxes:
                    if box.id is not None:
                        bbox = box.xyxy[0].tolist()
                        track_id = int(box.id.item())
                        conf = box.conf.item()
                        width = bbox[2] - bbox[0]
                        height = bbox[3] - bbox[1]
                        f.write(f'{frame_id},{track_id},{bbox[0]:.2f},{bbox[1]:.2f},{width:.2f},{height:.2f},{conf:.4f},-1,-1,-1\n')
    
    print(f"Initial tracking saved to: {RESULTS_FILE}")

# ==================== HIT POST-PROCESSING ====================
if not APPLY_HIT:
    print("HIT post-processing disabled. Exiting.")
    exit(0)

print("\nApplying HIT post-processing...")

class Tracklet:
    """Represents a sequence of detections with the same track ID"""
    def __init__(self, track_id, detections):
        self.track_id = track_id
        self.detections = sorted(detections, key=lambda x: x[0])  # Sort by frame
        self.start_frame = self.detections[0][0]
        self.end_frame = self.detections[-1][0]
        self.max_conf = max(det[6] for det in self.detections)
        
    def get_bbox_at_frame(self, frame):
        """Get bounding box at specific frame, or None if not present"""
        for det in self.detections:
            if det[0] == frame:
                return det[2:6]  # [x, y, w, h]
        return None
    
    def predict_bbox(self, target_frame):
        """Simple linear motion prediction"""
        if len(self.detections) < 2:
            # Use last known position
            return self.detections[-1][2:6]
        
        # Use last two detections for velocity estimation
        det1 = self.detections[-2]
        det2 = self.detections[-1]
        dt = det2[0] - det1[0]
        
        if dt == 0:
            return det2[2:6]
        
        # Calculate velocity
        vx = (det2[2] - det1[2]) / dt
        vy = (det2[3] - det1[3]) / dt
        vw = (det2[4] - det1[4]) / dt
        vh = (det2[5] - det1[5]) / dt
        
        # Predict forward
        steps = target_frame - det2[0]
        pred_x = det2[2] + vx * steps
        pred_y = det2[3] + vy * steps
        pred_w = max(1, det2[4] + vw * steps)
        pred_h = max(1, det2[5] + vh * steps)
        
        return [pred_x, pred_y, pred_w, pred_h]

def iou(bbox1, bbox2):
    """Calculate IoU between two bounding boxes [x, y, w, h]"""
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    
    # Convert to [x1, y1, x2, y2]
    box1 = [x1, y1, x1 + w1, y1 + h1]
    box2 = [x2, y2, x2 + w2, y2 + h2]
    
    # Calculate intersection
    xi1 = max(box1[0], box2[0])
    yi1 = max(box1[1], box2[1])
    xi2 = min(box1[2], box2[2])
    yi2 = min(box1[3], box2[3])
    
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    
    # Calculate union
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area
    
    if union_area == 0:
        return 0.0
    
    return inter_area / union_area

def load_tracking_results(filepath):
    """Load MOTChallenge format tracking results"""
    detections = []
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            frame = int(parts[0])
            track_id = int(parts[1])
            x, y, w, h = map(float, parts[2:6])
            conf = float(parts[6])
            detections.append([frame, track_id, x, y, w, h, conf])
    return detections

def split_into_tracklets(detections, max_gap=1):
    """Split detections into tracklets (continuous track segments)"""
    tracks = defaultdict(list)
    for det in detections:
        tracks[det[1]].append(det)
    
    tracklets = []
    for track_id, dets in tracks.items():
        dets = sorted(dets, key=lambda x: x[0])
        
        # Split at gaps
        current_tracklet = [dets[0]]
        for i in range(1, len(dets)):
            if dets[i][0] - dets[i-1][0] > max_gap:
                # Gap detected, save current tracklet and start new one
                tracklets.append(Tracklet(track_id, current_tracklet))
                current_tracklet = [dets[i]]
            else:
                current_tracklet.append(dets[i])
        
        if current_tracklet:
            tracklets.append(Tracklet(track_id, current_tracklet))
    
    return tracklets

def hierarchical_matching(tracklets, delta_t, config):
    """Match tracklets at a specific temporal scale"""
    high_conf_thr = config['high_conf_thr']
    motion_iou_thr = config['motion_iou_thr']
    iou_thr = config['iou_thr']
    
    # Separate high and low confidence tracklets
    high_conf_tracklets = [t for t in tracklets if t.max_conf >= high_conf_thr]
    low_conf_tracklets = [t for t in tracklets if t.max_conf < high_conf_thr]
    
    # Find candidate pairs (tracklets that could be linked)
    unmatched_tracklets = tracklets.copy()
    matches = []
    
    # Stage 1: Motion-based matching for high-confidence tracklets
    for t1 in high_conf_tracklets:
        best_match = None
        best_iou = motion_iou_thr
        
        for t2 in unmatched_tracklets:
            if t2.track_id == t1.track_id:
                continue
            
            # Check if t2 starts within delta_t after t1 ends
            gap = t2.start_frame - t1.end_frame
            if gap < 1 or gap > delta_t:
                continue
            
            # Predict t1's position at t2's start frame
            predicted_bbox = t1.predict_bbox(t2.start_frame)
            actual_bbox = t2.get_bbox_at_frame(t2.start_frame)
            
            if actual_bbox is None:
                continue
            
            # Calculate IoU
            iou_score = iou(predicted_bbox, actual_bbox)
            
            if iou_score > best_iou:
                best_iou = iou_score
                best_match = t2
        
        if best_match:
            matches.append((t1, best_match))
            unmatched_tracklets.remove(best_match)
    
    # Stage 2: IoU-based matching for remaining tracklets
    remaining = [t for t in unmatched_tracklets if t not in [m[0] for m in matches]]
    
    for t1 in remaining[:]:
        best_match = None
        best_iou = iou_thr
        
        for t2 in remaining:
            if t2.track_id == t1.track_id or t1 == t2:
                continue
            
            gap = t2.start_frame - t1.end_frame
            if gap < 1 or gap > delta_t:
                continue
            
            # Use last bbox of t1 and first bbox of t2
            bbox1 = t1.detections[-1][2:6]
            bbox2 = t2.detections[0][2:6]
            
            iou_score = iou(bbox1, bbox2)
            
            if iou_score > best_iou:
                best_iou = iou_score
                best_match = t2
        
        if best_match:
            matches.append((t1, best_match))
            remaining.remove(t1)
            if best_match in remaining:
                remaining.remove(best_match)
    
    return matches

def merge_tracklets(t1, t2, new_id, interpolate=False):
    """Merge two tracklets into one"""
    all_detections = t1.detections + t2.detections
    
    # Interpolate if there's a gap
    if interpolate and t2.start_frame - t1.end_frame > 1:
        gap_frames = range(t1.end_frame + 1, t2.start_frame)
        last_bbox = t1.detections[-1][2:6]
        first_bbox = t2.detections[0][2:6]
        
        # Linear interpolation
        for i, frame in enumerate(gap_frames):
            alpha = (i + 1) / (len(gap_frames) + 1)
            interp_x = last_bbox[0] + alpha * (first_bbox[0] - last_bbox[0])
            interp_y = last_bbox[1] + alpha * (first_bbox[1] - last_bbox[1])
            interp_w = last_bbox[2] + alpha * (first_bbox[2] - last_bbox[2])
            interp_h = last_bbox[3] + alpha * (first_bbox[3] - last_bbox[3])
            interp_conf = t1.detections[-1][6]  # Use confidence from last detection
            
            all_detections.append([frame, new_id, interp_x, interp_y, interp_w, interp_h, interp_conf])
    
    # Update track IDs
    for det in all_detections:
        det[1] = new_id
    
    return Tracklet(new_id, all_detections)

def apply_hit_postprocessing(input_file, output_file, config):
    """Apply HIT hierarchical matching to refine tracking results"""
    # Load detections
    detections = load_tracking_results(input_file)
    print(f"Loaded {len(detections)} detections")
    
    # Split into initial tracklets
    tracklets = split_into_tracklets(detections)
    print(f"Split into {len(tracklets)} initial tracklets")
    
    # Hierarchical matching
    for delta_t in config['delta_t_scales']:
        print(f"  Matching at delta_t={delta_t}...")
        matches = hierarchical_matching(tracklets, delta_t, config)
        print(f"    Found {len(matches)} matches")
        
        # Apply matches
        new_tracklets = []
        merged_tracklets = set()
        next_id = max(t.track_id for t in tracklets) + 1
        
        for t1, t2 in matches:
            if t1 in merged_tracklets or t2 in merged_tracklets:
                continue
            
            # Merge tracklets
            merged = merge_tracklets(t1, t2, next_id, config['interpolate'])
            new_tracklets.append(merged)
            merged_tracklets.add(t1)
            merged_tracklets.add(t2)
            next_id += 1
        
        # Keep unmerged tracklets
        for t in tracklets:
            if t not in merged_tracklets:
                new_tracklets.append(t)
        
        tracklets = new_tracklets
        print(f"    Reduced to {len(tracklets)} tracklets")
    
    # Filter short tracklets
    tracklets = [t for t in tracklets if len(t.detections) >= config['min_tracklet_len']]
    print(f"After filtering: {len(tracklets)} tracklets")
    
    # Renumber track IDs sequentially
    tracklets = sorted(tracklets, key=lambda t: t.start_frame)
    all_detections = []
    for new_id, tracklet in enumerate(tracklets, start=1):
        for det in tracklet.detections:
            det[1] = new_id
            all_detections.append(det)
    
    # Sort by frame
    all_detections = sorted(all_detections, key=lambda x: (x[0], x[1]))
    
    # Write results
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        for det in all_detections:
            f.write(f'{int(det[0])},{int(det[1])},{det[2]:.2f},{det[3]:.2f},{det[4]:.2f},{det[5]:.2f},{det[6]:.4f},-1,-1,-1\n')
    
    print(f"HIT post-processing complete!")
    return len(tracklets)

# Apply HIT post-processing
hit_output_file = f'data/trackers/moranget/moranget-test/{TRACKER_FOLDER_NAME}_HIT/data/{VIDEO}.txt'
num_tracks = apply_hit_postprocessing(RESULTS_FILE, hit_output_file, HIT_CONFIG)

print(f"\n{'='*60}")
print(f"RESULTS SUMMARY")
print(f"{'='*60}")
print(f"Video: {VIDEO}")
print(f"Base tracker: {TRACKER_FOLDER_NAME}")
print(f"Original results: {RESULTS_FILE}")
print(f"HIT refined results: {hit_output_file}")
print(f"Final number of tracks: {num_tracks}")
print(f"{'='*60}")

# ==================== COMPARISON VISUALIZATION ====================
if CREATE_COMPARISON_VIDEO:
    print("\nCreating side-by-side comparison images...")
    
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
        """Draw bounding boxes and IDs on image"""
        img = image.copy()
        
        # Add title
        cv2.putText(img, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   1.0, (0, 0, 255), 2, cv2.LINE_AA)
        
        for track in tracks:
            track_id = track['id']
            x, y, w, h = track['bbox']
            
            # Default color is red
            color = (0, 0, 255)
            
            # For HIT side, check if this detection is interpolated/new (not in original)
            if use_hit_colors and original_detections_set is not None:
                # Get current frame from context (we'll pass it)
                frame_key = (track.get('frame', 0), int(x), int(y), int(w), int(h))
                if frame_key not in original_detections_set:
                    # This is an interpolated/new detection - color it pink
                    color = (255, 37, 255)  # Pink in BGR
            
            # Draw bounding box
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
        
        # Add track count in red
        count_text = f'Tracks: {len(set(t["id"] for t in tracks))}'
        cv2.putText(img, count_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX,
                   0.8, (0, 0, 255), 2, cv2.LINE_AA)
        
        return img
    
    # Load both tracking results
    original_tracks = load_tracks_by_frame(RESULTS_FILE)
    hit_tracks = load_tracks_by_frame(hit_output_file)
    
    # Build mapping of original detections to track which ones are interpolated/new in HIT
    original_detections_set = set()
    with open(RESULTS_FILE, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            frame = int(parts[0])
            x, y, w, h = map(float, parts[2:6])
            # Store frame and approximate position
            original_detections_set.add((frame, int(x), int(y), int(w), int(h)))
    
    # Get frame range
    all_frames = sorted(set(list(original_tracks.keys()) + list(hit_tracks.keys())))
    
    # Create output directory
    comparison_dir = f'runs/detect/comparison_{TRACKER_FOLDER_NAME}_vs_HIT/{VIDEO}'
    os.makedirs(comparison_dir, exist_ok=True)
    
    print(f"Processing {len(all_frames)} frames...")
    
    for frame_num in all_frames:
        # Load original frame image - try different naming formats
        frame_path = f"{FRAMES_FOLDER}/frame-{frame_num:04d}.jpg"
        if not os.path.exists(frame_path):
            frame_path = f"{FRAMES_FOLDER}/frame-{frame_num:04d}.png"
        if not os.path.exists(frame_path):
            frame_path = f"{FRAMES_FOLDER}/{frame_num:06d}.jpg"
        if not os.path.exists(frame_path):
            frame_path = f"{FRAMES_FOLDER}/{frame_num:06d}.png"
        
        if not os.path.exists(frame_path):
            print(f"Warning: Frame {frame_num} not found, skipping...")
            continue
        
        img = cv2.imread(frame_path)
        if img is None:
            continue
        
        # Add frame number to HIT tracks for detection checking
        for track in hit_tracks.get(frame_num, []):
            track['frame'] = frame_num
        
        # Draw original tracking
        img_original = draw_tracks_on_image(
            img, 
            original_tracks.get(frame_num, []),
            f"{TRACKER_FOLDER_NAME.upper()}",
            use_hit_colors=False
        )
        
        # Draw HIT tracking (with pink for interpolated detections)
        img_hit = draw_tracks_on_image(
            img,
            hit_tracks.get(frame_num, []),
            f"{TRACKER_FOLDER_NAME.upper()} + HIT",
            use_hit_colors=True,
            original_detections_set=original_detections_set
        )
        
        # Concatenate side by side
        comparison = np.hstack([img_original, img_hit])
        
        # Save comparison image
        output_path = f"{comparison_dir}/frame_{frame_num:06d}.jpg"
        cv2.imwrite(output_path, comparison)
        
        if frame_num % 50 == 0:
            print(f"  Processed frame {frame_num}/{max(all_frames)}")
    
    print(f"\nComparison images saved to: {comparison_dir}")
    print(f"Total frames: {len(all_frames)}")
    
    # Optionally create a video from the comparison frames
    print("\nCreating comparison video...")
    
    # Get list of actual frame files
    frame_files = sorted([f for f in os.listdir(comparison_dir) if f.startswith('frame_') and f.endswith('.jpg')])
    
    if frame_files:
        first_frame_path = os.path.join(comparison_dir, frame_files[0])
        first_frame = cv2.imread(first_frame_path)
        
        if first_frame is not None:
            height, width = first_frame.shape[:2]
            video_path = os.path.join(comparison_dir, "comparison_video.mp4")
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(video_path, fourcc, 30.0, (width, height))
            
            for frame_file in frame_files:
                frame_path = os.path.join(comparison_dir, frame_file)
                frame = cv2.imread(frame_path)
                if frame is not None:
                    video_writer.write(frame)
            
            video_writer.release()
            print(f"Comparison video saved to: {video_path}")
    else:
        print("No comparison frames found to create video.")

# ==================== METRICS COMPARISON PDF ====================
print("\nGenerating metrics comparison PDF...")

def calculate_tracking_metrics(filepath):
    """Calculate basic tracking metrics from MOTChallenge format file"""
    detections = load_tracking_results(filepath)
    
    if not detections:
        return None
    
    # Group by track ID
    tracks = defaultdict(list)
    for det in detections:
        tracks[det[1]].append(det)
    
    # Calculate metrics
    total_tracks = len(tracks)
    total_detections = len(detections)
    
    # Track lengths
    track_lengths = [len(track_dets) for track_dets in tracks.values()]
    avg_track_length = np.mean(track_lengths)
    median_track_length = np.median(track_lengths)
    min_track_length = min(track_lengths)
    max_track_length = max(track_lengths)
    
    # Frame coverage
    frames_with_detections = sorted(set(det[0] for det in detections))
    first_frame = min(frames_with_detections)
    last_frame = max(frames_with_detections)
    total_frames = last_frame - first_frame + 1
    frames_with_objects = len(frames_with_detections)
    
    # Detections per frame
    detections_per_frame = defaultdict(int)
    for det in detections:
        detections_per_frame[det[0]] += 1
    avg_detections_per_frame = np.mean(list(detections_per_frame.values()))
    
    # Track fragmentation (count gaps in tracks)
    total_gaps = 0
    for track_id, track_dets in tracks.items():
        sorted_frames = sorted([det[0] for det in track_dets])
        for i in range(1, len(sorted_frames)):
            gap = sorted_frames[i] - sorted_frames[i-1] - 1
            if gap > 0:
                total_gaps += gap
    
    return {
        'total_tracks': total_tracks,
        'total_detections': total_detections,
        'avg_track_length': avg_track_length,
        'median_track_length': median_track_length,
        'min_track_length': min_track_length,
        'max_track_length': max_track_length,
        'total_frames': total_frames,
        'frames_with_objects': frames_with_objects,
        'avg_detections_per_frame': avg_detections_per_frame,
        'total_gaps': total_gaps,
        'track_lengths': track_lengths
    }

def evaluate_with_trackeval(tracker_folder, tracker_name):
    """Evaluate tracker using TrackEval to get HOTA, MOTA, IDF1, etc."""
    try:
        import trackeval
        
        # Configure dataset - GT is in videos/{VIDEO}/gt/gt.txt
        # Tracker is in data/trackers/moranget/moranget-test/{tracker_name}/data/{VIDEO}.txt
        dataset_config = {
            'GT_FOLDER': 'videos',
            'TRACKERS_FOLDER': 'data/trackers/moranget/moranget-test',
            'OUTPUT_FOLDER': None,  # Don't save outputs
            'TRACKERS_TO_EVAL': [tracker_name],
            'CLASSES_TO_EVAL': ['orange'],
            'BENCHMARK': '',
            'SPLIT_TO_EVAL': 'test',
            'SEQ_INFO': {VIDEO: len([f for f in os.listdir(FRAMES_FOLDER) if f.endswith(('.jpg', '.png'))])},
            'SKIP_SPLIT_FOL': True,  # Skip the benchmark-split folder structure
            'GT_LOC_FORMAT': '{gt_folder}/{seq}/gt/gt.txt',
            'PRINT_CONFIG': False,
            'DO_PREPROC': True,
            'TRACKER_SUB_FOLDER': 'data',
            'OUTPUT_SUB_FOLDER': '',
            'TRACKER_DISPLAY_NAMES': None,
        }
        
        # Configure evaluator
        eval_config = {
            'USE_PARALLEL': False,
            'NUM_PARALLEL_CORES': 1,
            'BREAK_ON_ERROR': False,
            'PRINT_RESULTS': False,
            'PRINT_CONFIG': False,
            'TIME_PROGRESS': False,
            'OUTPUT_SUMMARY': False,
            'OUTPUT_DETAILED': False,
            'PLOT_CURVES': False,
        }
        
        # Create dataset and metrics
        dataset = trackeval.datasets.MotChallenge2DBox(dataset_config)
        metrics_list = [
            trackeval.metrics.HOTA(),
            trackeval.metrics.CLEAR(),
            trackeval.metrics.Identity(),
        ]
        
        # Run evaluation
        evaluator = trackeval.Evaluator(eval_config)
        output_res, output_msg = evaluator.evaluate([dataset], metrics_list)
        
        # Extract metrics
        benchmark_key = list(output_res.keys())[0]  # Get the actual benchmark key
        res = output_res[benchmark_key][tracker_name]['COMBINED_SEQ']['orange']
        
        # Convert numpy arrays to scalars (use .item() for arrays)
        def to_scalar(val):
            if hasattr(val, 'item'):
                return val.item() if val.size == 1 else float(val.flatten()[0])
            return val
        
        return {
            'HOTA': to_scalar(res['HOTA']['HOTA']),
            'DetA': to_scalar(res['HOTA']['DetA']),
            'AssA': to_scalar(res['HOTA']['AssA']),
            'MOTA': to_scalar(res['CLEAR']['MOTA']),
            'MOTP': to_scalar(res['CLEAR']['MOTP']),
            'IDF1': to_scalar(res['Identity']['IDF1']),
            'IDP': to_scalar(res['Identity']['IDP']),
            'IDR': to_scalar(res['Identity']['IDR']),
            'MT': int(to_scalar(res['CLEAR']['MT'])),
            'ML': int(to_scalar(res['CLEAR']['ML'])),
            'FP': int(to_scalar(res['CLEAR']['CLR_FP'])),
            'FN': int(to_scalar(res['CLEAR']['CLR_FN'])),
            'IDs': int(to_scalar(res['Count']['IDs'])),
            'Frag': int(to_scalar(res['CLEAR']['Frag'])),
        }
    except Exception as e:
        import traceback
        print(f"Warning: Could not run TrackEval: {e}")
        print(traceback.format_exc())
        return None

# Calculate basic metrics for both trackers
original_metrics = calculate_tracking_metrics(RESULTS_FILE)
hit_metrics = calculate_tracking_metrics(hit_output_file)

# Calculate HOTA/MOTA/IDF1 metrics using TrackEval
print("Running TrackEval for original tracker...")
original_eval = evaluate_with_trackeval('data/trackers/moranget/moranget-test', TRACKER_FOLDER_NAME)

print("Running TrackEval for HIT tracker...")
hit_eval = evaluate_with_trackeval('data/trackers/moranget/moranget-test', f'{TRACKER_FOLDER_NAME}_HIT')

# Create PDF report
pdf_path = f'runs/detect/comparison_{TRACKER_FOLDER_NAME}_vs_HIT/{VIDEO}/metrics_comparison.pdf'
os.makedirs(os.path.dirname(pdf_path), exist_ok=True)

with PdfPages(pdf_path) as pdf:
    # Page 1: MOT Metrics (HOTA, MOTA, IDF1)
    if original_eval and hit_eval:
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis('tight')
        ax.axis('off')
        
        fig.suptitle(f'MOT Metrics Comparison: {TRACKER_FOLDER_NAME.upper()} vs HIT\nVideo: {VIDEO}', 
                     fontsize=16, fontweight='bold')
        
        mot_metrics_data = [
            ['Metric', f'{TRACKER_FOLDER_NAME.upper()}', f'{TRACKER_FOLDER_NAME.upper()} + HIT', 'Improvement'],
            ['HOTA ↑', f"{original_eval['HOTA']:.1f}%", f"{hit_eval['HOTA']:.1f}%",
             f"{hit_eval['HOTA'] - original_eval['HOTA']:+.1f}%"],
            ['DetA ↑', f"{original_eval['DetA']:.1f}%", f"{hit_eval['DetA']:.1f}%",
             f"{hit_eval['DetA'] - original_eval['DetA']:+.1f}%"],
            ['AssA ↑', f"{original_eval['AssA']:.1f}%", f"{hit_eval['AssA']:.1f}%",
             f"{hit_eval['AssA'] - original_eval['AssA']:+.1f}%"],
            ['MOTA ↑', f"{original_eval['MOTA']:.1f}%", f"{hit_eval['MOTA']:.1f}%",
             f"{hit_eval['MOTA'] - original_eval['MOTA']:+.1f}%"],
            ['MOTP ↑', f"{original_eval['MOTP']:.1f}%", f"{hit_eval['MOTP']:.1f}%",
             f"{hit_eval['MOTP'] - original_eval['MOTP']:+.1f}%"],
            ['IDF1 ↑', f"{original_eval['IDF1']:.1f}%", f"{hit_eval['IDF1']:.1f}%",
             f"{hit_eval['IDF1'] - original_eval['IDF1']:+.1f}%"],
            ['IDP ↑', f"{original_eval['IDP']:.1f}%", f"{hit_eval['IDP']:.1f}%",
             f"{hit_eval['IDP'] - original_eval['IDP']:+.1f}%"],
            ['IDR ↑', f"{original_eval['IDR']:.1f}%", f"{hit_eval['IDR']:.1f}%",
             f"{hit_eval['IDR'] - original_eval['IDR']:+.1f}%"],
            ['MT (Mostly Tracked) ↑', f"{original_eval['MT']}", f"{hit_eval['MT']}",
             f"{hit_eval['MT'] - original_eval['MT']:+d}"],
            ['ML (Mostly Lost) ↓', f"{original_eval['ML']}", f"{hit_eval['ML']}",
             f"{hit_eval['ML'] - original_eval['ML']:+d}"],
            ['FP (False Positives) ↓', f"{original_eval['FP']}", f"{hit_eval['FP']}",
             f"{hit_eval['FP'] - original_eval['FP']:+d}"],
            ['FN (False Negatives) ↓', f"{original_eval['FN']}", f"{hit_eval['FN']}",
             f"{hit_eval['FN'] - original_eval['FN']:+d}"],
            ['ID Switches ↓', f"{original_eval['IDs']}", f"{hit_eval['IDs']}",
             f"{hit_eval['IDs'] - original_eval['IDs']:+d}"],
            ['Fragmentations ↓', f"{original_eval['Frag']}", f"{hit_eval['Frag']}",
             f"{hit_eval['Frag'] - original_eval['Frag']:+d}"],
        ]
        
        table = ax.table(cellText=mot_metrics_data, cellLoc='center', loc='center',
                         colWidths=[0.3, 0.2, 0.2, 0.2])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Style header row
        for i in range(4):
            table[(0, i)].set_facecolor('#4472C4')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Color improvement column (green=better, red=worse)
        for i in range(1, len(mot_metrics_data)):
            improvement_str = mot_metrics_data[i][3]
            
            # Determine if metric should be higher or lower
            is_lower_better = i >= 11  # ML, FP, FN, IDs, Frag should be lower
            
            if '+' in improvement_str and '0.0' not in improvement_str and '+0' not in improvement_str:
                if is_lower_better:
                    table[(i, 3)].set_facecolor('#FFE6E6')  # Light red (worse)
                else:
                    table[(i, 3)].set_facecolor('#E6F4EA')  # Light green (better)
            elif '-' in improvement_str and not improvement_str.startswith('−'):
                if is_lower_better:
                    table[(i, 3)].set_facecolor('#E6F4EA')  # Light green (better)
                else:
                    table[(i, 3)].set_facecolor('#FFE6E6')  # Light red (worse)
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
    
    # Page 2: Basic Tracking Statistics
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis('tight')
    ax.axis('off')
    
    # Title
    fig.suptitle(f'Basic Tracking Statistics: {TRACKER_FOLDER_NAME.upper()} vs HIT\nVideo: {VIDEO}', 
                 fontsize=16, fontweight='bold')
    
    # Create comparison table
    metrics_data = [
        ['Metric', f'{TRACKER_FOLDER_NAME.upper()}', f'{TRACKER_FOLDER_NAME.upper()} + HIT', 'Improvement'],
        ['Total Tracks', f"{original_metrics['total_tracks']}", f"{hit_metrics['total_tracks']}", 
         f"{hit_metrics['total_tracks'] - original_metrics['total_tracks']:+d}"],
        ['Total Detections', f"{original_metrics['total_detections']}", f"{hit_metrics['total_detections']}", 
         f"{hit_metrics['total_detections'] - original_metrics['total_detections']:+d}"],
        ['Avg Track Length', f"{original_metrics['avg_track_length']:.1f}", f"{hit_metrics['avg_track_length']:.1f}",
         f"{hit_metrics['avg_track_length'] - original_metrics['avg_track_length']:+.1f}"],
        ['Median Track Length', f"{original_metrics['median_track_length']:.0f}", f"{hit_metrics['median_track_length']:.0f}",
         f"{hit_metrics['median_track_length'] - original_metrics['median_track_length']:+.0f}"],
        ['Min Track Length', f"{original_metrics['min_track_length']}", f"{hit_metrics['min_track_length']}",
         f"{hit_metrics['min_track_length'] - original_metrics['min_track_length']:+d}"],
        ['Max Track Length', f"{original_metrics['max_track_length']}", f"{hit_metrics['max_track_length']}",
         f"{hit_metrics['max_track_length'] - original_metrics['max_track_length']:+d}"],
        ['Avg Detections/Frame', f"{original_metrics['avg_detections_per_frame']:.1f}", 
         f"{hit_metrics['avg_detections_per_frame']:.1f}",
         f"{hit_metrics['avg_detections_per_frame'] - original_metrics['avg_detections_per_frame']:+.1f}"],
        ['Total Gaps in Tracks', f"{original_metrics['total_gaps']}", f"{hit_metrics['total_gaps']}",
         f"{hit_metrics['total_gaps'] - original_metrics['total_gaps']:+d}"],
    ]
    
    table = ax.table(cellText=metrics_data, cellLoc='center', loc='center',
                     colWidths=[0.3, 0.2, 0.2, 0.2])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style header row
    for i in range(4):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Color improvement column
    for i in range(1, len(metrics_data)):
        improvement_value = metrics_data[i][3]
        if '+' in improvement_value:
            if i == 1:  # Total Tracks - lower is better
                table[(i, 3)].set_facecolor('#FFE6E6')  # Light red
            else:  # Other metrics - higher is better
                table[(i, 3)].set_facecolor('#E6F4EA')  # Light green
        elif '-' in improvement_value and improvement_value != '0':
            if i == 1 or i == 8:  # Total Tracks or Gaps - lower is better
                table[(i, 3)].set_facecolor('#E6F4EA')  # Light green
            else:
                table[(i, 3)].set_facecolor('#FFE6E6')  # Light red
    
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()
    
    # Page 3: Track Length Distribution
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 6))
    
    ax1.hist(original_metrics['track_lengths'], bins=30, color='red', alpha=0.7, edgecolor='black')
    ax1.set_title(f'{TRACKER_FOLDER_NAME.upper()}\nTrack Length Distribution')
    ax1.set_xlabel('Track Length (frames)')
    ax1.set_ylabel('Number of Tracks')
    ax1.grid(True, alpha=0.3)
    
    ax2.hist(hit_metrics['track_lengths'], bins=30, color='blue', alpha=0.7, edgecolor='black')
    ax2.set_title(f'{TRACKER_FOLDER_NAME.upper()} + HIT\nTrack Length Distribution')
    ax2.set_xlabel('Track Length (frames)')
    ax2.set_ylabel('Number of Tracks')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close()

print(f"Metrics comparison PDF saved to: {pdf_path}")

print(f"\n{'='*60}")
print("PROCESSING COMPLETE!")
print(f"{'='*60}")
