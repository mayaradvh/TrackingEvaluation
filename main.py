"""
Main Script for YOLO Tracking with HIT Post-Processing
Coordinates YOLO tracking, HIT refinement, visualization, and metrics reporting.
"""

from ultralytics import YOLO
import os

# Import custom modules
from hit_tracker import apply_hit_postprocessing
from video_comparison import create_comparison_video
from metrics_report import generate_metrics_pdf


# ==================== CONFIGURATION ====================
YOLO_MODEL_PATH = "yolo_v5n_best.pt"
VIDEO = "V04"
FRAMES_FOLDER = f"./videos/{VIDEO}/images"
IMAGE_SIZE = 3048  # Usar 1920 ou 3840

# TRACKER CONFIGURATION
TRACKER_CONFIG = "botsort-config-v4.yaml"
TRACKER_FOLDER_NAME = 'botsort_v4'
RESULTS_FILE = f'data/trackers/moranget/moranget-test/{TRACKER_FOLDER_NAME}/data/{VIDEO}.txt'

# EXECUTION MODE
# Options: "yolo_only", "hit_only", "yolo_and_hit"
EXECUTION_MODE = "yolo_and_hit"

# ADDITIONAL FLAGS
CREATE_COMPARISON_VIDEO = True  # Set to False to skip comparison visualization
GENERATE_PDF_REPORT = True  # Set to False to skip PDF metrics report
USE_YOLO_FOR_GAPS = True  # Use YOLO to detect in gap areas (more accurate than interpolation)

# HIT HIERARCHICAL MATCHING CONFIG
HIT_CONFIG = {
    'high_conf_thr': 0.55,           # High confidence threshold for motion-based matching
    'delta_t_scales': [1, 2, 4, 8, 16],  # Temporal scales for hierarchical matching
    'motion_iou_thr': 0.25,          # IoU threshold for motion-based matching
    'iou_thr': 0.25,                 # IoU threshold for appearance-based matching
    'interpolate': True,             # Enable gap filling with YOLO
    'min_tracklet_len': 1,           # Minimum tracklet length (1 = keep all)
    'max_gap_to_split': 1,           # Split tracklets at gaps > 1 frame
}

# HIT GAP DETECTION CONFIG (Aggressive settings for maximum detection)
HIT_YOLO_CONFIG = {
    'gap_detection_conf': 0.05,      # Very aggressive: detect almost anything
    'gap_detection_iou': 0.20,       # Low NMS: keep overlapping detections
    'yolo_search_margin': 3.0,       # Large search area: 300% expansion
    'yolo_min_iou': 0.15,            # Lenient validation: accept if reasonably close
    'max_gap_size': 15,              # Fill larger gaps (up to 15 frames)
}


# ==================== YOLO TRACKING ====================
def run_yolo_tracking():
    """Run YOLO tracking on video frames"""
    directory = os.path.dirname(RESULTS_FILE)
    os.makedirs(directory, exist_ok=True)
    
    if os.path.exists(RESULTS_FILE):
        print(f"Using existing tracking results: {RESULTS_FILE}")
        return
    
    print(f"Running YOLO tracking on {VIDEO}...")
    model = YOLO(YOLO_MODEL_PATH)
    
    results = model.track(
        source=FRAMES_FOLDER,
        save=True,
        save_txt=False,
        imgsz=IMAGE_SIZE,
        conf=0.3,
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
    
    print(f"Tracking saved: {RESULTS_FILE}")


# ==================== MAIN EXECUTION ====================
def main():
    """Main execution pipeline"""
    
    # Step 1: Run YOLO tracking (if needed)
    if EXECUTION_MODE in ["yolo_only", "yolo_and_hit"]:
        run_yolo_tracking()
        
        if EXECUTION_MODE == "yolo_only":
            print(f"\nTracking complete: {RESULTS_FILE}")
            return
    
    # Step 2: Apply HIT post-processing (if needed)
    if EXECUTION_MODE in ["hit_only", "yolo_and_hit"]:
        if not os.path.exists(RESULTS_FILE):
            print(f"Error: Tracking results not found: {RESULTS_FILE}")
            print("Please run with EXECUTION_MODE='yolo_only' or 'yolo_and_hit' first.")
            return
        
        print("\nApplying HIT post-processing...")
        
        # Load YOLO model for gap detection
        yolo_model_for_gaps = None
        if USE_YOLO_FOR_GAPS:
            yolo_model_for_gaps = YOLO(YOLO_MODEL_PATH)
        
        hit_output_file = f'data/trackers/moranget/moranget-test/{TRACKER_FOLDER_NAME}_HIT/data/{VIDEO}.txt'
        num_tracks = apply_hit_postprocessing(
            RESULTS_FILE, 
            hit_output_file, 
            HIT_CONFIG,
            HIT_YOLO_CONFIG,
            yolo_model=yolo_model_for_gaps,
            frames_folder=FRAMES_FOLDER,
            verbose=True
        )
        
        print(f"\n{'='*60}")
        print(f"HIT POST-PROCESSING COMPLETE")
        print(f"{'='*60}")
        print(f"Video: {VIDEO}")
        print(f"Refined results: {hit_output_file}")
        print(f"Final tracks: {num_tracks}")
        print(f"{'='*60}")
        
        # Print evaluation metrics in table format
        from metrics_report import print_metrics_table
        print_metrics_table(TRACKER_FOLDER_NAME, VIDEO, FRAMES_FOLDER)
        print_metrics_table(f"{TRACKER_FOLDER_NAME}_HIT", VIDEO, FRAMES_FOLDER)
        
        # Step 3: Create comparison video
        if CREATE_COMPARISON_VIDEO:
            comparison_dir = f'runs/detect/comparison_{TRACKER_FOLDER_NAME}_vs_HIT/{VIDEO}'
            create_comparison_video(
                RESULTS_FILE,
                hit_output_file,
                FRAMES_FOLDER,
                comparison_dir,
                TRACKER_FOLDER_NAME,
                VIDEO,
                verbose=True
            )
        
        # Step 4: Generate metrics PDF report
        if GENERATE_PDF_REPORT:
            pdf_path = f'runs/detect/comparison_{TRACKER_FOLDER_NAME}_vs_HIT/{VIDEO}/metrics_comparison.pdf'
            generate_metrics_pdf(
                RESULTS_FILE,
                hit_output_file,
                pdf_path,
                TRACKER_FOLDER_NAME,
                VIDEO,
                FRAMES_FOLDER,
                verbose=True
            )


if __name__ == "__main__":
    main()
