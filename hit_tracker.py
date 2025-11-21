"""
HIT (Hierarchical IoU Tracking) Post-Processing Module
Implements hierarchical matching to refine tracking results and reduce ID switches.
"""

import numpy as np
from collections import defaultdict


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
                tracklets.append(Tracklet(track_id, current_tracklet))
                current_tracklet = [dets[i]]
            else:
                current_tracklet.append(dets[i])
        
        if current_tracklet:
            tracklets.append(Tracklet(track_id, current_tracklet))
    
    return tracklets


def hierarchical_matching(tracklets, delta_t, config):
    """Match tracklets at a specific temporal scale
    
    Uses two-stage matching:
    1. Motion-based matching for high-confidence tracklets
    2. IoU-based matching for remaining tracklets
    """
    high_conf_thr = config['high_conf_thr']
    motion_iou_thr = config['motion_iou_thr']
    iou_thr = config['iou_thr']
    
    high_conf_tracklets = [t for t in tracklets if t.max_conf >= high_conf_thr]
    unmatched_tracklets = tracklets.copy()
    matches = []
    
    # Stage 1: Motion-based matching for high-confidence tracklets
    for t1 in high_conf_tracklets:
        best_match = None
        best_iou = motion_iou_thr
        
        for t2 in unmatched_tracklets:
            if t2.track_id == t1.track_id:
                continue
            
            gap = t2.start_frame - t1.end_frame
            if gap < 1 or gap > delta_t:
                continue
            
            predicted_bbox = t1.predict_bbox(t2.start_frame)
            actual_bbox = t2.get_bbox_at_frame(t2.start_frame)
            
            if actual_bbox is None:
                continue
            
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


def fill_tracklet_gaps(tracklet, yolo_model=None, frames_folder=None, yolo_config=None):
    """Fill gaps within a single tracklet using YOLO detection
    
    Scans for gaps within a tracklet and attempts to detect missing objects
    using YOLO in predicted search areas.
    
    Args:
        tracklet: Tracklet to process
        yolo_model: YOLO model for detection
        frames_folder: Path to frame images  
        yolo_config: YOLO detection configuration dict
    
    Returns:
        Tracklet with gaps filled
    """
    if yolo_config is None:
        yolo_config = {}
    
    if yolo_model is None or frames_folder is None:
        return tracklet
    
    import cv2
    import os
    
    all_detections = tracklet.detections.copy()
    frames = sorted([d[0] for d in tracklet.detections])
    
    max_gap_size = yolo_config.get('max_gap_size', 10)
    detection_conf_thr = yolo_config.get('gap_detection_conf', 0.15)
    detection_iou = yolo_config.get('gap_detection_iou', 0.45)
    
    for i in range(len(frames) - 1):
        gap_size = frames[i+1] - frames[i] - 1
        
        if gap_size <= 0 or gap_size > max_gap_size:
            continue
        
        # Get bboxes before and after gap
        bbox_before = tracklet.get_bbox_at_frame(frames[i])
        bbox_after = tracklet.get_bbox_at_frame(frames[i+1])
        
        if bbox_before is None or bbox_after is None:
            continue
        
        gap_frames = range(frames[i] + 1, frames[i+1])
        
        for frame_num in gap_frames:
            # Load frame
            frame_path = f"{frames_folder}/frame-{frame_num:04d}.jpg"
            if not os.path.exists(frame_path):
                frame_path = f"{frames_folder}/{frame_num:06d}.jpg"
            if not os.path.exists(frame_path):
                continue
            
            # Predict where the orange should be
            j = frame_num - frames[i]
            total_gap = gap_size + 1
            alpha = j / total_gap
            pred_x = bbox_before[0] + alpha * (bbox_after[0] - bbox_before[0])
            pred_y = bbox_before[1] + alpha * (bbox_after[1] - bbox_before[1])
            pred_w = bbox_before[2] + alpha * (bbox_after[2] - bbox_before[2])
            pred_h = bbox_before[3] + alpha * (bbox_after[3] - bbox_before[3])
            
            # Expand search area
            search_margin = yolo_config.get('yolo_search_margin', 1.5)
            search_x = max(0, pred_x - pred_w * search_margin / 2)
            search_y = max(0, pred_y - pred_h * search_margin / 2)
            search_w = pred_w * (1 + search_margin)
            search_h = pred_h * (1 + search_margin)
            
            # Load and crop frame to search area
            img = cv2.imread(frame_path)
            if img is None:
                continue
            
            h_img, w_img = img.shape[:2]
            crop_x1 = int(max(0, search_x))
            crop_y1 = int(max(0, search_y))
            crop_x2 = int(min(w_img, search_x + search_w))
            crop_y2 = int(min(h_img, search_y + search_h))
            
            if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
                continue
            
            cropped_img = img[crop_y1:crop_y2, crop_x1:crop_x2]
            
            # Run YOLO on cropped region with aggressive settings
            results = yolo_model(cropped_img, conf=detection_conf_thr, iou=detection_iou, verbose=False)
            
            # Find best detection in search area
            best_detection = None
            best_iou = yolo_config.get('yolo_min_iou', 0.15)
            
            if results and len(results) > 0 and results[0].boxes is not None:
                for box in results[0].boxes:
                    # Convert coordinates back to full image
                    bbox_crop = box.xyxy[0].tolist()
                    det_x = bbox_crop[0] + crop_x1
                    det_y = bbox_crop[1] + crop_y1
                    det_w = bbox_crop[2] - bbox_crop[0]
                    det_h = bbox_crop[3] - bbox_crop[1]
                    det_conf = box.conf.item()
                    
                    # Calculate IoU with predicted position
                    iou_score = iou([pred_x, pred_y, pred_w, pred_h], [det_x, det_y, det_w, det_h])
                    
                    if iou_score > best_iou:
                        best_iou = iou_score
                        best_detection = [frame_num, tracklet.track_id, det_x, det_y, det_w, det_h, det_conf]
            
            # Add detection if found
            if best_detection is not None:
                all_detections.append(best_detection)
    
    return Tracklet(tracklet.track_id, all_detections)


def merge_tracklets(t1, t2, new_id=None, interpolate=False, yolo_model=None, frames_folder=None, yolo_config=None, verbose=False):
    """Merge two tracklets, optionally filling gaps with YOLO detection
    
    Preserves the track ID from the longer/higher confidence tracklet.
    Optionally uses YOLO to detect objects in gaps between tracklets.
    
    Args:
        t1, t2: Tracklets to merge
        new_id: Unused (kept for compatibility)
        interpolate: Enable YOLO gap detection
        yolo_model: YOLO model for detection
        frames_folder: Path to frame images
        yolo_config: YOLO detection configuration dict
        verbose: Enable debug output
    
    Returns:
        (merged_tracklet, num_added_detections)
    """
    if yolo_config is None:
        yolo_config = {}
    all_detections = t1.detections + t2.detections
    added_count = 0
    
    preserved_id = t1.track_id if len(t1.detections) >= len(t2.detections) else t2.track_id
    
    max_gap_size = yolo_config.get('max_gap_size', 10)
    detection_conf_thr = yolo_config.get('gap_detection_conf', 0.15)
    detection_iou = yolo_config.get('gap_detection_iou', 0.45)
    
    # Fill gaps using YOLO detection or interpolation
    if interpolate and t2.start_frame - t1.end_frame > 1:
        gap_size = t2.start_frame - t1.end_frame - 1
        
        if gap_size <= max_gap_size:
            gap_frames = range(t1.end_frame + 1, t2.start_frame)
            last_bbox = t1.detections[-1][2:6]
            first_bbox = t2.detections[0][2:6]
            
            if yolo_model is not None and frames_folder is not None:
                import cv2
                import os
                
                for frame_num in gap_frames:
                    # Try different frame naming formats
                    frame_path = f"{frames_folder}/frame-{frame_num:04d}.jpg"
                    if not os.path.exists(frame_path):
                        frame_path = f"{frames_folder}/{frame_num:06d}.jpg"
                    if not os.path.exists(frame_path):
                        continue
                    
                    # Predict where the orange should be
                    i = frame_num - t1.end_frame
                    total_gap = gap_size + 1
                    alpha = i / total_gap
                    pred_x = last_bbox[0] + alpha * (first_bbox[0] - last_bbox[0])
                    pred_y = last_bbox[1] + alpha * (first_bbox[1] - last_bbox[1])
                    pred_w = last_bbox[2] + alpha * (first_bbox[2] - last_bbox[2])
                    pred_h = last_bbox[3] + alpha * (first_bbox[3] - last_bbox[3])
                    
                    # Expand search area
                    search_margin = yolo_config.get('yolo_search_margin', 1.0)
                    search_x = max(0, pred_x - pred_w * search_margin)
                    search_y = max(0, pred_y - pred_h * search_margin)
                    search_w = pred_w * (1 + 2 * search_margin)
                    search_h = pred_h * (1 + 2 * search_margin)
                    
                    # Load and crop frame to search area
                    img = cv2.imread(frame_path)
                    if img is None:
                        continue
                    
                    h_img, w_img = img.shape[:2]
                    crop_x1 = int(max(0, search_x))
                    crop_y1 = int(max(0, search_y))
                    crop_x2 = int(min(w_img, search_x + search_w))
                    crop_y2 = int(min(h_img, search_y + search_h))
                    
                    if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
                        continue
                    
                    cropped_img = img[crop_y1:crop_y2, crop_x1:crop_x2]
                    
                    # Run YOLO on cropped region with aggressive settings
                    results = yolo_model(cropped_img, conf=detection_conf_thr, iou=detection_iou, verbose=False)
                    
                    best_detection = None
                    best_iou = yolo_config.get('yolo_min_iou', 0.2)
                    
                    if results and len(results) > 0 and results[0].boxes is not None:
                        for box in results[0].boxes:
                            # Convert coordinates back to full image
                            bbox_crop = box.xyxy[0].tolist()
                            det_x = bbox_crop[0] + crop_x1
                            det_y = bbox_crop[1] + crop_y1
                            det_w = bbox_crop[2] - bbox_crop[0]
                            det_h = bbox_crop[3] - bbox_crop[1]
                            det_conf = box.conf.item()
                            
                            # Calculate IoU with predicted position
                            iou_score = iou([pred_x, pred_y, pred_w, pred_h], [det_x, det_y, det_w, det_h])
                            
                            if iou_score > best_iou:
                                best_iou = iou_score
                                best_detection = [frame_num, preserved_id, det_x, det_y, det_w, det_h, det_conf]
                    
                    if best_detection is not None:
                        all_detections.append(best_detection)
                        added_count += 1
                    elif gap_size <= 5:
                        # Fallback: simple interpolation for very small gaps
                        interp_conf = min(t1.detections[-1][6], t2.detections[0][6]) * 0.6
                        all_detections.append([frame_num, preserved_id, pred_x, pred_y, pred_w, pred_h, interp_conf])
                        added_count += 1
            else:
                # Fallback: simple interpolation for small gaps
                if gap_size <= 5:
                    for i, frame in enumerate(gap_frames):
                        alpha = (i + 1) / (len(gap_frames) + 1)
                        interp_x = last_bbox[0] + alpha * (first_bbox[0] - last_bbox[0])
                        interp_y = last_bbox[1] + alpha * (first_bbox[1] - last_bbox[1])
                        interp_w = last_bbox[2] + alpha * (first_bbox[2] - last_bbox[2])
                        interp_h = last_bbox[3] + alpha * (first_bbox[3] - last_bbox[3])
                        interp_conf = min(t1.detections[-1][6], t2.detections[0][6]) * 0.7
                        
                        all_detections.append([frame, preserved_id, interp_x, interp_y, interp_w, interp_h, interp_conf])
                        added_count += 1
    
    for det in all_detections:
        det[1] = preserved_id
    
    return Tracklet(preserved_id, all_detections), added_count


def apply_hit_postprocessing(input_file, output_file, hit_config, yolo_config, yolo_model=None, frames_folder=None, verbose=True):
    """Apply HIT hierarchical matching to refine tracking results
    
    Processes tracking results through:
    1. Split into tracklets
    2. Fill gaps within tracklets using YOLO
    3. Hierarchical matching across temporal scales
    4. Deduplication
    
    Args:
        input_file: Input MOTChallenge format tracking file
        output_file: Output refined tracking file
        hit_config: HIT hierarchical matching configuration dict
        yolo_config: YOLO gap detection configuration dict (aggressive settings)
        yolo_model: Optional YOLO model for gap detection
        frames_folder: Path to video frames
        verbose: Enable progress messages
    
    Returns:
        Number of final tracklets
    """
    import os
    
    # Load detections
    detections = load_tracking_results(input_file)
    if verbose:
        print(f"Loaded {len(detections)} detections")
    
    max_gap = hit_config.get('max_gap_to_split', 1)
    tracklets = split_into_tracklets(detections, max_gap=max_gap)
    if verbose:
        print(f"Split into {len(tracklets)} tracklets")
    
    # Fill gaps within tracklets using YOLO with aggressive settings
    if yolo_model is not None and frames_folder is not None and hit_config.get('interpolate', False):
        filled_tracklets = []
        total_added = 0
        for t in tracklets:
            filled = fill_tracklet_gaps(
                t,
                yolo_model=yolo_model,
                frames_folder=frames_folder,
                yolo_config=yolo_config
            )
            total_added += len(filled.detections) - len(t.detections)
            filled_tracklets.append(filled)
        tracklets = filled_tracklets
        if verbose and total_added > 0:
            print(f"Filled {total_added} gaps in tracklets")
    
    # Hierarchical matching
    total_gap_fills = 0
    for delta_t in hit_config['delta_t_scales']:
        matches = hierarchical_matching(tracklets, delta_t, hit_config)
        if verbose:
            print(f"Matching at Δt={delta_t}: {len(matches)} matches")
        # Apply matches
        new_tracklets = []
        merged_tracklets = set()
        
        for t1, t2 in matches:
            if t1 in merged_tracklets or t2 in merged_tracklets:
                continue
            
            merged, added = merge_tracklets(
                t1, t2, None, 
                interpolate=hit_config['interpolate'],
                yolo_model=yolo_model,
                frames_folder=frames_folder,
                yolo_config=yolo_config,
                verbose=verbose
            )
            total_gap_fills += added
            new_tracklets.append(merged)
            merged_tracklets.add(t1)
            merged_tracklets.add(t2)
        
        for t in tracklets:
            if t not in merged_tracklets:
                new_tracklets.append(t)
        
        tracklets = new_tracklets
    
    if verbose and total_gap_fills > 0:
        print(f"Added {total_gap_fills} detections in merge gaps")
    
    # Filter short tracklets
    tracklets = [t for t in tracklets if len(t.detections) >= hit_config['min_tracklet_len']]
    
    # Collect and sort all detections
    all_detections = []
    for tracklet in sorted(tracklets, key=lambda t: t.start_frame):
        all_detections.extend(tracklet.detections)
    all_detections = sorted(all_detections, key=lambda x: (x[0], x[1]))
    
    # Remove duplicate (frame, track_id) pairs
    frame_track_detections = defaultdict(lambda: defaultdict(list))
    for det in all_detections:
        frame_track_detections[det[0]][det[1]].append(det)
    
    deduplicated_detections = []
    duplicate_count = 0
    
    for frame in sorted(frame_track_detections.keys()):
        for track_id in sorted(frame_track_detections[frame].keys()):
            dets = frame_track_detections[frame][track_id]
            if len(dets) > 1:
                duplicate_count += len(dets) - 1
                deduplicated_detections.append(max(dets, key=lambda x: x[6]))
            else:
                deduplicated_detections.append(dets[0])
    
    if verbose and duplicate_count > 0:
        print(f"Removed {duplicate_count} duplicates")
    
    # Write results
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        for det in deduplicated_detections:
            f.write(f'{int(det[0])},{int(det[1])},{det[2]:.2f},{det[3]:.2f},{det[4]:.2f},{det[5]:.2f},{det[6]:.4f},-1,-1,-1\n')
    
    if verbose:
        print(f"HIT complete: {len(tracklets)} tracks, {len(deduplicated_detections)} detections")
    
    return len(tracklets)
