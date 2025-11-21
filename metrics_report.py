"""
Metrics Report Module
Generates PDF reports comparing tracking metrics between original and HIT-refined trackers.
"""

import os
import sys
import numpy as np
from collections import defaultdict
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

# Add trackeval to path
sys.path.insert(0, os.path.abspath('.'))


def calculate_tracking_metrics(filepath):
    """Calculate basic tracking metrics from MOTChallenge format file"""
    from hit_tracker import load_tracking_results
    
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


def evaluate_with_trackeval(tracker_name, video_name, frames_folder):
    """Evaluate tracker using TrackEval (HOTA, MOTA, IDF1)"""
    try:
        import trackeval
        
        dataset_config = {
            'GT_FOLDER': 'videos',
            'TRACKERS_FOLDER': 'data/trackers/moranget/moranget-test',
            'OUTPUT_FOLDER': None,
            'TRACKERS_TO_EVAL': [tracker_name],
            'CLASSES_TO_EVAL': ['orange'],
            'BENCHMARK': '',
            'SPLIT_TO_EVAL': 'test',
            'SEQS_TO_EVAL': [video_name],
            'SEQ_INFO': {video_name: len([f for f in os.listdir(frames_folder) if f.endswith(('.jpg', '.png'))])},
            'SKIP_SPLIT_FOL': True,
            'GT_LOC_FORMAT': '{gt_folder}/{seq}/gt/gt.txt',
            'PRINT_CONFIG': False,
            'DO_PREPROC': True,
            'TRACKER_SUB_FOLDER': 'data',
            'OUTPUT_SUB_FOLDER': '',
            'TRACKER_DISPLAY_NAMES': None,
        }
        
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
        
        dataset = trackeval.datasets.MotChallenge2DBox(dataset_config)
        metrics_list = [
            trackeval.metrics.HOTA(),
            trackeval.metrics.CLEAR(),
            trackeval.metrics.Identity(),
        ]
        
        evaluator = trackeval.Evaluator(eval_config)
        output_res, output_msg = evaluator.evaluate([dataset], metrics_list)
        
        benchmark_key = list(output_res.keys())[0]
        res = output_res[benchmark_key][tracker_name]['COMBINED_SEQ']['orange']
        
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


def print_metrics_table(tracker_name, video_name, frames_folder):
    """Print evaluation metrics in a formatted table"""
    eval_results = evaluate_with_trackeval(tracker_name, video_name, frames_folder)
    
    if not eval_results:
        print(f"Could not evaluate {tracker_name}")
        return None
    
    print(f"\n{'='*10} {tracker_name.upper()} {'='*10}")
    print("|  HOTA |  DetA |  AssA |  MOTA |  IDF1 |  IDP  |  IDR  |  IDSW |  LocA |   FP  |   FN  |")
    print(f"| {eval_results['HOTA']*100:5.2f} | {eval_results['DetA']*100:5.2f} | {eval_results['AssA']*100:5.2f} | "
          f"{eval_results['MOTA']*100:5.2f} | {eval_results['IDF1']*100:5.2f} | {eval_results['IDP']*100:5.2f} | "
          f"{eval_results['IDR']*100:5.2f} | {eval_results['IDs']:5d} | "
          f"{eval_results.get('MOTP', 0)*100:5.2f} | {eval_results['FP']:5d} | {eval_results['FN']:5d} |")
    print()
    
    return eval_results


def generate_metrics_pdf(original_results_file, hit_results_file, pdf_path, 
                        tracker_name, video_name, frames_folder, verbose=True):
    """Generate PDF report comparing tracking metrics"""
    
    if verbose:
        print("\nGenerating PDF report...")
    
    original_metrics = calculate_tracking_metrics(original_results_file)
    hit_metrics = calculate_tracking_metrics(hit_results_file)
    
    if verbose:
        print("Evaluating trackers...")
    original_eval = evaluate_with_trackeval(tracker_name, video_name, frames_folder)
    hit_eval = evaluate_with_trackeval(f'{tracker_name}_HIT', video_name, frames_folder)
    
    # Create PDF report
    os.makedirs(os.path.dirname(pdf_path), exist_ok=True)
    
    with PdfPages(pdf_path) as pdf:
        # Page 1: MOT Metrics (HOTA, MOTA, IDF1)
        if original_eval and hit_eval:
            fig, ax = plt.subplots(figsize=(11, 8.5))
            ax.axis('tight')
            ax.axis('off')
            
            fig.suptitle(f'MOT Metrics Comparison: {tracker_name.upper()} vs HIT\nVideo: {video_name}', 
                         fontsize=16, fontweight='bold')
            
            mot_metrics_data = [
                ['Metric', f'{tracker_name.upper()}', f'{tracker_name.upper()} + HIT', 'Improvement'],
                ['HOTA ↑', f"{original_eval['HOTA']*100:.1f}%", f"{hit_eval['HOTA']*100:.1f}%",
                 f"{(hit_eval['HOTA'] - original_eval['HOTA'])*100:+.1f}%"],
                ['DetA ↑', f"{original_eval['DetA']*100:.1f}%", f"{hit_eval['DetA']*100:.1f}%",
                 f"{(hit_eval['DetA'] - original_eval['DetA'])*100:+.1f}%"],
                ['AssA ↑', f"{original_eval['AssA']*100:.1f}%", f"{hit_eval['AssA']*100:.1f}%",
                 f"{(hit_eval['AssA'] - original_eval['AssA'])*100:+.1f}%"],
                ['MOTA ↑', f"{original_eval['MOTA']*100:.1f}%", f"{hit_eval['MOTA']*100:.1f}%",
                 f"{(hit_eval['MOTA'] - original_eval['MOTA'])*100:+.1f}%"],
                ['MOTP ↑', f"{original_eval['MOTP']*100:.1f}%", f"{hit_eval['MOTP']*100:.1f}%",
                 f"{(hit_eval['MOTP'] - original_eval['MOTP'])*100:+.1f}%"],
                ['IDF1 ↑', f"{original_eval['IDF1']*100:.1f}%", f"{hit_eval['IDF1']*100:.1f}%",
                 f"{(hit_eval['IDF1'] - original_eval['IDF1'])*100:+.1f}%"],
                ['IDP ↑', f"{original_eval['IDP']*100:.1f}%", f"{hit_eval['IDP']*100:.1f}%",
                 f"{(hit_eval['IDP'] - original_eval['IDP'])*100:+.1f}%"],
                ['IDR ↑', f"{original_eval['IDR']*100:.1f}%", f"{hit_eval['IDR']*100:.1f}%",
                 f"{(hit_eval['IDR'] - original_eval['IDR'])*100:+.1f}%"],
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
                is_lower_better = i >= 11
                
                if '+' in improvement_str and '0.0' not in improvement_str and '+0' not in improvement_str:
                    if is_lower_better:
                        table[(i, 3)].set_facecolor('#FFE6E6')
                    else:
                        table[(i, 3)].set_facecolor('#E6F4EA')
                elif '-' in improvement_str and not improvement_str.startswith('−'):
                    if is_lower_better:
                        table[(i, 3)].set_facecolor('#E6F4EA')
                    else:
                        table[(i, 3)].set_facecolor('#FFE6E6')
            
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
        
        # Page 2: Basic Tracking Statistics
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis('tight')
        ax.axis('off')
        
        fig.suptitle(f'Basic Tracking Statistics: {tracker_name.upper()} vs HIT\nVideo: {video_name}', 
                     fontsize=16, fontweight='bold')
        
        metrics_data = [
            ['Metric', f'{tracker_name.upper()}', f'{tracker_name.upper()} + HIT', 'Improvement'],
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
                if i == 1:
                    table[(i, 3)].set_facecolor('#FFE6E6')
                else:
                    table[(i, 3)].set_facecolor('#E6F4EA')
            elif '-' in improvement_value and improvement_value != '0':
                if i == 1 or i == 8:
                    table[(i, 3)].set_facecolor('#E6F4EA')
                else:
                    table[(i, 3)].set_facecolor('#FFE6E6')
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # Page 3: Track Length Distribution
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 6))
        
        ax1.hist(original_metrics['track_lengths'], bins=30, color='red', alpha=0.7, edgecolor='black')
        ax1.set_title(f'{tracker_name.upper()}\nTrack Length Distribution')
        ax1.set_xlabel('Track Length (frames)')
        ax1.set_ylabel('Number of Tracks')
        ax1.grid(True, alpha=0.3)
        
        ax2.hist(hit_metrics['track_lengths'], bins=30, color='blue', alpha=0.7, edgecolor='black')
        ax2.set_title(f'{tracker_name.upper()} + HIT\nTrack Length Distribution')
        ax2.set_xlabel('Track Length (frames)')
        ax2.set_ylabel('Number of Tracks')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()
    
    if verbose:
        print(f"PDF saved: {pdf_path}")
    
    return pdf_path
