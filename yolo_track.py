from ultralytics import YOLO
import os

YOLO_MODEL_PATH = "yolo_v5n_best.pt" # modelo yolo treinado

VIDEO = "V01" # escolha o vídeo

FRAMES_FOLDER = f"./videos/{VIDEO}/images" #path onde estão os frames dos vídeos

IMAGE_SIZE = 1920 # Usar 1920 ou 3840

TRACKER_FOLDER_NAME = 'botSort_optimized' # Nome otimizado do tracker
RESULTS_FILE = f'data/trackers/moranget/moranget-test/{TRACKER_FOLDER_NAME}/data/{VIDEO}.txt' # Arquivo onde será salvas as detecções e tracking

directory = os.path.dirname(RESULTS_FILE)
os.makedirs(directory, exist_ok=True)

# Carrega um modelo YOLO pré-treinado
model = YOLO(YOLO_MODEL_PATH) 

# Parâmetros otimizados para tracking
# - conf: threshold de confiança reduzido para capturar mais detecções
# - iou: threshold de IoU para NMS ajustado
# - persist: mantém os IDs dos tracks entre frames
# - verbose: mostra progresso
results = model.track(
    source=FRAMES_FOLDER, 
    save=True, 
    save_txt=False, 
    imgsz=IMAGE_SIZE, 
    conf=0.40,  # Ultra aggressive
    iou=0.40,   # Very strict NMS
    tracker="botsort-config-v4.yaml",
    persist=True,  # Mantém IDs consistentes
    verbose=True   # Mostra progresso
)

# Salva resultados no formato MOTChallenge
with open(RESULTS_FILE, 'w') as f:
    for frame_id, result in enumerate(results, start=1):
        # Verifica se há detecções e se têm IDs de track
        if result.boxes is not None and len(result.boxes) > 0:
            for box in result.boxes:
                # Verifica se o box tem ID de track
                if box.id is not None:
                    bbox = box.xyxy[0].tolist()  # Convert from tensor to list
                    track_id = int(box.id.item())  # Get track id
                    conf = box.conf.item()  # Get confidence score
                    
                    # Formato MOTChallenge: <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
                    width = bbox[2] - bbox[0]
                    height = bbox[3] - bbox[1]
                    
                    f.write(f'{frame_id},{track_id},{bbox[0]:.2f},{bbox[1]:.2f},{width:.2f},{height:.2f},{conf:.4f},-1,-1,-1\n')

print(f"Tracking completo! Resultados salvos em: {RESULTS_FILE}")