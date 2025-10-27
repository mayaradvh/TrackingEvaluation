from ultralytics import YOLO
import os

YOLO_MODEL_PATH = "yolo_v5n_best.pt" # modelo yolo treinado

VIDEO = "V01" # escolha o vídeo

FRAMES_FOLDER = f"/home/mayarahoger/mestrado/datasets/videos_orange/moranget-test/{VIDEO}/images" #path onde estão os frames dos vídeos

IMAGE_SIZE = 1940 # Usar 1920 ou 3840

TRACKER_FOLDER_NAME = 'botSort1940' # pode escolher o nome que quiser. Aqui eu coloquei o nome do tracker padrão da YOLO e o tamanho da imagem
RESULTS_FILE = f'data/trackers/moranget/moranget-test/{TRACKER_FOLDER_NAME}/data/{VIDEO}.txt' # Arquivo onde será salvas as detecções e tracking

directory = os.path.dirname(RESULTS_FILE)
os.makedirs(directory, exist_ok=True)

# Carrega um modelo YOLO pré-treinado
model = YOLO(YOLO_MODEL_PATH) 

results = model.track(source=FRAMES_FOLDER, save=False, save_txt=False, imgsz=IMAGE_SIZE, conf=0.35)

with open(RESULTS_FILE, 'w') as f:
    for frame_id, result in enumerate(results):
        for box in result.boxes:
            bbox = box.xywh[0].tolist()  # Convert from tensor to list
            if not box.id:
                continue
            track_id = int(box.id.item())  # Get track id
            conf = box.conf.item()  # Get confidence score
            f.write(f'{frame_id+1},{track_id},{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]},1,1,1.0\n')