from ultralytics_MB import YOLO
#import neptune
import os
import re


#neptune params
NEPTUNE_PROJECT = os.environ.get('NEPTUNE_PROJECT', '')
NEPTUNE_MODEL = os.environ.get('NEPTUNE_MODEl', '')

def parse_bands(bands_str):
    parsed_bands = []

    # Remove outer brackets if present
    bands_str = bands_str.strip('[]')

    # Use a regex to split, accounting for function-like expressions
    # Matches either integers or function-like strings
    bands_split = re.findall(r'\d+|diff\(\[.*?\],\[.*?\]\)|grey_diff\(\[.*?\],\[.*?\]\)', bands_str)

    for band in bands_split:
        band = band.strip()

        # Check if the band is an integer
        if band.isdigit():
            parsed_bands.append(int(band))

        # Check if it's a valid function-like expression
        elif re.match(r'(diff|grey_diff)\(\[[0-9,]+\],\[[0-9,]+\]\)', band):
            parsed_bands.append(band)

        # Invalid format
        else:
            raise ValueError(f"Unsupported band format: {band}")

    return parsed_bands

#model params
model_path = os.environ.get('MODEL_PATH', 'yolov9es.yaml')
dataset_path= os.environ.get('DATASET_PATH', 'your/ds/path')
epochs = int(os.environ.get('EPOCHS', 100))
batch_size = int(os.environ.get('BATCH_SIZE',8))
imgsz = int(os.environ.get('IMGSZ', 1024))
time = float(os.environ.get('TIME', 24))
adjust_layers = [int(i) for i in os.environ.get('ADJUST_LAYERS', "").split(",") if i != ""]

channels = int(os.environ.get('CHANNELS', 6))
dual_stream = os.environ.get('DUAL_STREAM', True)
dual_stream = dual_stream == "True" 
bands = parse_bands(os.environ.get("BANDS","[0,1,2,3,4,5]"))


print("-----------------PARAMS-----------------")
print(f"Model path: {model_path}")
print(f"Dataset path: {dataset_path}")
print(f"Epochs: {epochs}")
print(f"Batch size: {batch_size}")
print(f"Image size: {imgsz}")
print(f"Time: {time}")
print(f"Channels: {channels}")
print(f"Dual stream: {dual_stream}")
print(f"bands: {bands}")
print("----------------------------------------")

if os.environ.get('STORE','local') == "local":
    project=os.environ.get("STORE_PATH,'your/local/path'")
else:
    import neptune 
    project = NEPTUNE_PROJECT

#init model and dataset
model = YOLO(model_path,task="detect",verbose=False)
model.train(data=dataset_path, project=project, name=NEPTUNE_MODEL, epochs=epochs, batch=batch_size, imgsz=imgsz,time = time,single_cls=True,channels=channels,adjust_layers=adjust_layers,bands=bands,dual_stream=dual_stream,workers=10,device=0,)
