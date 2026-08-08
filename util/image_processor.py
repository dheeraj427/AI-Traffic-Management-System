import torch

model = torch.hub.load('ultralytics/yolov5','yolov5s',pretrained=True)

vehicle_classes = ["car","bus","truck","motorcycle"]

def preparing_image(image):
    results = model(image)
    detections = results.pandas().xyxy[0]
    vehicles = detections[detections['name'].isin(vehicle_classes)]
    return len(vehicles)