import torch

class Darknet:
    def __init__(self):
        self.model = torch.hub.load('ultralytics/yolov5','yolov5s',pretrained=True)

    def detect(self, image):
        results = self.model(image)
        return results