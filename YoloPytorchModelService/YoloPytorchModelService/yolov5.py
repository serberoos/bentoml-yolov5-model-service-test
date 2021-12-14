# -*- coding: utf-8 -*-
from bentoml import env, artifacts, api, BentoService
import torch
import torchvision
from bentoml.adapters import ImageInput
from bentoml.frameworks.pytorch import PytorchModelArtifact

#PATH = '/usr/src/app/runs/train/train_result_task_0/weights/'

#model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
#model.load_state_dict(torch.load(PATH + 'best.pt'), strict=False)

@env(infer_pip_packages=True)
@artifacts([PytorchModelArtifact('yolo_model')]) 
class YoloPytorchModelService(BentoService): 

    @api(input=ImageInput(), batch=True) 
    def predict(self, imgs):
        outputs = self.artifacts.model(imgs) 
        return outputs

   # svc = YoloPytorchModelService()

    #Pytorch model can be packed directly.
    #svc.pack('yolo_model',yolo_model)

