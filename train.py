import matplotlib.pyplot as plt
import numpy as np
import time
import torch
from torch import nn
from torch import tensor
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
import torchvision.models as models
import argparse

import predict

arg = argparse.ArgumentParser(description='train.py')
arg.add_argument('data_dir',nargs='*',default="flowers")
arg.add_argument('--device',dest='device',default="gpu")
arg.add_argument('--save_dir',dest='save_dir',default="checkpoint.pth")
arg.add_argument('--arch',dest='arch',default="vgg16")
arg.add_argument('--learning_rate',dest='learning_rate',default="0.01")
arg.add_argument('--hidden_units',dest='hidden_units',default="512")
arg.add_argument('--epochs',dest='epochs',default="12")
arg.add_argument('--dropout',dest='dropout',default="0.5")
arg.add_argument('--topk',dest='topk',default="5")
arg.add_argument('--cat_fileName',dest='cat_fileName',default="cat_to_name.json")
arg.add_argument('--image_path',dest='image_path', default='flowers/test/1/image_06743.jpg')

parse_results = arg.parse_args()

data_dir = parse_results.data_dir
save_dir = parse_results.save_dir
arch = parse_results.arch
learning_rate = float(parse_results.learning_rate)
hidden_units = int(parse_results.hidden_units)
epochs = int(parse_results.epochs)
device = parse_results.device
dropout = float(parse_results.dropout)
topk = int(parse_results.topk) #--topk 5 
cat_fileName = parse_results.cat_fileName #--cat_fileName cat_to_name.json
image_path =  parse_results.image_path

#LoadData 
data_transforms ,image_datasets, dataloaders = predict.data_transforms()
#build the Model 
model = predict.modelSetup(arch)
#froze paramter 
predict.frozeParameters(model)

#Getting device info.
device = torch.device("cuda:0" if device=='gpu' else "cpu")


#Build and trainning the classifer 
classifier,model,criterion,optimizer = predict.make_classifier(arch,model,device,hidden_units,dropout,learning_rate)
predict.train_model(dataloaders,optimizer,model,device,epochs,criterion)

#Save CheckPoint
predict.saveCheckPoint(model,image_datasets,epochs,optimizer)

#Accuracy 
predict.validation_accuracy(dataloaders["test"],model,device,learning_rate)

print('Model Trained')

#Get Category Name 
#--cat_fileName
#cat_to_name = predict.label_mapping(cat_fileName)




