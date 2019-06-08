# Imports here
import torch 
#import helper 
from torch import nn
import torch.nn.functional as F
from torchvision import datasets, transforms , models
from torch.autograd import Variable
import matplotlib.pyplot as plt
from torch import optim 
import time
from PIL import Image
import numpy as np
from matplotlib.ticker import FormatStrFormatter
from collections import OrderedDict
import json

#global variables 
data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

#Structures 
structures ={'vgg16':25088,'densenet161':2208,'resnet18':25088,'alexnet':1024}


#Methos data transforms 
def data_transforms():
    data_transforms = {
                    'train':transforms.Compose([
                                       transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])]),
                   'valid':transforms.Compose([transforms.Resize(256),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])]),
                   'test' :transforms.Compose([transforms.Resize(256),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])                         
                  }



    # TODO: Load the datasets with ImageFolder
    image_datasets  = {
                        'train':datasets.ImageFolder(data_dir + '/test', transform=data_transforms['train']),
                        'valid':datasets.ImageFolder(data_dir + '/valid', transform=data_transforms['valid']),
                        'test' :datasets.ImageFolder(data_dir + '/test', transform=data_transforms['test']) 
                    }

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    dataloaders = { 'train':torch.utils.data.DataLoader(image_datasets['train'], batch_size=64, shuffle=True),
                'valid':torch.utils.data.DataLoader(image_datasets['valid'], batch_size=32, shuffle=True),
                'test' :torch.utils.data.DataLoader(image_datasets['test'], batch_size=20, shuffle=True)
             }
    
    return data_transforms,image_datasets,dataloaders

#label mapping 
def label_mapping(filename):
    with open(filename, 'r') as f:
        cat_to_name = json.load(f)
    return cat_to_name

#Modal Setup 
def modelSetup(structure='vgg16'):
    if structure.lower()=='vgg16':
        model =  models.vgg16(pretrained=True)
    elif structure.lower()=='resnet18':
        model =  models.resnet18(pretrained=True)
    elif structure.lower()=='alexnet':
         model =  models.alexnet(pretrained=True)
    elif structure.lower()=='densenet161':
         model =  models.densenet161(pretrained=True)       
    return model

#frozen parameters
def frozeParameters(model):
    #Paramter frozenn 
    for param in model.parameters():
        param.requires_grad = False

#Make_classifier Method
def make_classifier(structure,model,device,hidden_layer=120,dropout=0.5,lr=0.001):
    if hidden_layer < 0:
        classifier = nn.Sequential(OrderedDict([
                          ('drop0',nn.Dropout(dropout)),
                          ('inputs', nn.Linear(structures[structure], hidden_layer)),
                          ('relu1', nn.ReLU()),
                          ('hidden_layer1', nn.Linear(hidden_layer, 90)),
                          ('relu2', nn.ReLU()),
                          ('hidden_layer2', nn.Linear(90, 80)),
                          ('relu3', nn.ReLU()),
                          ('hidden_layer3', nn.Linear(80, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    else:
        classifier = nn.Sequential(OrderedDict([
                          ('drop0',nn.Dropout(dropout)),
                          ('inputs', nn.Linear(structures[structure], hidden_layer)),
                          ('relu1', nn.ReLU()),
                          ('hidden_layer1', nn.Linear(hidden_layer, 90)),
                          ('relu2', nn.ReLU()),
                          ('hidden_layer2', nn.Linear(90, 80)),
                          ('relu3', nn.ReLU()),
                          ('hidden_layer3', nn.Linear(80, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
        
        model.classifier = classifier
        criterion = nn.NLLLoss()
        optimizer = optim.Adam(model.classifier.parameters(), lr )
        model.to(device)
    return classifier,model,criterion,optimizer


#function trainModel
def train_model(dataloader,optimizer,model,device,epochs,criterion):
    steps = 0
    running_loss = 0
    print_every = 5
    train_losses, test_losses = [], []
    for epoch in range(epochs):
        for inputs, labels in dataloader['train']:
            steps += 1
            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                test_loss = 0
                accuracy = 0
                model.eval()
                
                with torch.no_grad():
                    for inputs, labels in dataloader['test']:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)

                        test_loss += batch_loss.item()

                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Test loss: {test_loss/len(dataloader['train']):.3f}.. "
                      f"Test accuracy: {accuracy/len(dataloader['test']):.3f}")
                running_loss = 0
                model.train()


# TODO: Do validation on the test set
def validation_accuracy(dataloader,model,device,lr=0.001):
    correct = 0
    total = 0
    criterion = nn.NLLLoss()
    # Only train the classifier parameters, feature parameters are frozen
    optimizer = optim.Adam(model.classifier.parameters(), lr)
    model.to(device)  
    with torch.no_grad():
        for ii, (inputs, labels) in enumerate(dataloader):
            # Move input and label tensors to the GPU
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
    print('Accuracy of the networks:%d %%' % (100 * correct / total))

# TODO: Save the checkpoint 
def saveCheckPoint(model,dataloaders,epochs,optimizer):
    model.class_to_idx = dataloaders['train'].class_to_idx
    checkpoint = {'input_size': 25088,
              'output_size': 102,
              'epochs': epochs,
              'batch_size': 64,
              'model': model,
              'classifier': model.classifier,
              'optimizer': optimizer.state_dict(),
              'state_dict': model.state_dict(),
              'class_to_idx': model.class_to_idx
             }
    torch.save(checkpoint, 'checkpoint.pth')
    
# TODO: Write a function that loads a checkpoint and rebuilds the model
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = checkpoint['model']
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    optimizer = checkpoint['optimizer']
    epochs = checkpoint['epochs']
    
    for param in model.parameters():
        param.requires_grad = False
        
    return model, checkpoint['class_to_idx']

#Process images 

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    pil_image = Image.open(image)
    preprocesses=transforms.Compose([transforms.Resize(256),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])  
    img_tensor = preprocesses(pil_image)
    return img_tensor

#Imshow 
def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

#Predict 
def predict(image, model,device, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    model.to(device)
    
     # Convert 2D image to 1D vector
    image = np.expand_dims(image, 0)
    
    image = torch.from_numpy(image)
    
    model.eval()
    inputs = Variable(image).to(device)
    logits = model.forward(inputs)
        
    probability = F.softmax(logits,dim=1)
    topk = probability.cpu().topk(topk)
    
    return (e.data.numpy().squeeze().tolist() for e in topk)


#Display an image along with the top 5 classes
def check_sanity(image,probs,classes,cat_to_name):
    
    plt.rcParams["figure.figsize"] = (10,5)
    plt.subplot(211)
    #print(fig)
   
    index = 1
    #probs,classes =  predict(image,model,device,topk)
    #image = process_image(image_path)
    
    #print(probs)
    #print(classes)
    
    ax1 = imshow(image, ax = plt)
    ax1.axis('off')
    ax1.title(cat_to_name[str(index)])
    ax1.show()
   
    y_pos = np.arange(len(probs))
    flower_names = [cat_to_name[str(index)] for index in classes]
    #print(flower_names)
    
    #N=float(len(b))
    fig,ax2 = plt.subplots(figsize=(8,3))
    width = 0.8
    ax2.barh(y_pos, probs, align='center')
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(flower_names)
    ax2.invert_yaxis()  # labels read top-to-bottom
    ax2.set_title('Class Probability')
    plt.show()