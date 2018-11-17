# Imports
import argparse
import json
import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

# Define network class
class Network(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers, drop_p=0.5):
        ''' Builds a feedforward network with arbitrary hidden layers.
        
            Arguments
            ---------
            input_size: integer, size of the input
            output_size: integer, size of the output layer
            hidden_layers: list of integers, the sizes of the hidden layers
            drop_p: float between 0 and 1, dropout probability
        '''
        super().__init__()
        # If no hidden layers
        if hidden_layers is None:
            self.output = nn.Linear(input_size, output_size)
            self.hidden_layers = None
        else:
            # Add the first layer, input to a hidden layer
            self.hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_layers[0])])
            # Add a variable number of more hidden layers
            layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
            self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])
            self.output = nn.Linear(hidden_layers[-1], output_size)
            self.dropout = nn.Dropout(p=drop_p)
        
    def forward(self, x):
        ''' Forward pass through the network, returns the output logits '''
        
        # If not None, forward through each layer in `hidden_layers`, with ReLU activation and dropout
        if self.hidden_layers is not None:
            for linear in self.hidden_layers:
                x = F.relu(linear(x))
                x = self.dropout(x)
        
        x = self.output(x)
        
        return F.log_softmax(x, dim=1)
def main():
    # Create a parser
    parser = argparse.ArgumentParser(description='Trains a new network on a dataset and save the model as a checkpoint, prints out training loss, validation loss, and validation accuracy as the network trains.')
    # Adding arguments
    parser.add_argument('data_dir', action= 'store', default='flowers', type= str, help= 'Directory of   the flower images.')
    parser.add_argument('--save_dir', action= 'store', default='.', type= str, dest= 'save_dir', help= 'Directory to save checkpoints.')
    parser.add_argument('--arch', action= 'store', default='vgg13', type= str, dest= 'arch', choices=[ 'vgg13', 'vgg16', 'resnet18'], help = "Choose architechture 'vgg13', 'vgg16', or 'resnet18'.")
    parser.add_argument("--learning_rate", action= "store", dest= "learning_rate", type= float, default= 0.01 , help = "Learning rate with default 0.01")
    parser.add_argument("--hidden_units", action= "store", dest= "hidden_layers", type= int, default= None, nargs='+', help = "Hidden layers as integer array.")
    parser.add_argument("--epochs", action= "store", dest= "epochs", type= int, default= 3 , help = "Number of epochs with default 3")
    parser.add_argument('--gpu_mode', default=False, type=bool, help='Use gpu mode for training.')

    # Getting arguments
    args = parser.parse_args()
    data_dir = args.data_dir
    save_dir = args.save_dir
    arch = args.arch
    learning_rate = args.learning_rate
    hidden_layers = args.hidden_layers
    epochs = args.epochs
    gpu_mode = args.gpu_mode

    # Defining training, validation and testing directories
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    # Define your transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], 
                                                                [0.229, 0.224, 0.225])])

    test_valid_transforms = transforms.Compose([transforms.Resize(256),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406], 
                                                                     [0.229, 0.224, 0.225])])

    # Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=test_valid_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_valid_transforms)

    # Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=32) 
    testloader = torch.utils.data.DataLoader(test_data, batch_size=32) 

    # Load in a mapping from category label to category name
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)

    # Enable GPU if it's available and gpu_mode is true
    if gpu_mode and torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    print('Current device: {}'.format(device))

    # Load the chosen pretrained model
    if arch == 'vgg13':
       model = models.vgg13(pretrained=True)
       # Getting the size of the input layer of the model's classifier
       input_size = model.classifier[0].in_features
    elif arch == 'vgg16':
         model = models.vgg16(pretrained=True)
         # Getting the size of the input layer of the model's classifier
         input_size = model.classifier[0].in_features
    elif arch == 'resnet18':
         model = models.resnet18(pretrained=True)
         # Getting the size of the input layer of the model's classifier
         input_size = model.fc.in_features
    # Output size is alwayes 102, since there're 102 flower types
    output_size = 102

    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False

    # Create the new classifier using Network class
    classifier = Network(input_size, output_size, hidden_layers, drop_p=0.5)
    # Define the criterion
    criterion = nn.NLLLoss()
    # Properly for each model, replace the pre-trianed model's classifier with the new one, and define optimizer
    if arch == 'vgg13' or arch == 'vgg16':
        model.classifier = classifier
        optimizer = optim.Adam(model.classifier.parameters(), learning_rate)
    elif arch == 'resnet18':
        model.fc = classifier
        optimizer = optim.Adam(model.fc.parameters(), learning_rate)
    else:
        print('Unrecognizable model!')

    # Training
    print('-----------' + ' Training ' + '-----------')
    model = training(model, trainloader, validloader, epochs, criterion, optimizer, device)
    # Testing
    print('-----------' + ' Testing ' + '-----------')
    testing(model, testloader, device)

    # Saving checkpoint
    print('-----------' + ' Saving checkpoint ' + '-----------') 
    model.class_to_idx = train_data.class_to_idx
    model.cat_to_name = cat_to_name

    checkpoint = {'input_size': input_size,
                  'output_size': output_size,
                  'hidden_layers': hidden_layers,
                  'state_dict': model.state_dict(),
                  'class_to_idx': model.class_to_idx,
                  'cat_to_name': cat_to_name,
                  'optimizer_state_dict': optimizer.state_dict(),
                  'epochs': epochs }

    torch.save(checkpoint, 'checkpoint.pth')
    print('Saved to the checkpoint: {}'.format('checkpoint.pth'))

def training(model, trainloader, validloader, epochs, criterion, optimizer, device='cpu'):
    epochs = epochs
    print_every = 40
    steps = 0
    model.to(device)

    for e in range(epochs):
        model.train()
        running_loss = 0
        for ii, (inputs, labels) in enumerate(trainloader):
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            # Forward and backward passes
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
            # Make sure network is in eval mode for inference
                model.eval()
            
            # Turn off gradients for validation, saves memory and computations
                with torch.no_grad():
                    valid_loss, accuracy = validation(model, validloader, criterion, device)
                
                print("Epoch: {}/{}.. ".format(e+1, epochs),
                      "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                      "Validation Loss: {:.3f}.. ".format(valid_loss/len(validloader)),
                      "Validation Accuracy: {:.3f}".format(accuracy/len(validloader)))
                
                running_loss = 0
            
                # Make sure training is back on
                model.train()
    return model            
# Implement a function for the validation pass
def validation(model, validloader, criterion, device):
    valid_loss = 0
    accuracy = 0
    model.to(device)
    for images, labels in validloader:
        images, labels = images.to(device), labels.to(device)
        output = model.forward(images)
        valid_loss += criterion(output, labels).item()

        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    
    return valid_loss, accuracy

def testing(model, testloader, device):    
    correct = 0
    total = 0
    model.to(device)
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))
    
if __name__ == '__main__':
    main() 