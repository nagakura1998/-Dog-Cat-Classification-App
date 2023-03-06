from torchvision import models, transforms, datasets
import torch
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import matplotlib.pyplot as plt
import os
import torchvision
import numpy as np
import torch.nn as nn
from collections import OrderedDict
train_dir = '/workspace/DogCatRecognition/archive/training_set/'
test_dir = '/workspace/DogCatRecognition/archive/test_set/'

class catdogDataset(Dataset):
    def __init__(self, file_list, dir, mode='train', transform = None):
        self.file_list = file_list
        self.dir = dir
        self.mode= mode
        self.transform = transform
        if self.mode == 'train':
            if 'dog' in self.file_list[0]:
                self.label = 1
            else:
                self.label = 0
            
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.dir, self.file_list[idx]))
        if self.transform:
            img = self.transform(img)
        if self.mode == 'train':
            img = img.numpy()
            return img.astype('float32'), self.label
        else:
            img = img.numpy()
            return img.astype('float32'), self.file_list[idx]

def train(target_epochs=10):
    num_train_samples = len(cat_files_train) + len(dog_files_train)
    num_test_samples = len(cat_files_test) + len(dog_files_test)
    lambda_ = 0.01  # L2 regulizer factor

    train_loss = []
    train_acc = []
    test_loss = []
    test_acc = []
    print("train on ", num_train_samples, " samples, test on ", num_test_samples, " samples")
    for epoch in range(target_epochs):
        epoch_loss = 0
        num_corrects = 0
        # train
        vgg_19.train()  # switch the model to `train` mode
        for num_iter, (batch_x, labels) in enumerate(train_dloader):
            #l2_loss = 0
            optimizer.zero_grad()
            output = vgg_19(batch_x)
            loss = criterion(output, labels)  # calculate the error
                        
            loss.backward()                 # back-propagation
            optimizer.step()                # update parameter using the gradient from the backpropagation

            # monitoring some metrics, note, l2_loss will not be monitored...
            # ...since it is just the regulizer for the parameters, not the actual metric of the AI model
            epoch_loss += loss.item()
            num_corrects += (output.argmax(dim=-1).eq(labels)).sum().item()

        train_loss.append(epoch_loss / (num_iter + 1))
        train_acc.append(num_corrects / num_train_samples)

        # validation. We don't learn anything here, so we don't need regulizer
        epoch_loss = 0
        num_corrects = 0
        vgg_19.eval()  # switch the model to `test` mode
        with torch.no_grad():  # since we don't need to do back propagation in test mode, we turn it off to save the memory
            for num_iter, (batch_x, labels) in enumerate(test_dloader):
                output = vgg_19(batch_x)
                loss = criterion(output, labels)
                epoch_loss += loss.item()
                num_corrects += (output.argmax(dim=-1).eq(labels)).sum().item()

        test_loss.append(epoch_loss / (num_iter + 1))
        test_acc.append(num_corrects / num_test_samples)

        # print metrics
        print(f"epoch: {epoch + 1}/{target_epochs} - loss: {train_loss[-1]:.4f} - accuracy: {train_acc[-1]:.4f}"
              f" - test_loss: {test_loss[-1]:.4f} - test_acc: {test_acc[-1]:.4f}")
    train_loss = np.array(train_loss)
    train_acc = np.array(train_acc)
    test_loss = np.array(test_loss)
    test_acc = np.array(test_acc)
    return train_loss, train_acc, test_loss, test_acc

def predict(path):
    # try:
            
    #     img = Image.open(operationBytes)
    #     img.save('tel.png')
    #     print(" ----------  here   ------------")
            
    # except Exception as e:
    #     print(e)

    vgg_19=models.vgg19_bn(pretrained=True)
    classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(25088, 1028)),
                          ('relu1', nn.ReLU()),
                          ('dropout1',nn.Dropout(0.5)),
                          ('fc2', nn.Linear(1028, 512)),
                          ('relu2', nn.ReLU()), 
                          ('dropout2',nn.Dropout(0.5)),
                          ('fc3', nn.Linear(512, 2)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    
    vgg_19.classifier = classifier
    vgg_19.load_state_dict(torch.load('./cnn.pt'))
    img = Image.open(path)
    img = img.resize((224, 224))
    convert_tensor = transforms.ToTensor()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                            [0.229, 0.224, 0.225])])
    print(transforms.ToTensor()(img))
    a = transform(img)
    print(a)
    vgg_19.eval()
    with torch.no_grad():
        output = vgg_19(a[None])
        prediction = output.max(1, keepdim=True)[1]
        print(output)
    classes = ['Cat', 'Dog']
    return classes[prediction[0,0].item()]

if __name__ == "__main__":
    data_transform = transforms.Compose([transforms.Resize((224,224)),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

    cat_files_train = [tf for tf in os.listdir(train_dir+'/cats')]
    dog_files_train = [tf for tf in os.listdir(train_dir+'/dogs')]

    cat_files_test = [tf for tf in os.listdir(test_dir+'/cats')]
    dog_files_test = [tf for tf in os.listdir(test_dir+'/dogs')]

    cats_train = catdogDataset(cat_files_train, train_dir+'/cats', transform = data_transform)
    dogs_train = catdogDataset(dog_files_train, train_dir+'/dogs', transform = data_transform)

    catdogs_train = ConcatDataset([cats_train, dogs_train])

    train_dloader = DataLoader(catdogs_train, batch_size = 32, shuffle=True, num_workers=0)


    cats_test = catdogDataset(cat_files_test, test_dir+'/cats', transform = data_transform)
    dogs_test = catdogDataset(dog_files_test, test_dir+'/dogs', transform = data_transform)

    catdogs_test = ConcatDataset([cats_test, dogs_test])

    test_dloader = DataLoader(catdogs_test, batch_size = 32, shuffle=True, num_workers=0)

    vgg_19=models.vgg19_bn(pretrained=True)

    for param in vgg_19.parameters():
        param.requires_grad = False


    classifier = nn.Sequential(OrderedDict([
                            ('fc1', nn.Linear(25088, 1028)),
                            ('relu1', nn.ReLU()),
                            ('dropout1',nn.Dropout(0.5)),
                            ('fc2', nn.Linear(1028, 512)),
                            ('relu2', nn.ReLU()), 
                            ('dropout2',nn.Dropout(0.5)),
                            ('fc3', nn.Linear(512, 2)),
                            ('output', nn.LogSoftmax(dim=1))
                            ]))
        
    vgg_19.classifier = classifier

    criterion = nn.NLLLoss()
    optimizer=torch.optim.Adam(vgg_19.parameters(),lr=0.01)
    train_loss, train_acc, test_loss, test_acc = train(5)
