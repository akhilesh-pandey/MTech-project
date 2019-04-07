#utility modules

# import torch
# from dataset import ConcatDataset, CustomDataset
import torch
import torch.utils.data as data_utils
# from torch.utils.data import Dataset
from torchvision.datasets.mnist import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from pathlib import Path
# import numpy as np


def save_model(epoch, model, best_acc, optimizer, model_path= './best_model.pth'):
    state= {'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(), 'best_acc': best_acc}
    torch.save(state, model_path)
    print("===> model save at: \n", model_path)


def load_model(model, model_path= './best_model.pth'):
    print('Loading pretrained model {} =======>', model_path)
    checkpoint= torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer'])


def get_dataloader(batch_size, root= 'mnist_data'):
    root= Path(root).expanduser()
    if not root.exists():
        root.mkdir()
    root= str(root)
    transform = transforms.ToTensor()
    train_data= MNIST(root, train= True, download= True, transform= transform)
    test_data= MNIST(root, train= False, transform= transform)

    train_loader= DataLoader(train_data, batch_size= batch_size, shuffle= True)
    test_loader= DataLoader(test_data, batch_size= batch_size, shuffle= True)

    return train_loader, test_loader


def get_customDataLoader(path= './dataf/data_for_autoencoder.pth', batch_size=256):

    """
    To load the GPU trained weights to CPU you will have to map them. This can be done by simply loading the model using:

    model=torch.load(trained_model.pt,map_location={'cuda:0': 'cpu'})
    """
    dataset = torch.load(path)

    train_data= dataset['total_train_data']
    train_label= dataset['total_train_label']

    test_data= dataset['total_test_data']
    test_label= dataset['total_test_label']
    """

    customDataset_train = CustomDataset(train_data, train_label)
    customDataset_test = CustomDataset(test_data, test_label)

    customDataLoader_train = DataLoader(customDataset_train, batch_size= batch_size)
    customDataLoader_test = DataLoader(customDataset_test, batch_size= batch_size)

    """
    print("Train data: {} Test data: {}".format(train_data.size(), test_data.size()))

    train = data_utils.TensorDataset(train_data, train_label)
    train_loader = data_utils.DataLoader(train, batch_size=batch_size, shuffle=True)

    test= data_utils.TensorDataset(test_data, test_label)
    test_loader= data_utils.DataLoader(test, batch_size= batch_size, shuffle= False)

    return train_loader, test_loader

