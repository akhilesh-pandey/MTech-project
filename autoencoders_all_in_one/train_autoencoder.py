# This program is for training the autoencoder
#
#

import torch.utils.data
import torch.nn as nn
from model import AE
import torch.optim as optim
from utils import save_model
from utils import  get_customDataLoader


device = 'cuda' if torch.cuda.is_available() else 'cpu'

if device == 'cuda':
    print("Using GPU")



def train(model, epoch, optimizer, criterion, train_loader):
    model.train()
    train_loss= 0
    total= 0

    for i, (data, label) in enumerate(train_loader):
        data= data.to(device).view(-1, 28*28)
        label = label.to(device).view(-1, 28*28)
        optimizer.zero_grad()
        hidden_representation, recons_x = model(data)
        loss= criterion(recons_x, label)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        total += label.size(1)

        if (i+1) % 100 == 0:
            print('Train epoch: {} [{}/{} ({:.0f}%)]\t Loss: {:.6f}'.format(
                epoch, i*len(data), len(train_loader.dataset),
                100*i/len(train_loader), loss.item()/len(data)))

    print('\n====> Epoch: {} Average loss: {:.5f}\n'.format(
        epoch, train_loss/total
    ))

    model.samples_write(recons_x, epoch)


def trainer( model, train_loader, test_loader, optimizer, criterion, save_path='./pretrained_models/autoencoder_pretrained.pth'):
    epochs = 100
    best_loss = 100
    for epoch in range(epochs):
        train(model, epoch, optimizer, criterion, train_loader)
        val_acc = test(model, test_loader)
        if best_loss > val_acc:
            best_loss = val_acc
            save_model(epoch, model, best_loss, optimizer, model_path=save_path)

    return


def test(model, test_loader):
    model.eval()
    test_loss = 0
    total = 0

    with torch.no_grad():
        for i, (data, label) in enumerate(test_loader):
            data = data.to(device).view(-1, 28 * 28)
            label = label.to(device).view(-1, 28 * 28)
            _, recons_x = model(data)
            loss = criterion(recons_x, label)
            test_loss += loss.item()
            total += label.size(0)

    avg_loss = test_loss/total

    print('===> Test Average loss: {:.7f}\n'.format(avg_loss))

    return avg_loss


ae_model= AE()
ae_model.to(device)
optimizer= optim.Adam(ae_model.parameters(), lr = 0.001, betas=(0.9, 0.999))
criterion= nn.MSELoss()
batch_size=256


train_loader, test_loader = get_customDataLoader('./data_for_ae/data_for_autoencoder.pth', batch_size= batch_size)
trainer( ae_model, train_loader, test_loader, optimizer, criterion, save_path='./pretrained_models/autoencoder_pretrained_1.pth')
#load_model(ae_model, './autoencoder_pretrained.pth')
#test(ae_model, test_loader)
