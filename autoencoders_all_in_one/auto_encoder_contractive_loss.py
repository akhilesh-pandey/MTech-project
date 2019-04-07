

import torch.utils.data
import torch.nn as nn
from model import AE
import torch.optim as optim
from utils import save_model, load_model
from utils import  get_customDataLoader



device = 'cuda' if torch.cuda.is_available() else 'cpu'

if device == 'cuda':
    print("Using GPU")

# mse_loss = nn.BCELoss(size_average=False)
mse_loss = nn.MSELoss()


def contractive_loss(model,x, clean_x, optim, lam=1e-3):
    x.requires_grad = True
    y, x_ = model(x)
    optim.zero_grad()

    # compute jacobain of hidden representation w.r.t input
    y.backward(torch.ones(y.size()).to(device), retain_graph=True)

    x.grad.requires_grad = True
    loss = mse_loss(x_, clean_x) + lam * torch.mean(pow(x.grad, 2))
    optim.zero_grad()
    loss.backward()
    optim.step()
    return loss


def train(model, epoch, train_loader, optimizer):
    model.train()
    train_loss= 0
    total= 0

    """
    data = data contains clean as well as perturbed images
    labels = label contains only clean images
    """
    for i, (data, label) in enumerate(train_loader):
        data= data.to(device).view(-1, 28*28)
        label = label.to(device).view(-1, 28*28)
        # hidden_representation, recons_x = model(data)
        loss = contractive_loss(model, data, label, optimizer)
        total += label.size(1)
        train_loss += loss.item()

        if (i+1) % 100 == 0:
            print('Train epoch: {} [{}/{} ({:.0f}%)]\t Loss: {:.6f}'.format(
                epoch, i*len(data), len(train_loader.dataset),
                100*i/len(train_loader), loss.item()/len(data)))

    print('\n====> Epoch: {} Average loss: {:.6f}\n'.format(
        epoch, train_loss/total
    ))

    # ae_model.samples_write(recons_x, epoch)


def trainer( model, train_loader, test_loader, optimizer, save_path='./autoencoder_pretrained.pth'):
    epochs = 101
    best_loss = 100
    for epoch in range(epochs):
        train(model, epoch, train_loader, optimizer)
        val_acc = test(model, test_loader)
        if best_loss > val_acc:
            best_loss = val_acc
        save_model(epoch, model, best_loss, optimizer, model_path=save_path)

    return

def test(model, test_loader):
    model.eval()
    test_loss = []

    with torch.no_grad():
        for i, (data, label) in enumerate(test_loader):
            data = data.to(device).view(-1, 28 * 28)
            label = label.to(device).view(-1, 28 * 28)
            _, recons_x = model(data)
            loss = mse_loss(recons_x, label)
            test_loss.append(loss.item())

    avg_loss= sum(test_loss)/ len(test_loss)

    print('===> Test Average loss: {:.7f}\n'.format(avg_loss))

    return avg_loss

def main():
    ae_model = AE()
    batch_size = 200
    train_loader, test_loader = get_customDataLoader('./data/data_for_autoencoder.pth', batch_size=batch_size)
    optimizer = optim.Adam(ae_model.parameters(), lr=0.001, betas=(0.9, 0.999))
    ae_model.to(device)
    trainer(ae_model, train_loader, test_loader, optimizer, save_path='./pretrained_models/autoencoder_pretrained_con_test.pth')

if __name__ == '__main__':
    main()


#load_model(ae_model, './autoencoder_pretrained_con_1.pth')
#test(ae_model, test_loader)