# load necessary libraries

from model import FC_model
from utils import save_model, get_dataloader
import torch
import torch.nn as nn
import torch.optim as optim

device = ('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 200

fc_model = FC_model()   # Base model, used for classification, also used for crafting adversarial samples
# defender = AE()         # Defender: input goes through this before going to the base model

fc_model.to(device)
# defender.to(device)

criterion= nn.CrossEntropyLoss()
optimizer= optim.Adam(fc_model.parameters(), lr=1e-3)


def train_base_model(model, train_loader, test_loader,
                     save_path='./pretrained_models/fc_model.pth'):
    epochs = 200
    best_acc = 0
    for epoch in range(epochs):
        train(model, epoch, train_loader)
        val_acc = test(model, test_loader)
        if best_acc < val_acc:
            best_acc = val_acc
            save_model(epoch, model, best_acc, optimizer, model_path=save_path)


def train(model, epoch, train_loader):
    model.train()
    loss_list, batch_list = [], []
    for i, (images, labels) in enumerate(train_loader):
        images = images.view(-1, 28*28)
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        loss_list.append(loss.item())
        batch_list.append(i + 1)

        if i + 1 % 50 == 0:
            print('Train epoch: {}, Batch: {}, loss: {}'.format(epoch, i, loss.item()))
        loss.backward()
        optimizer.step()


def test(model, test_loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            images= images.view(-1, 28*28)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        print('Accuracy on testset = {:.2%}'.format(correct/total))
        return correct/total


def main():
    train_loader, test_loader = get_dataloader(256)

    # Train the base model and save the parameters
    train_base_model(fc_model, train_loader, test_loader, save_path='./pretrained_models/fc_model.pth')


if __name__ == '__main__':
    main()
