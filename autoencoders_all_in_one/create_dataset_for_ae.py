from attacks import fgsm
from utils import load_model, get_dataloader

import torch
import torch.nn as nn
from model import FC_model


device = ('cuda' if torch.cuda.is_available() else 'cpu')

def createDatasetForAE(model, train_loader, test_loader, criterion, eps=0.25):
    adv_train_data_list = []
    clean_train_data_list = []
    train_label_list = []

    for i, (data, label) in enumerate(train_loader):
        clean_train_data_list.append(data)
        size = data.size()
        adv_train_data_list.append(
            fgsm(model, data.view(-1, size[-1] * size[-2]), label, epsilon=eps, loss_fn=criterion).view(size))
        train_label_list.append(data)  # Label for autoencoders are clean images
        # train_label_list.append(label)

    clean_test_data_list = []
    adv_test_data_list = []
    test_label_list = []

    for i, (data, label) in enumerate(test_loader):
        clean_test_data_list.append(data)
        size = data.size()
        adv_test_data_list.append(
            fgsm(model, data.view(-1, size[-1] * size[-2]), label, epsilon=eps, loss_fn=criterion).view(size))
        test_label_list.append(data)  # Label for autoencoders are clean images
        # test_label_list.append(label)

    clean_train_data_tensor = torch.cat(clean_train_data_list, 0)
    adv_train_data_tensor = torch.cat(adv_train_data_list, 0)
    train_label_tensor = torch.cat(train_label_list, 0)

    clean_test_data_tensor = torch.cat(clean_test_data_list, 0)
    adv_test_data_tensor = torch.cat(adv_test_data_list, 0)
    test_label_tensor = torch.cat(test_label_list, 0)

    total_train_data = torch.cat([clean_train_data_tensor, adv_train_data_tensor],
                                 0)  # 1,20,000 images (60000 clean + 60000 adversarial images)
    total_train_label = torch.cat([train_label_tensor, train_label_tensor], 0)  # 1,20,000 clean images are labels

    total_test_data = torch.cat([clean_test_data_tensor, adv_test_data_tensor], 0)
    total_test_label = torch.cat([test_label_tensor, test_label_tensor], 0)

    complete_data = {'total_train_data': total_train_data, 'total_train_label': total_train_label,
                     'total_test_data': total_test_data, 'total_test_label': total_test_label}

    torch.save(complete_data, './data/data_for_autoencoder.pth')
    print('data saved')
# torch.save(complete_data, 'adversarial_data.pth')


def main():
    batch_size = 200
    train_loader, test_loader = get_dataloader(batch_size)

    model = FC_model()
    load_model(model=model, model_path='./pretrained_models/fc_model.pth')
    criterion = nn.CrossEntropyLoss()
    model.to(device)

    createDatasetForAE(model, train_loader, test_loader, criterion)


if __name__ == '__main__':
    main()
