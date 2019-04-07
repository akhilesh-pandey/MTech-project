# load necessary libraries

from model import AE, FC_model
from utils import get_dataloader, load_model
from fgsm import fgsm  # Fast gradient sign method
from deepfool import deepfool
import torch
import torch.nn as nn
import numpy as np

from universal import load_universal_perturbation

device = ('cuda' if torch.cuda.is_available() else 'cpu')


def adv_attack(model, defender, test_loader, criterion, attack_type = 'fgsm', eps=0.25):
    model.eval()
    correct, total = 0, 0
    # if universal attack , then load the universal perturbation
    if attack_type == 'universal':
        pert = load_universal_perturbation(model, test_loader)

    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        if attack_type == 'fgsm':
            images = images.view(-1, 28 * 28)
            images = fgsm(model, images, labels, epsilon=eps, loss_fn=criterion) # use fc_model to generate adv examples

        elif attack_type == 'deepfool':
            _, _, _, _, images = deepfool(images[0], model, num_classes=10, overshoot=0.2, max_iter=20)
            images = images.view(-1, 28 * 28)

        elif attack_type == 'universal':
            images = torch.from_numpy(np.clip(np.array(images) + pert, 0, 1)).view(-1,28*28).to(device)

        if defender:
            _, images = defender(images) # comment this line to see the effect of adv examples without defender
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    if attack_type == "fgsm":
        print('---FGSM---\n Eps: {:.2}  Adv_acc: {:.3%}'.format(eps,correct/total))
    elif attack_type == "deepfool":
        print('---DeepFool---\n Adv acc: {:.3f}'.format(correct / total))
    else:
        print('---Universal attack---\n Adv acc: {:.3f}'.format(correct / total))

    return correct/total

def main():

    # test(fc_model, test_loader)

    batch_size = 1 # plese note: For deep fool batch_size MUST be 1
    train_loader, test_loader = get_dataloader(batch_size)

    fc_model = FC_model()  # Base model, used for classification, also used for crafting adversarial samples
    defender = AE()  # Defender: input goes through this before going to the base model

    load_model(fc_model, './pretrained_models/fc_model.pth')
    load_model(defender, './pretrained_models/autoencoder_pretrained.pth')

    fc_model.to(device)
    defender.to(device)

    criterion = nn.CrossEntropyLoss()

    # craft adversarial examples for epsilon value in [0,1] at step size of 0.05
    '''
    acc_list= []
    for i in range(21):
        acc_list.append(adv_attack(fc_model, defender, test_loader, criterion, i*0.05))
    print(acc_list)
    '''
    # defender = None

    # FGSM attack
    adv_attack(fc_model, defender, test_loader, criterion, attack_type="fgsm")

    # deep fool attack
    adv_attack(fc_model, defender, test_loader, criterion, attack_type="deepfool")

    # universal attack
    adv_attack(fc_model, defender, test_loader, criterion, attack_type="universal")


if __name__ == '__main__':
    main()

