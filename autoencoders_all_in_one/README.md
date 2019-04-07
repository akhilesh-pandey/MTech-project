## This repository contains the PyTorch implementation of Denoising Autoencoder (DAE) as a defense against adversarial attacks.

### We consider three major attacks for our work:
1. Fast Gradient Sign Method (FGSM)
2. DeepFool
3. Universal Adversarial Perturbation


##### Follow the following steps:

1. Train the base model ( save as './pretrained_modles/FC_model.pth' ) [train_base_model.py]
2. Load the base model from './pretrained_modles/FC_model.pth' and generate adversarial samples and save the samples at './data_for_ae/adversarial_data.pth' [create_dataset_for_ae.py]
3. Using the adversarial samples crafted in step 2, train the autoencoder and save the weights at './pretrained_models/autoencoder_pretrained.pth' [train_autoencoder.py]
4. Load the pretrained base model and  pretrained autoencoder for testing the accuracy of the base model in the presence of DAE as defender [main.py]