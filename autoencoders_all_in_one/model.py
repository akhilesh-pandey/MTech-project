import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os


# Base model FC:784->100->100->10
class FC_model(nn.Module):
    def __init__(self):
        super(FC_model, self).__init__()

        self.fc1= nn.Linear(784,100)
        self.fc2= nn.Linear(100, 100)
        self.fc3= nn.Linear(100, 10)
        self.relu= nn.ReLU()

    def forward(self, input):
        shape = input.size()
        input = input.view(shape[0], -1)
        output = self.relu(self.fc1(input))

        output = self.relu(self.fc2(output))
        output= self.fc3(output)
        return output


# Denoising autoencoder (DAE) : 784->256->128->64->128->256->784
# same as mentioned in the paper Combating adversarial attacks
# through denoising and dimensionality reduction
class AE(nn.Module):
    def __init__(self):
        super(AE, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(784, 256),nn.Linear(256, 128), nn.Linear(128, 64))
        self.decoder = nn.Sequential(nn.Linear(64, 128), nn.Linear(128, 256), nn.Linear(256, 784))
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        h1 = self.sigmoid(self.encoder(input))
        h2 = self.decoder(h1)
        sigmoid = self.sigmoid(h2)
        return h1,sigmoid

    def samples_write(self, x, epoch):
        _, samples = self.forward(x)
        # pdb.set_trace()
        samples = samples.data.cpu().numpy()[:16]
        fig = plt.figure(figsize=(4, 4))
        gs = gridspec.GridSpec(4, 4)
        gs.update(wspace=0.05, hspace=0.05)
        for i, sample in enumerate(samples):
            ax = plt.subplot(gs[i])
            plt.axis('off')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')
            plt.imshow(sample.reshape(28, 28), cmap='Greys_r')
        if not os.path.exists('out/'):
            os.makedirs('out/')
        plt.savefig('out/{}.png'.format(str(epoch).zfill(3)), bbox_inches='tight')
        # self.c += 1
        plt.close(fig)


"""
class AE_shallow(nn.Module):

    def __init__(self, input_size=784, hidden_size=80, output_size=784):
        super(AE_shallow, self).__init__()
        self.fc1 = nn.Linear(input_size,hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        shape = x.size()
        x = x.view(shape[0], -1)
        h1 = self.fc1(x)
        h2 = self.fc2(h1)
        recons_x = self.sigmoid(h2)
        return h1, recons_x


class FC_model_small(nn.Module):
    def __init__(self):
        super(FC_model_small, self).__init__()

        self.fc1= nn.Linear(80,100)
        self.fc2= nn.Linear(100, 100)
        self.fc3= nn.Linear(100, 10)
        self.relu= nn.ReLU()
        self.softmax= nn.Softmax()


    def forward(self, input):
        output = self.relu(self.fc1(input))

        output = self.relu(self.fc2(output))
        output= self.fc3(output)
        #output= self.softmax(output)
        return output
"""


