
# We modify the original  code available at :
from deepfool import deepfool
import os
import numpy as np
import torch
import torch.backends.cudnn as cudnn

def project_lp(v, xi, p):

    if p==2:
        pass
    elif p == np.inf:
        v=np.sign(v)*np.minimum(abs(v),xi)
    else:
        raise ValueError("Values of a different from 2 and Inf are currently not surpported...")

    return v


def generate(test_loader, net, delta=0.2, max_iter_uni=np.inf, xi=10, p=np.inf, num_classes=10,
                 overshoot=0.2, max_iter_df=20):
    '''

    :param path:
    :param dataset:
    :param testset:
    :param net:
    :param delta:
    :param max_iter_uni:
    :param p:
    :param num_class:
    :param overshoot:
    :param max_iter_df:
    :return:
    '''
    net.eval()
    if torch.cuda.is_available():
        device = 'cuda'
        net.cuda()
        cudnn.benchmark = True
    else:
        device = 'cpu'

    # num_img_trn = 60000
    num_img_test = 10000

    order = np.arange(num_img_test)

    v = torch.zeros([1,1,28,28]) # size of mnist images are 28, 28, 1
    fooling_rate = 0.0
    iter = 0

    # start an epoch
    while fooling_rate < 1-delta and iter < max_iter_uni:
        print("Starting pass number ", iter)

        for k, (cur_img1, label) in enumerate(test_loader):
            cur_img1 = cur_img1.to(device)
            r2 = int(net(cur_img1).max(1)[1])
            torch.cuda.empty_cache()
            per_img = cur_img1 + v # v.astype(np.uint8)
            per_img1 = per_img.to(device)
            r1 = int(net(per_img1).max(1)[1])
            torch.cuda.empty_cache()

            if r1 == r2:
                print(">> k =", np.where(k==order)[0][0], ', pass #', iter, end='      ')
                dr, iter_k, label, k_i, pert_image = deepfool(per_img1[0], net, num_classes=num_classes, overshoot=overshoot, max_iter=max_iter_df)
                if iter_k < max_iter_df-1:

                    v += torch.from_numpy(dr) # we need only one change for gray scale
                    v = project_lp(v, xi, p)

        iter = iter + 1

    return v


def load_universal_perturbation(net, test_loader):
    print('>> Loading perturbation...')
    file_perturbation = 'data/universal.npy'
    if os.path.isfile(file_perturbation) == 0:
        print('   >> No perturbation found, computing...')
        v = generate(test_loader, net, max_iter_uni=1000, delta=0.8, p=np.inf,
                     num_classes=10, overshoot=0.2, max_iter_df=10)
        # Saving the universal perturbation
        np.save('./data/universal.npy', v)
    else:
        print('   >> Found a pre-computed universal perturbation! Retrieving it from', file_perturbation)
        v = np.load(file_perturbation)[0]
    return v;

