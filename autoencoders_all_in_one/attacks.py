"""
Aum Sri Sairam

"""

"""
Implements the Saliency Map Attack.
    The attack was introduced in [1]. During each iteration this method finds the pixel (or two pixels) that has the most influence on the result (most salient pixel) and add noise to the pixel.
    References:
    ----------
    .. [1] Nicolas Papernot, Patrick McDaniel, Somesh Jha, Matt Fredrikson,
           Z. Berkay Celik, Ananthram Swami, "The Limitations of Deep Learning
           in Adversarial Settings", https://arxiv.org/abs/1511.07528
    .. [2] Foolbox : https://github.com/bethgelab/foolbox/blob/master/foolbox/attacks/saliency.py#L11-L179
    .. [3] https://github.com/gongzhitaao/tensorflow-adversarial/blob/master/attacks/saliency_map.py
"""

import math
import torch

from datetime import datetime


device= ('cuda' if torch.cuda.is_available() else 'cpu')

def compute_jacobian(model, images):
    """
    :param model: a trained model that is used for creating adversarial examples
    :param input: tensor of size batch_size x classes x h x w
    :return: a Jacobian matrix of shape
    """
    assert images.requires_grad
    output = model(images)
    # input= _ravel(images)
    num_classes = output.size(1)

    b, c, h, w = images.size()

    """
    For general case just replace the nextline with jacobian = torch.zeros(b, num_classes, c,h,w)
    """
    # print("input size: ",images.size())
    # print("input.requires_grad:", images.requires_grad)
    jacobian = torch.zeros(b, num_classes, c, h, w)
    # print("jacobian: ",jacobian.size())
    model.zero_grad()
    for i in range(num_classes):
        jacobian[:, i] = torch.autograd.grad(output[:, i].sum(), images, retain_graph=True)[0].data


    return jacobian.view(b, num_classes, -1)


def jsma(model, inputs, target_class, eps=1.0, k=2, max_distortion= 1):

    adv_input = torch.zeros_like(inputs)
    start_time= datetime.now()

    """
    :param model: a trained model used to create adversarial examples
    :param inputs: set of images , size= [b x c x h x w]
    :param target_class: [b x 1]
    :param eps: Amount of noise to be added at each iteration
    :param k: number of pixels to perturb at a time (only k=1 or 2 are accepted)
    :param max_distortion: maximum distortion in each image
    :return: A tensor containing adversarial examples for each input
    """

    #compute jacobian
    #jacobian = compute_jacobian(model, input)
    b, c, h, w = inputs.size()
    num_features= c*h*w

    #create a list of numbers to keep track of pixels not modified yet
    search_space = torch.ones(num_features).byte()
    max_iter= math.floor(num_features * max_distortion)
    inputs= inputs.unsqueeze(dim= 1)
    for i, input in enumerate(inputs):
        #input = input.unsqueeze(dim=0)
        output= model(input)
        _, source_class = torch.max(output, dim=1)
        #print("original class: ", source_class)
        count= 0
        #print(source_class.item() == target_class)
        while (count <  max_iter) and ( source_class.item() != target_class) and search_space.sum() > 0:
            p1, p2 = compute_saliency_map(model, input, search_space, target_class, k)

            size= input.size()
            #print('input size:', size)
            input= input.view(1,-1)
            #print('p1,p2: ',p1,p2)
            input[0][p1] =  torch.clamp(input[0][p1] + eps, 0, 1)
            input[0][p2] = torch.clamp(input[0][p2] + eps, 0, 1)
            input= input.view(size)
            output = model(input)
            _, source_class = torch.max(output, dim=1)
            search_space[p1]= 0
            search_space[p2]= 0

            count += 1
    adv_input[i,:,:,:] = input

    # print('count:{} max_iter:{}, source:{}, target:{}, search_space: {}'.format(count, max_iter, source_class.item(),
    #                                                                                 target_class, search_space.sum()))
    # print(adv_input.size())
    '''
    if (source_class.item() == target_class):
        print('hurrah found!!!')
    else:
        print('OOPs hard to attack!!!')
    '''
    return adv_input



def compute_saliency_map(model, input, search_space, target, k):
    jacobian = compute_jacobian(model, input)
    all_sum= torch.sum(jacobian[0],0).squeeze()

    #compute gradient of target w.r.t input
    dt_dx= jacobian[0][target].squeeze()
    #gradient of non-targets w.r.t input

    do_dx= all_sum - dt_dx
    mask1= torch.ge(dt_dx, 0.0)
    mask2= torch.le(do_dx, 0.0)

    """
    a salient pixel must satisfy condition 1 condition 2 given below:
    1. it must increase the probability of target class if increased and
    2. it must decrease the probability of other classes.
    """

    mask = torch.mul(torch.mul(mask1, mask2), search_space)
    salient_map=  (dt_dx * torch.abs(do_dx)) * mask.float()
    max_val, max_idx = torch.topk(salient_map, k)
    return max_idx

#TODO untargeted attack :
def jsma_untargeted_all_classes(model, inputs, target_class= None, eps=1.0, k=2, max_distortion= 1):

    """
    :param model: a trained model used to create adversarial examples
    :param inputs: set of images , size= [b x c x h x w]
    :param target_class: [b x 1]
    :param eps: Amount of noise to be added at each iteration
    :param k: number of pixels to perturb at a time (only k=1 or 2 are accepted)
    :param max_distortion: maximum distortion in each image
    :return: A tensor containing adversarial examples for each input
    """
    start_time = datetime.now()
    total_distortion = 0  # need for computing average distortion
    successful_distortion = 0

    adv_input = torch.zeros_like(inputs)

    # compute jacobian
    # jacobian = compute_jacobian(model, input)
    b, c, h, w = inputs.size()
    num_features = c * h * w

    # create a list of numbers to keep track of pixels not modified yet
    search_space = torch.ones(num_features).byte()
    max_iter = math.floor(num_features * max_distortion)
    inputs = inputs.unsqueeze(dim=1)

    for i, input in enumerate(inputs):
        output = model(input)
        _, source_class = torch.max(output, dim=1)
        # print("original class: ", source_class)
        count = 0
        # print(source_class.item() == target_class)
        while (count < max_iter) and (source_class.item() != target_class) and search_space.sum() > 0:
            p1, p2 = compute_saliency_map(model, input, search_space, target_class, k)

            size = input.size()
            # print('input size:', size)
            input = input.view(1, -1)
            # print('p1,p2: ',p1,p2)
            input[0][p1] = torch.clamp(input[0][p1] + eps, 0, 1)
            input[0][p2] = torch.clamp(input[0][p2] + eps, 0, 1)
            input = input.view(size)
            output = model(input)
            _, source_class = torch.max(output, dim=1)
            search_space[p1] = 0
            search_space[p2] = 0

            count += 1

        if count == max_iter:
            total_distortion += count*2
        else:
            total_distortion += count * 2
            successful_distortion += count*2

        adv_input[i, :, :, :] = input

    return adv_input, total_distortion, successful_distortion, output


def fgsm(model, images, labels, loss_fn, epsilon= 0.1):
    images.requires_grad = True
    adv_images = torch.zeros_like(images)
    labels= labels.to(device)
    output= model(images)
    loss= loss_fn(output, labels)
    model.zero_grad()
    loss.backward()
    image_grad= images.grad.data
    sign_image_grad= image_grad.sign()

    adv_images += images + epsilon * sign_image_grad

    adv_images= torch.clamp(adv_images,0,1)
    adv_images= adv_images.detach()
    return adv_images