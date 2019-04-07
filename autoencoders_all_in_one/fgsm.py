import torch
device= ('cuda' if torch.cuda.is_available() else 'cpu')


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