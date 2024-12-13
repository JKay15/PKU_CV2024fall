import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse


class VGGNet(nn.Module):
    def __init__(self):
        super().__init__()
        # content on conv4_2
        # style on conv1_1, conv2_1, conv3_1, conv4_1, conv5_1
        self.select = ['0', '5', '10', '19', '28', '20']
        self.vgg = models.vgg19(weights=True).features  # pretrained VGG19 model
        for param in self.vgg.parameters():
            param.requires_grad_(False)

    def forward(self, x):
        features = []
        for name, layer in self.vgg._modules.items():
            x = layer(x)
            if name in self.select:
                features.append(x)
        return features


def preprocess(
        image_path,
        max_size=None,
        shape=None,
        transform=None,
        device=None
):
    """
    read an image
    process it to tensor available for vgg
    """
    image = Image.open(image_path)
    if max_size:
        scale = max_size / max(image.size)
        size = np.array(image.size) * scale
        image = image.resize(size.astype(int), Image.ANTIALIAS)
    if shape:
        image = image.resize(shape, Image.LANCZOS)
    if transform:
        image = transform(image).unsqueeze(0)
    return image.to(device)


def postprocess(tensor):
    """
    convert a tensor to an image
    """
    image = tensor.to("cpu").clone().detach()
    image = image.numpy().squeeze()
    image = image.transpose(1, 2, 0)
    image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
    image = image.clip(0, 1)
    return image


def get_features(image, model):
    """
    get feature vectors
    """
    layers = model(image)
    features = []
    for layer in layers:
        feature = layer.reshape(layer.shape[1], -1)
        features.append(feature)
    return features


def gram_matrix(tensor):
    gram = torch.mm(tensor, tensor.t())
    return gram


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image_size = 356
    img_path = './img/'
    save_path = './results/'
    content_path = f'{img_path}content_palace.jpg'
    style_path = f'{img_path}style_vanGogh.jpg'

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225)
        )
    ])

    # load content and style image
    content = preprocess(
        content_path,
        shape=[image_size, image_size],
        transform=transform,
        device=device
    )
    style = preprocess(
        style_path,
        shape=[image_size, image_size],
        transform=transform,
        device=device
    )

    model = VGGNet().to(device).eval()

    # get content and style features
    content_features = get_features(content, model)
    style_features = get_features(style, model)
    style_grams = [gram_matrix(feature[:-1]) for feature in style_features]

    # clone content image since content image will be optimized
    target = content.clone().requires_grad_(True).to(device)

    style_weights = [1e3 / n ** 2 for n in [64, 128, 256, 512, 512]]
    content_weight = 1  # alpha
    style_weight = 1e6  # beta
    optimizer = optim.Adam([target], lr=0.003)
    steps = 2000
    show_every = 400  # show result every 400 steps

    # iteration
    for ii in tqdm(
            range(1, steps + 1),
            desc="Transfering",
    ):
        target_features = get_features(target, model)
        content_loss = F.mse_loss(target_features[-1], content_features[-1])
        style_loss = 0
        for target_feature, style_gram, weight in zip(
                target_features,
                style_grams,
                style_weights
        ):
            target_gram = gram_matrix(target_feature[:-1])
            style_loss += F.mse_loss(target_gram, style_gram) * weight
        total_loss = content_weight * content_loss + style_weight * style_loss
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # plot
        if ii % show_every == 0:
            print(f"Total loss: {total_loss.item()}")
            plt.imshow(postprocess(target))
            plt.axis("off")
            plt.show()

    # save image
    plt.imshow(postprocess(target))
    plt.axis("off")
    content_name = content_path.split('/')[-1].split('.')[0]
    style_name = style_path.split('/')[-1].split('.')[0]
    plt.savefig(f'{save_path}{content_name}_{style_name}.png')
    plt.show()
