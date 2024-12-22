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


# https://pytorch.org/tutorials/advanced/neural_style_tutorial.html
# https://github.com/leongatys/PytorchNeuralStyleTransfer/blob/master/NeuralStyleTransfer.ipynb


class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        # content on conv4_2
        # style on conv1_1, conv2_1, conv3_1, conv4_1, conv5_1
        self.select = [0, 5, 10, 19, 28, 21]
        self.vgg = models.vgg19(weights=True).features[:29]  # pretrained
        for param in self.vgg.parameters():
            param.requires_grad_(False)
        for i, layer in enumerate(self.vgg):
            if isinstance(layer, nn.MaxPool2d):
                self.vgg[i] = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        features = []
        for i, layer in enumerate(self.vgg):
            x = layer(x)
            if i in self.select:
                features.append(x)
        return features


def image_loader(
        image_path,
        transform=None,
        device=torch.device("cpu"),
):
    """
    read an image
    process it to tensor available for vgg
    """
    image = Image.open(image_path)
    if transform:
        image = transform(image).unsqueeze(0)
    return image.to(device, torch.float)


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
    c, hw = tensor.size()
    gram = torch.mm(tensor, tensor.t())
    return gram / hw


if __name__ == "__main__":
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")

    image_size = 512 if device == torch.device("cuda") else 256
    img_path = './img/'
    save_path = './results/'
    # content_name = 'Tuebingen_Neckarfront'
    # style_name = 'vangogh_starry_night'
    content_name = 'content_palace'
    style_name = 'style_vanGogh'
    content_path = f'{img_path}{content_name}.jpg'
    style_path = f'{img_path}{style_name}.jpg'

    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.485, 0.456, 0.406),
            (1, 1, 1),
        ),
        transforms.Lambda(lambda x: x[torch.LongTensor([2, 1, 0])]),
        transforms.Lambda(lambda x: x.mul(255)),
    ])

    post_process = transforms.Compose([
        transforms.Lambda(lambda x: x.mul(1/255)),
        transforms.Lambda(lambda x: x[torch.LongTensor([2, 1, 0])]),
        transforms.Normalize(
            (-0.485, -0.456, -0.406),
            (1, 1, 1),
        ),
        transforms.Lambda(lambda x: x.clamp(0, 1)),
        transforms.ToPILImage(),
    ])

    # load content and style image
    content = image_loader(
        content_path,
        transform=transform,
        device=device
    )
    style = image_loader(
        style_path,
        transform=transform,
        device=device
    )

    model = FeatureExtractor().to(device).eval()

    # get content and style features
    content_feature = get_features(content, model)[-1]
    style_features = get_features(style, model)[:-1]
    style_grams = [gram_matrix(feature) for feature in style_features]

    # # white noise image
    # target = torch.randn_like(content).requires_grad_(True).to(device)
    target = content.clone().requires_grad_(True)

    content_weight = 1
    style_weights = [1e3 / n**2 for n in [64, 128, 256, 512, 512]]
    optimizer = optim.LBFGS([target])
    steps = 100
    show_every = 10

    def closure():
        optimizer.zero_grad()
        target_features = get_features(target, model)
        target_content = target_features[-1]
        target_styles = target_features[:-1]
        content_loss = F.mse_loss(target_content, content_feature) * content_weight
        style_loss = 0
        for target_style, style_gram, style_weight in zip(
                target_styles,
                style_grams,
                style_weights,
        ):
            target_gram = gram_matrix(target_style)
            style_loss += F.mse_loss(target_gram, style_gram) * style_weight
        total_loss = content_loss + style_loss
        total_loss.backward()
        return total_loss

    for i in tqdm(range(1, steps + 1)):
        optimizer.step(closure)
        if i % show_every == 0:
            loss = closure()
            plt.imshow(post_process(target[0].cpu().detach()))
            plt.title(f"Iter {i}, Loss: {loss.item()}")
            plt.show()

    # save image
    result = post_process(target[0].cpu().detach())
    result = Image.fromarray((result * 255).astype(np.uint8))
    result.save(f'{save_path}{content_name}_{style_name}.jpg')
