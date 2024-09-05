import torch
import segmentation_models_pytorch as smp

from unet.train import loss_result


def demo01():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = smp.PSPNet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
        activation="sigmoid"
    )
    model = model.to(device)

    print(model)


demo01()