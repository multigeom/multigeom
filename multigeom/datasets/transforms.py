"""
The image transform of the pipeline: images 64 x 64, single-channel, in [-1, 1].
"""

from torchvision import transforms

# Inception/ResNet input convention used in the paper:
#   [-1, 1] grayscale, replicated to 3 channels at the encoder boundary.
DEEP_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((64, 64)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),  # → [1, 64, 64] in [0, 1]
        transforms.Normalize(mean=[0.5], std=[0.5]),  # → [1, 64, 64] in [-1, 1]
    ]
)

# Canonical pipeline input, shared by every encoder and by the diffusion
# model: [1, 64, 64] grayscale in [-1, 1]. Encoders whose backbone needs
# another range (ResNet-50 with ImageNet statistics, OnOff in [0, 1]) convert
# internally; see multigeom/encoders. Never feed [0, 1] images to an encoder.
CANONICAL_TRANSFORM = DEEP_TRANSFORM
