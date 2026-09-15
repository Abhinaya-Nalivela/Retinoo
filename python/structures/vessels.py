try:
    import torch
    import torch.nn as nn
except Exception as e:
    # Fallback when torch import fails (e.g., missing sym_float)
    torch = None
    nn = None
if nn is not None:
    class DoubleConv(nn.Module):
        """(convolution => [BN] => ReLU) * 2"""

        def __init__(self, in_channels, out_channels, mid_channels=None):
            super().__init__()
            if not mid_channels:
                mid_channels = out_channels
            self.double_conv = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

        def forward(self, x):
            return self.double_conv(x)


    class Down(nn.Module):
        """Downscaling with maxpool then double conv"""

        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.maxpool_conv = nn.Sequential(
                nn.MaxPool2d(2),
                DoubleConv(in_channels, out_channels)
            )

        def forward(self, x):
            return self.maxpool_conv(x)


    class Up(nn.Module):
        """Upscaling then double conv"""

        def __init__(self, in_channels, out_channels, bilinear=True):
            super().__init__()
            if bilinear:
                self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
                self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
            else:
                self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
                self.conv = DoubleConv(in_channels, out_channels)

        def forward(self, x1, x2):
            x1 = self.up(x1)
            diffY = x2.size()[2] - x1.size()[2]
            diffX = x2.size()[3] - x1.size()[3]

            import torch.nn.functional as F
            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2])
            x = torch.cat([x2, x1], dim=1)
            return self.conv(x)


    class OutConv(nn.Module):
        def __init__(self, in_channels, out_channels):
            super(OutConv, self).__init__()
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        def forward(self, x):
            return self.conv(x)


    class UNet(nn.Module):
        """
        Standard U-Net for Vessel Segmentation.
        """
        def __init__(self, n_channels=3, n_classes=1, bilinear=False):
            super(UNet, self).__init__()
            self.n_channels = n_channels
            self.n_classes = n_classes
            self.bilinear = bilinear

            self.inc = DoubleConv(n_channels, 64)
            self.down1 = Down(64, 128)
            self.down2 = Down(128, 256)
            self.down3 = Down(256, 512)
            factor = 2 if bilinear else 1
            self.down4 = Down(512, 1024 // factor)
            self.up1 = Up(1024, 512 // factor, bilinear)
            self.up2 = Up(512, 256 // factor, bilinear)
            self.up3 = Up(256, 128 // factor, bilinear)
            self.up4 = Up(128, 64, bilinear)
            self.outc = OutConv(64, n_classes)

        def forward(self, x):
            x1 = self.inc(x)
            x2 = self.down1(x1)
            x3 = self.down2(x2)
            x4 = self.down3(x3)
            x5 = self.down4(x4)
            x = self.up1(x5, x4)
            x = self.up2(x, x3)
            x = self.up3(x, x2)
            x = self.up4(x, x1)
            logits = self.outc(x)
            return logits

        def predict_mask(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
            logits = self.forward(x)
            probs = torch.sigmoid(logits)
            return (probs > threshold).float()
else:
    # torch unavailable – skip NN definitions; still expose segment_vessels_classical
    pass


def segment_vessels_classical(image_rgb, mask=None):
    """
    Robust classical vessel segmentation using Green-channel CLAHE,
    morphological Top-Hat, and adaptive thresholding.
    Provides a fast, zero-dependency medical image processing baseline.
    """
    import cv2
    import numpy as np

    h, w = image_rgb.shape[:2]
    green = image_rgb[:, :, 1]

    # Retinal mask if not supplied
    if mask is None:
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        _, mask = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
        mask = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
    else:
        # Ensure mask is a single‑channel uint8 image matching the input size
        if mask.ndim == 3:
            # Convert possible RGB/RGBA mask to grayscale
            mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
        mask = cv2.resize(mask, (image_rgb.shape[1], image_rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
        mask = mask.astype(np.uint8)
    # Ensure mask has same size and type as vessels later
    # (mask may be provided externally; we'll align it just before use)

    # Invert green channel so vessels are bright
    inv_green = cv2.bitwise_not(green)

    # CLAHE enhancement
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(inv_green)

    # Morphological Top-Hat with multi-scale linear elements or structuring element
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    tophat = cv2.morphologyEx(enhanced, cv2.MORPH_TOPHAT, kernel)

    # Adaptive thresholding
    vessels = cv2.adaptiveThreshold(
        tophat, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 19, -4
    )
    # Align mask size and dtype with vessels before combining
    if mask.shape != vessels.shape:
        mask_resized = cv2.resize(mask, (vessels.shape[1], vessels.shape[0]), interpolation=cv2.INTER_NEAREST)
    else:
        mask_resized = mask
    mask_resized = mask_resized.astype(vessels.dtype)
    vessels = cv2.bitwise_and(vessels, vessels, mask=mask_resized)

    # Remove small isolated noise specks (< 15 pixels)
    contours, _ = cv2.findContours(vessels, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cleaned_mask = np.zeros((h, w), dtype=np.uint8)
    for cnt in contours:
        if cv2.contourArea(cnt) >= 15:
            cv2.drawContours(cleaned_mask, [cnt], -1, 255, -1)

    return cleaned_mask


def visualize_vessels(image_rgb, vessel_mask, color=(0, 255, 0), alpha=0.4):
    """
    Overlays segmented vessels on the original fundus image.
    """
    import cv2
    import numpy as np

    overlay = image_rgb.copy()
    overlay[vessel_mask > 0] = color
    blended = cv2.addWeighted(image_rgb, 1 - alpha, overlay, alpha, 0)
    return blended

