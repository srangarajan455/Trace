import cv2
import torch.nn as nn
import numpy as np
import torch

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, pad, bias=True, bn=True):
        super().__init__()
        if bn:
            self.block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, padding=pad, bias=bias),
                nn.ReLU(),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, padding=pad, bias=bias),
                nn.ReLU()
            )

    def forward(self, x):
        return self.block(x)

class ShuttleTrackerNet(nn.Module):
    """
    Deep network for shuttlecock detection (badminton)
    """
    def __init__(self, out_channels=256, bn=True):
        super().__init__()
        self.out_channels = out_channels

        # Encoder layers
        self.encoder = nn.Sequential(
            ConvBlock(9, 64, 3, 1, bn=bn),
            ConvBlock(64, 64, 3, 1, bn=bn),
            nn.MaxPool2d(2, 2),
            ConvBlock(64, 128, 3, 1, bn=bn),
            ConvBlock(128, 128, 3, 1, bn=bn),
            nn.MaxPool2d(2, 2),
            ConvBlock(128, 256, 3, 1, bn=bn),
            ConvBlock(256, 256, 3, 1, bn=bn),
            ConvBlock(256, 256, 3, 1, bn=bn),
            nn.MaxPool2d(2, 2),
            ConvBlock(256, 512, 3, 1, bn=bn),
            ConvBlock(512, 512, 3, 1, bn=bn),
            ConvBlock(512, 512, 3, 1, bn=bn)
        )

        # Decoder layers
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2),
            ConvBlock(512, 256, 3, 1, bn=bn),
            ConvBlock(256, 256, 3, 1, bn=bn),
            ConvBlock(256, 256, 3, 1, bn=bn),
            nn.Upsample(scale_factor=2),
            ConvBlock(256, 128, 3, 1, bn=bn),
            ConvBlock(128, 128, 3, 1, bn=bn),
            nn.Upsample(scale_factor=2),
            ConvBlock(128, 64, 3, 1, bn=bn),
            ConvBlock(64, 64, 3, 1, bn=bn),
            ConvBlock(64, self.out_channels, 3, 1, bn=bn)
        )

        self.softmax = nn.Softmax(dim=1)
        self._init_weights()

    def forward(self, x, testing=False):
        batch_size = x.size(0)
        features = self.encoder(x)
        score_map = self.decoder(features)
        output = score_map.reshape(batch_size, self.out_channels, -1)
        if testing:
            output = self.softmax(output)
        return output

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.uniform_(module.weight, -0.05, 0.05)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def inference(self, frames: torch.Tensor):
        self.eval()
        with torch.no_grad():
            if len(frames.shape) == 3:
                frames = frames.unsqueeze(0)
            if next(self.parameters()).is_cuda:
                frames = frames.cuda()
            output = self(frames, testing=True)
            output = output.argmax(dim=1).detach().cpu().numpy()
            if self.out_channels == 2:
                output *= 255
            x, y = self.get_center_shuttle(output)
        return x, y

    def get_center_shuttle(self, output):
        """
        Detect the center of the shuttlecock using connected component analysis
        :param output: output from model (single channel heatmap)
        :return: x, y coordinates of shuttlecock
        """
        output = output.reshape((360, 640)).astype(np.uint8)
        heatmap = cv2.resize(output, (640, 360))
        ret, binary_map = cv2.threshold(heatmap, 127, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(binary_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            largest = max(contours, key=cv2.contourArea)
            M = cv2.moments(largest)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                return cx, cy

        return None, None
"""🔧 1. Upgrade Backbone
Replace the VGG-style layers with MobileNetV3 or EfficientNet for speed + accuracy.

🧠 2. Add Attention
Use CBAM or SE blocks to help the model focus on the shuttlecock area.

🎯 3. Heatmap Output
Change final output to a single-channel heatmap instead of classification.

Train with MSE loss on a 2D Gaussian centered on shuttlecock.

🕒 4. Use Temporal Layers
Instead of stacking 3 frames as input, try ConvLSTM or 3D CNN to capture motion better.

⚡ 5. Post-Processing
Add a Kalman Filter to smooth detections and reject outliers.

Consider using Euclidean distance + velocity thresholding."""