import numpy as np
import cv2
import torch
from PIL import Image, ImageDraw

from ShuttlecockTrackNet import ShuttlecockTrackerNet  # Make sure to retrain or fine-tune for badminton

def combine_three_frames(frame1, frame2, frame3, width, height):
    # Resize and convert each frame to float32
    img = cv2.resize(frame1, (width, height)).astype(np.float32)
    img1 = cv2.resize(frame2, (width, height)).astype(np.float32)
    img2 = cv2.resize(frame3, (width, height)).astype(np.float32)

    # Concatenate images along depth (channel) axis: 3 x (H x W x 3) -> (H x W x 9)
    imgs = np.concatenate((img, img1, img2), axis=2)

    # Rearrange axes for PyTorch (channels first)
    imgs = np.rollaxis(imgs, 2, 0)
    return np.array(imgs)

class ShuttlecockDetector:
    """
    Shuttlecock Detector using a lightweight CNN model for amateur badminton matches.
    """
    def __init__(self, save_state, out_channels=2):
        self.device = torch.device("cpu")

        self.detector = ShuttlecockTrackerNet(out_channels=out_channels)
        saved_state_dict = torch.load(save_state, map_location=self.device)
        self.detector.load_state_dict(saved_state_dict['model_state'])
        self.detector.eval().to(self.device)

        self.current_frame = None
        self.last_frame = None
        self.before_last_frame = None

        self.video_width = None
        self.video_height = None

        # Resize model input size for better precision with small shuttlecock
        self.model_input_width = 720
        self.model_input_height = 480

        # Reduced threshold since shuttle moves faster and covers more ground
        self.threshold_dist = 60

        self.xy_coordinates = np.array([[None, None]])
        self.bounces_indices = []

    def detect_shuttlecock(self, frame):
        """
        Detects shuttlecock position after receiving 3 consecutive frames.
        :param frame: current frame
        """
        if self.video_width is None:
            self.video_width = frame.shape[1]
            self.video_height = frame.shape[0]

        self.last_frame = self.before_last_frame
        self.before_last_frame = self.current_frame
        self.current_frame = frame.copy()

        if self.last_frame is not None:
            # Prepare input tensor
            frames = combine_three_frames(self.current_frame, self.before_last_frame, self.last_frame,
                                          self.model_input_width, self.model_input_height)
            frames = (torch.from_numpy(frames) / 255).to(self.device)

            # Model prediction
            x, y = self.detector.inference(frames)
            if x is not None:
                # Rescale coordinates
                x = int(x * (self.video_width / self.model_input_width))
                y = int(y * (self.video_height / self.model_input_height))

                # Outlier rejection based on speed
                if self.xy_coordinates[-1][0] is not None:
                    if np.linalg.norm(np.array([x, y]) - self.xy_coordinates[-1]) > self.threshold_dist:
                        x, y = None, None

            self.xy_coordinates = np.append(self.xy_coordinates, np.array([[x, y]]), axis=0)
            """Model retraining: The ShuttlecockTrackerNet must be trained on badminton footage, ideally with labeled shuttlecock positions.

            Higher frame rate videos: Shuttlecock movement is much faster than tennis. Use at least 60 fps videos.

            Smarter bounce detection: In badminton, shuttle bounces are rare (e.g., smashes or net shots that graze), so bounce detection might need to be replaced with landing detection (on court or not).

            Court boundary processing: Add morphological processing to check shuttlecock relative to court lines."""