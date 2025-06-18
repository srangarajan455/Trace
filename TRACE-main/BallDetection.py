import numpy as np
import cv2
import torch
from PIL import Image, ImageDraw
import ShuttlecockTrackerNet  # Make sure this is trained for badminton!

def combine_three_frames(frame1, frame2, frame3, width, height):
    """
    Combines three frames into one 9-channel tensor for model input.
    """
    img = cv2.resize(frame1, (width, height)).astype(np.float32)
    img1 = cv2.resize(frame2, (width, height)).astype(np.float32)
    img2 = cv2.resize(frame3, (width, height)).astype(np.float32)
    imgs = np.concatenate((img, img1, img2), axis=2)
    imgs = np.rollaxis(imgs, 2, 0)  # Channels first
    return np.array(imgs)

class ShuttlecockDetector:
    """
    Shuttlecock Detector using a CNN model (modified TrackNet) for badminton.
    Tracks shuttlecock positions across video frames.
    """
    def __init__(self, save_state, out_channels=2):
        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load the trained model
        self.detector = ShuttlecockTrackerNet(out_channels=out_channels)
        saved_state_dict = torch.load(save_state, map_location=self.device)
        self.detector.load_state_dict(saved_state_dict['model_state'])
        self.detector.eval().to(self.device)

        # Frame placeholders
        self.current_frame = None
        self.last_frame = None
        self.before_last_frame = None

        # Video properties
        self.video_width = None
        self.video_height = None

        # Input size to the model
        self.model_input_width = 720
        self.model_input_height = 480

        # Detection threshold (distance between frames)
        self.threshold_dist = 60

        # Tracking history
        self.xy_coordinates = np.array([[None, None]])
        self.bounces_indices = []

    def rescale_coordinates(self, x, y):
        if x is None or y is None:
            return None, None
        x = int(x * (self.video_width / self.model_input_width))
        y = int(y * (self.video_height / self.model_input_height))
        return x, y

    def get_smoothed_position(self, window=3):
        coords = self.xy_coordinates[-window:]
        valid = [c for c in coords if None not in c]
        if not valid:
            return None, None
        return np.mean(valid, axis=0).astype(int)

    def detect_shuttlecock(self, frame, show_debug=False):
        """
        Detect shuttlecock using 3 consecutive frames.
        :param frame: Current OpenCV frame
        :param show_debug: If True, shows detection on screen
        :return: (x, y) coordinates of detected shuttlecock (smoothed)
        """
        if self.video_width is None:
            self.video_width = frame.shape[1]
            self.video_height = frame.shape[0]

        # Shift frame history
        self.last_frame = self.before_last_frame or frame.copy()
        self.before_last_frame = self.current_frame or frame.copy()
        self.current_frame = frame.copy()

        # Prepare input for the model
        frames = combine_three_frames(self.current_frame, self.before_last_frame, self.last_frame,
                                      self.model_input_width, self.model_input_height)
        frames = (torch.from_numpy(frames) / 255.0).float().unsqueeze(0).to(self.device)

        # Predict position
        with torch.no_grad():
            x, y = self.detector.inference(frames)

        # Rescale to original resolution
        x, y = self.rescale_coordinates(x, y)

        # Optional outlier rejection
        if self.xy_coordinates[-1][0] is not None:
            if np.linalg.norm(np.array([x, y]) - self.xy_coordinates[-1]) > self.threshold_dist:
                x, y = None, None

        # Store and smooth
        self.xy_coordinates = np.append(self.xy_coordinates, np.array([[x, y]]), axis=0)
        x_smooth, y_smooth = self.get_smoothed_position()

        # Show debug view
        if show_debug and x_smooth is not None and y_smooth is not None:
            debug_frame = frame.copy()
            cv2.circle(debug_frame, (x_smooth, y_smooth), 10, (0, 0, 255), -1)
            cv2.imshow("Shuttlecock Detection", debug_frame)
            cv2.waitKey(1)

        return x_smooth, y_smooth