import cv2
import numpy as np

class MotionDetectorEngine:
    def __init__(self):
        self.temp_frame = None
        self.is_moving = False
        self.threshold = 25

    def process_frame(self, frame_colour, mode='overlay'):
        
        # Convert to greyscale 
        gray = cv2.cvtColor(frame_colour, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (11, 11), 0)
        mask_colour = np.zeros_like(frame_colour)

        # Check for change
        if self.temp_frame is not None:
            diff = cv2.absdiff(self.temp_frame, blurred)
            _, thresh = cv2.threshold(diff, self.threshold, 255, cv2.THRESH_BINARY)
            mask_colour = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
            mask_colour[:, :, 0] = 0
            mask_colour[:, :, 1] = 0
            self.is_moving = bool(np.any(thresh))
        
        # Update last frame
        self.temp_frame = blurred

        if mode == 'mask':
            output = mask_colour
        else:
            output = cv2.addWeighted(
                cv2.convertScaleAbs(frame_colour, alpha=1.0, beta=-50),
                1.0, mask_colour, 0.5, 0
            )
        return output

    def set_threshold(self, value):
        self.threshold = int(value)

    def get_motion_status(self):
        return self.is_moving