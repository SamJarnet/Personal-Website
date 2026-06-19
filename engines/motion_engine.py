import cv2
import numpy as np


class MotionDetectorEngine:
    def __init__(self, camera_index=0):
        self.camera_index = camera_index
        self.temp_frame = None
        self.cap = None
        self.is_moving = False
        self.threshold = 25

    def _get_capture(self): # reuse the capture opject if it is open
        if self.cap is not None and self.cap.isOpened():
            return self.cap
        self.cap = cv2.VideoCapture(self.camera_index)
        return self.cap
    

    def release_camera(self):
        if self.cap is not None:
            if self.cap.isOpened():
                self.cap.release()
            self.cap = None
            self.temp_frame = None
            print(f"Camera {self.camera_index} successfully released.")
            return True
        return False

    def show_camera(self, mode='overlay'):
        cap = self._get_capture()

        if not cap.isOpened():
            print(f"Cannot open camera {self.camera_index}")

            # show blank screen instead of failing dangerously 
            blank = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(blank, "Camera unavailable", (40, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            ret, buffer = cv2.imencode('.jpg', blank)
            if ret:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            return

        try:
            while True:
                ret, frame_colour = cap.read()
                if not ret:
                    print("Exit")
                    break

                # Convert to greyscale
                gray = cv2.cvtColor(frame_colour, cv2.COLOR_BGR2GRAY)
                blurred_frame = cv2.GaussianBlur(gray, (11, 11), 0)
                mask_colour = np.zeros_like(frame_colour)

                # Check for change
                if self.temp_frame is not None:
                    if self.temp_frame.shape == blurred_frame.shape:
                        diff = cv2.absdiff(self.temp_frame, blurred_frame)
                        _, threshold = cv2.threshold(diff, self.threshold, 255, cv2.THRESH_BINARY)
                        mask_colour = cv2.cvtColor(threshold, cv2.COLOR_GRAY2BGR)  # convert to 3 channels or it doesnt work
                        mask_colour[:, :, 0] = 0
                        mask_colour[:, :, 1] = 0
                        self.is_moving = bool(np.any(threshold))

                # Update last frame
                self.temp_frame = blurred_frame

                if mode == 'mask':
                    output_frame = mask_colour
                else:
                    output_frame = cv2.addWeighted(
                        cv2.convertScaleAbs(frame_colour, alpha=1.0, beta=-50),
                        1.0, mask_colour, 0.5, 0
                    )

                # output .jpg instead of displaying with cv2 so the website can display the image
                ret, buffer = cv2.imencode('.jpg', output_frame)
                if not ret:
                    continue
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        except GeneratorExit:
            print("Stream client disconnected.")


if __name__ == "__main__":
    sim = MotionDetectorEngine()
    for _ in sim.show_camera():
        pass