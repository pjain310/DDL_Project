import cv2

class Equalised_Histogram:
    def __init__(self):
        self.id = None
        pass

    def apply(self, frames):
        for frame in frames:
            equ = cv2.equalizeHist(frame.image)
            frame.image = equ
        return frames
