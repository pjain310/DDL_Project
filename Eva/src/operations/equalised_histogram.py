import cv2

class Equalised_Histogram:
    def __init__(self):
        self.id = None
        pass

    def apply(self, frames):
        for frame in frames:
            equ = cv2.equalizeHist(cv2.cvtColor(frame.image, cv2.COLOR_BGR2GRAY))
            frame.image = equ
        return frames
