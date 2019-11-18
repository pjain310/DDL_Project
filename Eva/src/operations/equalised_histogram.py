import cv2
from grayscale import Grayscale
import numpy as np
class Equalised_Histogram:
	def __init__(self):
		self.id = None
		pass

	def apply(self, frames):
		g = Grayscale()
		frames = g.apply(frames)
		output_frames = []
		for frame in frames:
			frame = frame.astype(np.uint8)
			equ = cv2.equalizeHist(frame)
			output_frames.append(equ)
		return output_frames
