import cv2
import numpy as np


class Grayscale:

	def __init__(self):
		self.id = None

	def apply(self, frames):
		output_frames = []
		for frame in frames:
			gray2d = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
			gray_im = np.repeat(gray2d[:, :, None], 3, axis=2)
			output_frames.append(gray_im)
		return output_frames
