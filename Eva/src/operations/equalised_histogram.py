import cv2

class Equalised_Histogram:
	def __init__(self):
		self.id = None
		pass

	def apply(self, frames):
		frames = Grayscale.apply()
		output_frames = []
		for frame in frames:
			equ = cv2.equalizeHist(frame)
			output_frames.append(equ)
		return output_frames
