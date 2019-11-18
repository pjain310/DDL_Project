import cv2

class Laplacian:
	def __init__(self):
		self.id = None
		pass

	def apply(self, frames):
		output_frames = []
		for frame in frames:
			laplacian = cv2.Laplacian(frame,cv2.CV_64F)
			output_frames.append(laplacian)
		return output_frames
