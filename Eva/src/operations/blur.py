import cv2


class Blur:
	def __init__(self, kernel_size=5):
		self.id = None
		self.kernel_size = kernel_size
		pass

	def apply(self, frames):
		output_frames = []
		for frame in frames:
			blur = cv2.blur(frame, (self.kernel_size, self.kernel_size))
			output_frames.append(blur)
		return output_frames
