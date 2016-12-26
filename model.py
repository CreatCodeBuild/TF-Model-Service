import tensorflow as tf
import numpy as np
import cv2 as cv

class DefaultModelServer():
	def __init__(self):
		self.sess = tf.Session()
		self.new_saver = tf.train.import_meta_graph('E:\Git Repo\TF-Model-Service\model\default.ckpt.meta')
		self.new_saver.restore(self.sess, 'E:\Git Repo\TF-Model-Service\model\default.ckpt')
		self.test_prediction = tf.get_default_graph().get_tensor_by_name('test/single_prediction:0')
		self.single_input = tf.get_default_graph().get_tensor_by_name('test/single_input:0')
		print('DefaultModelServer is up')

	def compute(self, inputData):
		'''
		:param inputData: a ndarray with shape (1, 32, 32, 1), representing 1 grayscaled image
		:return: the softmaxed result
		'''
		return self.sess.run(self.test_prediction, feed_dict={self.single_input: inputData})

	def transform_data(self, image):
		# :param image: ndarray
		# the shape of image should be (32, 32, 1)
		if image.shape[2] != 1:
			image = tf.image.rgb_to_grayscale(image)
		image = tf.image.resize_images(image, (32, 32))
		return [image.eval(session=self.sess)] # todo: should to grayscale and resize outside and use OpenCV maybe?

	def serve(self, image):
		return self.compute(self.transform_data(image))


# Test
if __name__ == '__main__':
	server = DefaultModelServer()
	for i in range(10):
		im = cv.imread(str(i) + ".png", cv.IMREAD_COLOR)
		result = server.serve(im)
		print(str(result))

