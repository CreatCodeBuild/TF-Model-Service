import tensorflow as tf
import numpy as np

class DefaultModelServer():
	def __init__(self):
		self.sess = tf.Session()

	def compute(self, inputData):
		'''
		:param inputData: a np array with shape (1, 32, 32, 1), representing 1 grayscaled image
		:return: the softmaxed result
		'''
		new_saver = tf.train.import_meta_graph('model/default.ckpt.meta')
		new_saver.restore(self.sess, 'model/default.ckpt')
		test_prediction = tf.get_default_graph().get_tensor_by_name('test/single_prediction:0')
		single_input = tf.get_default_graph().get_tensor_by_name('test/single_input:0')
		return self.sess.run(test_prediction, feed_dict={single_input: inputData})

	def transform_data(self, image):
		# the shape of image should be (32, 32, 1)
		if image.shape[2] != 1:
			image = tf.image.rgb_to_grayscale(image)
		image = tf.image.resize_images(image, (32, 32))
		return [image.eval(session=self.sess)]

	def serve(self, image):
		return self.compute(self.transform_data(image))

server = DefaultModelServer()

from PIL import Image
im = Image.open("1.png")



image = np.array(im.getdata(), np.float32).reshape(im.size[1], im.size[0], 3)
#print(type(image), image.shape)
result = server.serve(image)
print(result)

