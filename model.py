import tensorflow as tf
import numpy as np

samples = np.zeros(shape=(1, 32, 32, 1))

sess = tf.Session()
new_saver = tf.train.import_meta_graph('model/default.ckpt.meta')
new_saver.restore(sess, 'model/default.ckpt')
# tf.get_collection() returns a list. In this example we only want the
# first one.
# test = tf.get_default_graph().get_all_collection_keys()
# print(tf.get_default_graph().get_collection('prediction'))
test_prediction = tf.get_default_graph().get_tensor_by_name('test/single_prediction:0')
single_input = tf.get_default_graph().get_tensor_by_name('test/single_input:0')

for i in range(10):
	result = sess.run(
		test_prediction,
		feed_dict={single_input: samples}
	)
	print(result)
