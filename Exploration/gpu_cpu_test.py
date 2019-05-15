# import tensorflow as tf
# with tf.device('/gpu:0'):
#     a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
#     b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
#     c = tf.matmul(a, b)

# with tf.Session() as sess:
#   devices = sess.list_devices()

# with tf.Session() as sess:
#     print (sess.run(c))

# import tensorflow.python.framework.test_util

# tf = tensorflow

# tf.test.is_gpu_available(
#     cuda_only=False,
#     min_cuda_compute_capability=None
# )