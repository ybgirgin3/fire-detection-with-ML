import tensorflow as tf

print('gpu: {}'.format(tf.test.is_gpu_available()))
print('cuda: {}'.format(tf.test.is_built_with_cuda()))
