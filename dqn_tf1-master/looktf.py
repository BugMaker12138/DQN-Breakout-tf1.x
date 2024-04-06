# import tensorflow as tf
#
# # 检查是否有可用的GPU设备
# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#     print('Default GPU Device: {}'.format(gpus[0].name))
# else:
#     print("Please install GPU version of TF or check if your GPU is properly installed and enabled.")

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
