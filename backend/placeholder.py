def placeholder(dtype, shape=None, name=None):
  if context.executing_eagerly():
    raise RuntimeError("tf.placeholder() is not compatible with "
                       "eager execution.")

  return placeholder(dtype=dtype, shape=shape, name=name)