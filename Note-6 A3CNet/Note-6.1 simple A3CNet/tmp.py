# Set up optimizer with global norm clipping.
trainable_variables = tf.trainable_variables()
grads, _ = tf.clip_by_global_norm(
    tf.gradients(self._loss, trainable_variables), max_gard_norm)
global_step = tf.get_variable(
    name="global_step",
    shape=[],
    dtype=tf.int64,
    initializer=tf.zeros_initializer(),
    trainable=False,
    collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.GLOBAL_STEP])
optimizer = tf.train.RMSPropOptimizer(
    learning_rate=learning_rate, epsilon=optimizer_epsilon)
self._train_op = optimizer.apply_gradients(
    zip(grads, trainable_variables), global_step=global_step)