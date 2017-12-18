import tensorflow as tf
from env.main import Account
from agent.main import Agent, Access, Framework


env = Account()
init = env.reset()
print(init.shape)


name = 'W0'
input_shape = [381, 240, 58]
action_size = 3
A = Access(input_shape, action_size)
W0 = Agent(name, A, input_shape, action_size)

print(W0.a_interface)

# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     W0.init_or_update_local(sess)
#     da = W0.get_deterministic_policy_action(sess, init)
#     sa = W0.get_stochastic_action(sess, init)
#
#     next_state, reward, done = env.step(sa)
#     print(next_state.shape)
#     print(reward)
#     print(done)



