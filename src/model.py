import tensorflow as tf
from nets.policies import get_policy
from nets.value_functions import MLPValueFunc

# Implementation of Deterministic Soft Policy Gradients (DSPG)

class SoftPolicyGradient(object):
    def __init__(self, sess, conf, state_shape, action_shape):
        self.sess = sess
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.tau = conf.tau
        self.gamma = conf.gamma
        self.global_norm = conf.global_norm
        self.num_samples = conf.num_samples
        self.actor_lr = conf.actor_learning_rate
        self.Q_critic_lr = conf.Q_critic_learning_rate

        self.policy = get_policy(conf.pi_type, conf.hidden_dims_pi, conf.reg, conf.reg_factor)
        self.value = MLPValueFunc(conf.hidden_dims_Q, conf.hidden_dims_Q)

        # self._init_placeholders() #-------------------------------
        self.state = state = tf.placeholder(tf.float32, [None] + list(self.state_shape), name="obs_ph")
        self.action = action = tf.placeholder(tf.float32, (None, self.action_shape), name="act_ph")
        self.reward = tf.placeholder(tf.float32, [None], name="r_ph")
        self.next_state = tf.placeholder(tf.float32, [None] + list(self.state_shape), name="nextobs_ph")
        self.done = tf.placeholder(tf.float32, [None], name="done_ph")
        # networks
        self.sampled_action, self.deterministic_action = self.sample_pi_network(state, 'pi')  # [-inf, inf]
        self.Q = self.value.Q_network(state, tf.tanh(action), 'Q')
        _, _ = self.sample_pi_network(state, 'target_pi')
        self.target_Q = self.value.Q_network(state, tf.tanh(action), 'target_Q')

        # self._init_actor_update()  #-------------------------------
        
        # constructing pi loss
        apply_grads = True
        Q_sampled, log_pi_sampled = self.get_sampled_Q_and_pi(self.state, 'Q', 'pi', apply_grads)
        

        self.pi_variables = pi_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='pi/')

        if apply_grads:
            pass
            advantage = tf.subtract(Q_sampled, log_pi_sampled, name="advantage")
            advantage = tf.stop_gradient(advantage)
            # Whiten the advantage.
            # adv_mean = tf.reduce_mean(advantage)
            # advantage = advantage - adv_mean 

            
            pi_grad = tf.gradients(log_pi_sampled, pi_variables, -advantage)
            pi_grad, _ = tf.clip_by_global_norm(pi_grad, self.global_norm)
            
            
            self.pi_loss = tf.multiply(log_pi_sampled, (-advantage), name="pi_loss_apply_grad") # just a placeholder

            self.train_pi = tf.train.AdamOptimizer(
                learning_rate=self.actor_lr).apply_gradients(zip(pi_grad, pi_variables), name="apply_grad_adam")

        else:
            advantage = tf.subtract(Q_sampled, log_pi_sampled, name="advantage")
            self.pi_loss = tf.multiply(log_pi_sampled, (-advantage), name="pi_loss")
            self.train_pi = tf.train.AdamOptimizer(
                learning_rate=self.actor_lr).minimize(-advantage, var_list=pi_variables, name="adam_pi")

        
        
        
        # Why apply gradients above when you can just minimize. :)
        # self.train_pi = tf.train.AdamOptimizer(
        #     learning_rate=self.actor_lr).minimize(self.pi_loss, var_list=self.pi_variables)


        # My logic to instead simply maximise q value along the pi action, so as to only train variables in scope main/pi
        

        # self._init_critic_update() #-------------------------------
        # constructing Q loss
        target_Q_s2_sampled, log_pi_s2_sampled = self.get_sampled_Q_and_pi(self.next_state, 'target_Q', 'target_pi')
        target_Q_s2_sampled = tf.reduce_mean(tf.reshape(target_Q_s2_sampled, (self.num_samples, -1)), axis=0)
        log_pi_s2_sampled = tf.reduce_mean(tf.reshape(log_pi_s2_sampled, (self.num_samples, -1)), axis=0)
        target_Q = tf.add(self.reward , (1 - self.done) * self.gamma * (target_Q_s2_sampled - log_pi_s2_sampled), name="q_backup")
        self.Q_loss = Q_loss = tf.reduce_mean(tf.square(self.Q - target_Q), name="shi_sqr_q_loss")

        self.Q_variables = Q_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Q/')
        self.train_Q = tf.train.AdamOptimizer(
            learning_rate=self.Q_critic_lr).minimize(Q_loss, var_list=Q_variables, name="s_train_q_op")

        # self._init_target_ops() #-------------------------------
        target_Q_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='target_Q/')
        target_pi_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='target_pi/')
        with tf.name_scope('soft_polyak'):
            # soft update
            soft_update_Q_ops = [tf.assign(xbar, self.tau*x + (1 - self.tau)*xbar)
                                    for (xbar, x) in zip(target_Q_variables, self.Q_variables)]
            soft_update_pi_ops = [tf.assign(xbar, self.tau * x + (1 - self.tau) * xbar)
                                for (xbar, x) in zip(target_pi_variables, self.pi_variables)]
            soft_update_Q_ops.extend(soft_update_pi_ops)
            self.soft_update_ops = tf.group(*soft_update_Q_ops)
        with tf.name_scope('hard_polyak'):
            # hard update
            hard_update_Q_ops = [tf.assign(xbar, x) for (xbar, x) in zip(target_Q_variables, self.Q_variables)]
            hard_update_pi_ops = [tf.assign(xbar, x) for (xbar, x) in zip(target_pi_variables, self.pi_variables)]
            hard_update_Q_ops.extend(hard_update_pi_ops)
            self.hard_update_ops = tf.group(*hard_update_Q_ops)

        sess.run(tf.global_variables_initializer())
        sess.run(self.hard_update_ops)


    def get_sampled_Q_and_pi(self, state, name_Q, name_pi, apply_grads=True):
        state = tf.tile(state, [self.num_samples, 1])
        sampled_action, _ = self.sample_pi_network(state, name_pi, reuse=True)
        if(apply_grads):
            sampled_action = tf.stop_gradient(sampled_action)
        Q_sampled = self.value.Q_network(state, tf.tanh(sampled_action), name_Q, reuse=True)
        log_pi_sampled = self.pi_network_log_prob(sampled_action, state, name_pi, reuse=True)
        return Q_sampled, log_pi_sampled

    def pi_network_log_prob(self, action, state, name, reuse=None):
        with tf.variable_scope(name, reuse=reuse):
            state_feature = self.policy.feature_extractor(state)
            parameters = self.policy.produce_policy_parameters(self.action_shape, state_feature)
            log_prob = self.policy.policy_parameters_to_log_prob(action, parameters)
        return log_prob


    def sample_pi_network(self, state, name, reuse=None):
        with tf.variable_scope(name, reuse=reuse):
            state_feature = self.policy.feature_extractor(state)
            parameters = self.policy.produce_policy_parameters(self.action_shape, state_feature)
            sample = self.policy.policy_parameters_to_sample(parameters)
        return sample, parameters[0]


    def sampling_actions(self, state, is_deterministic=False):
        if is_deterministic:
            return self.sess.run(self.deterministic_action, feed_dict={self.state: state})
        else:
            return self.sess.run(self.sampled_action, feed_dict={self.state: state})


    def trainer(self, transitions):
        state, action, reward, next_state, done = transitions
        [_, _, Q, Q_loss, pi_loss] = self.sess.run(
            [self.train_Q, self.train_pi, self.Q, self.Q_loss, self.pi_loss],
            feed_dict={self.state: state,
                       self.action: action,
                       self.reward: reward,
                       self.next_state: next_state,
                       self.done: done
                       })
        self.sess.run(self.soft_update_ops)
        return Q, Q_loss, pi_loss

