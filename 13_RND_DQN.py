# Import modules
import tensorflow as tf
import pygame
import random
import numpy as np
import matplotlib.pyplot as plt
import datetime
import time
import cv2
import os

# Import game
import sys
sys.path.append("DQN_GAMES/")

import Parameters
game = Parameters.game

class RND_DQN:
	def __init__(self):

		# Game Information
		self.algorithm = 'RND_DQN'
		self.game_name = game.ReturnName()

		# Get parameters
		self.progress = ''
		self.Num_action = game.Return_Num_Action()

		# Parameters for RND
		self.beta = 0.2
		self.lamb = 1.0
		self.eta = 0.01
		self.extrinsic_coeff = 1.0
		self.intrinsic_coeff = 0.001

		# Initial parameters
		self.Num_Exploration = Parameters.Num_start_training
		self.Num_Training    = Parameters.Num_training
		self.Num_Testing     = Parameters.Num_test

		self.learning_rate = Parameters.Learning_rate
		self.gamma = Parameters.Gamma

		self.Num_plot_episode = Parameters.Num_plot_episode

		self.Is_train = Parameters.Is_train
		self.load_path = Parameters.Load_path

		self.step = 1
		self.score = 0
		self.episode = 0

		# date - hour - minute - second of training time
		self.date_time = str(datetime.date.today()) + '_' + \
            			 str(datetime.datetime.now().hour) + '_' + \
						 str(datetime.datetime.now().minute) + '_' + \
            			 str(datetime.datetime.now().second)

		# parameters for skipping and stacking
		self.state_set = []
		self.Num_skipping = Parameters.Num_skipFrame
		self.Num_stacking = Parameters.Num_stackFrame

		# Parameter for Experience Replay
		self.Num_replay_memory = Parameters.Num_replay_memory
		self.Num_batch = Parameters.Num_batch
		self.replay_memory = []

		# Parameter for Target Network
		self.Num_update_target = Parameters.Num_update

		# Parameters for network
		self.img_size = 80
		self.Num_colorChannel = Parameters.Num_colorChannel

		self.first_conv   = Parameters.first_conv
		self.second_conv  = Parameters.second_conv
		self.third_conv   = Parameters.third_conv
		self.first_dense  = Parameters.first_dense
		self.second_dense = Parameters.second_dense

		# Variables for tensorboard
		self.loss = 0
		self.maxQ = 0

		self.score_board = 0
		self.maxQ_board  = 0
		self.loss_board  = 0
		self.r_i_board = 0

		self.step_old    = 0
		self.episode_old = 0

		# Initialize Network
		self.input, self.output = self.network('network')
		self.input_target, self.output_target = self.network('target')
		self.s_next, self.r_i, self.Lf = self.RND()
		self.train_step, self.action_target, self.y_target, self.loss_train = self.loss_and_train()

		self.sess, self.saver, self.summary_placeholders, self.update_ops, self.summary_op, self.summary_writer = self.init_sess()

	def main(self):
		# Define game state
		game_state = game.GameState()

		# Initialization
		state = self.initialization(game_state)
		stacked_state = self.skip_and_stack_frame(state)

		while True:
			# Get progress:
			self.progress = self.get_progress()

			# Select action
			action = self.select_action(stacked_state)

			# Take action and get info. for update
			next_state, r_e, terminal = game_state.frame_step(action)
			next_state = self.reshape_input(next_state)
			stacked_next_state = self.skip_and_stack_frame(next_state)

			reward = r_e
			r_i = 0.
			
			# Experience Replay
			self.experience_replay(stacked_state, action, reward, stacked_next_state, terminal)

			# Training!
			if self.progress == 'Training':
				# Update target network
				if self.step % self.Num_update_target == 0:
					self.update_target()

				# Training
				r_i = self.train(self.replay_memory)

				# Save model
				self.save_model()

			# Update former info.
			stacked_state = stacked_next_state
			self.score += r_e
			self.step += 1

			# Plotting
			self.plotting(terminal, r_i)

			# If game is over (terminal)
			if terminal:
				stacked_state = self.if_terminal(game_state)

			# Finished!
			if self.progress == 'Finished':
				print('Finished!')
				break

	def init_sess(self):
		# Initialize variables
		config = tf.ConfigProto()
		config.gpu_options.allow_growth = True

		sess = tf.InteractiveSession(config=config)

		# Make folder for save data
		os.makedirs('saved_networks/' + self.game_name + '/' + self.date_time + '_' + self.algorithm)

		# Summary for tensorboard
		summary_placeholders, update_ops, summary_op = self.setup_summary()
		summary_writer = tf.summary.FileWriter('saved_networks/' + self.game_name + '/' + self.date_time + '_' + self.algorithm, sess.graph)

		init = tf.global_variables_initializer()
		sess.run(init)

		# Load the file if the saved file exists
		saver = tf.train.Saver()
		# check_save = 1
		check_save = input('Load Model? (1=yes/2=no): ')

		if check_save == 1:
			# Restore variables from disk.
			saver.restore(sess, self.load_path + "/model.ckpt")
			print("Model restored.")

			check_train = input('Inference or Training? (1=Inference / 2=Training): ')
			if check_train == 1:
				self.Num_Exploration = 0
				self.Num_Training = 0

		return sess, saver, summary_placeholders, update_ops, summary_op, summary_writer

	def initialization(self, game_state):
		action = np.zeros([self.Num_action])
		state, _, _ = game_state.frame_step(action)
		state = self.reshape_input(state)

		for i in range(self.Num_skipping * self.Num_stacking):
			self.state_set.append(state)

		return state

	def skip_and_stack_frame(self, state):
		self.state_set.append(state)

		state_in = np.zeros((self.img_size, self.img_size, self.Num_colorChannel * self.Num_stacking))

		# Stack the frame according to the number of skipping frame
		for stack_frame in range(self.Num_stacking):
			state_in[:,:,stack_frame] = self.state_set[-1 - (self.Num_skipping * stack_frame)]

		del self.state_set[0]

		state_in = np.uint8(state_in)
		return state_in

	def get_progress(self):
		progress = ''
		if self.step <= self.Num_Exploration:
			progress = 'Exploring'
		elif self.step <= self.Num_Exploration + self.Num_Training:
			progress = 'Training'
		elif self.step <= self.Num_Exploration + self.Num_Training + self.Num_Testing:
			progress = 'Testing'
		else:
			progress = 'Finished'

		return progress

	# Resize and make input as grayscale
	def reshape_input(self, state):
		state_out = cv2.resize(state, (self.img_size, self.img_size))
		if self.Num_colorChannel == 1:
			state_out = cv2.cvtColor(state_out, cv2.COLOR_BGR2GRAY)
			state_out = np.reshape(state_out, (self.img_size, self.img_size))

		state_out = np.uint8(state_out)

		return state_out

	# Code for tensorboard
	def setup_summary(self):
		episode_score = tf.Variable(0.)
		episode_maxQ = tf.Variable(0.)
		episode_loss = tf.Variable(0.)
		episode_r_i = tf.Variable(0.)

		tf.summary.scalar('Average Score/' + str(self.Num_plot_episode) + ' episodes', episode_score)
		tf.summary.scalar('Average MaxQ/' + str(self.Num_plot_episode) + ' episodes', episode_maxQ)
		tf.summary.scalar('Average Loss/' + str(self.Num_plot_episode) + ' episodes', episode_loss)
		tf.summary.scalar('Average r_i/' + str(self.Num_plot_episode) + ' episodes', episode_r_i)

		summary_vars = [episode_score, episode_maxQ, episode_loss, episode_r_i]

		summary_placeholders = [tf.placeholder(tf.float32) for _ in range(len(summary_vars))]
		update_ops = [summary_vars[i].assign(summary_placeholders[i]) for i in range(len(summary_vars))]
		summary_op = tf.summary.merge_all()
		return summary_placeholders, update_ops, summary_op

	# Convolution and pooling
	def conv2d(self, x, w, stride):
		return tf.nn.conv2d(x,w,strides=[1, stride, stride, 1], padding='SAME')

	# Get Variables
	def conv_weight_variable(self, name, shape):
	    return tf.get_variable(name, shape = shape, initializer = tf.contrib.layers.xavier_initializer_conv2d())

	def weight_variable(self, name, shape):
	    return tf.get_variable(name, shape = shape, initializer = tf.contrib.layers.xavier_initializer())

	def bias_variable(self, name, shape):
	    return tf.get_variable(name, shape = shape, initializer = tf.contrib.layers.xavier_initializer())

	def network(self, network_name):
		# Input
		x_image = tf.placeholder(tf.float32, shape = [None,
													  self.img_size,
													  self.img_size,
													  self.Num_stacking * self.Num_colorChannel])

		x_normalize = (x_image - (255.0/2)) / (255.0/2)

		with tf.variable_scope(network_name):
			# Convolution variables
			w_conv1 = self.conv_weight_variable('_w_conv1', self.first_conv)
			b_conv1 = self.bias_variable('_b_conv1',[self.first_conv[3]])

			w_conv2 = self.conv_weight_variable('_w_conv2',self.second_conv)
			b_conv2 = self.bias_variable('_b_conv2',[self.second_conv[3]])

			w_conv3 = self.conv_weight_variable('_w_conv3',self.third_conv)
			b_conv3 = self.bias_variable('_b_conv3',[self.third_conv[3]])

			# Densely connect layer variables
			w_fc1 = self.weight_variable('_w_fc1',self.first_dense)
			b_fc1 = self.bias_variable('_b_fc1',[self.first_dense[1]])

			w_fc2 = self.weight_variable('_w_fc2',self.second_dense)
			b_fc2 = self.bias_variable('_b_fc2',[self.second_dense[1]])

		# Network
		h_conv1 = tf.nn.relu(self.conv2d(x_normalize, w_conv1, 4) + b_conv1)
		h_conv2 = tf.nn.relu(self.conv2d(h_conv1, w_conv2, 2) + b_conv2)
		h_conv3 = tf.nn.relu(self.conv2d(h_conv2, w_conv3, 1) + b_conv3)

		h_flat = tf.reshape(h_conv3, [-1, self.first_dense[0]])
		h_fc1 = tf.nn.relu(tf.matmul(h_flat, w_fc1)+b_fc1)

		output = tf.matmul(h_fc1, w_fc2) + b_fc2
		return x_image, output

	# RND Module
	def RND(self):
		# Input
		s_next = tf.placeholder(tf.float32, shape = [None,
												     self.img_size,
												     self.img_size,
												     self.Num_stacking * self.Num_colorChannel])

		s_next = (s_next - (255.0/2)) / (255.0/2)

		with tf.variable_scope('curiosity_predict'):
			# Convolution variables
			w_conv1 = self.conv_weight_variable('_w_conv1', [3,3,self.Num_stacking * self.Num_colorChannel,32])
			b_conv1 = self.bias_variable('_b_conv1',[32])

			w_conv2 = self.conv_weight_variable('_w_conv2', [3,3,32,32])
			b_conv2 = self.bias_variable('_b_conv2',[32])

			w_conv3 = self.conv_weight_variable('_w_conv3', [3,3,32,32])
			b_conv3 = self.bias_variable('_b_conv3',[32])

			w_conv4 = self.conv_weight_variable('_w_conv4', [3,3,32,32])
			b_conv4 = self.bias_variable('_b_conv4',[32])

		with tf.variable_scope('curiosity_target'):
			# Convolution variables
			w_conv1_t = self.conv_weight_variable('_w_conv1', [3,3,self.Num_stacking * self.Num_colorChannel,32])
			b_conv1_t = self.bias_variable('_b_conv1',[32])

			w_conv2_t = self.conv_weight_variable('_w_conv2', [3,3,32,32])
			b_conv2_t = self.bias_variable('_b_conv2',[32])

			w_conv3_t = self.conv_weight_variable('_w_conv3', [3,3,32,32])
			b_conv3_t = self.bias_variable('_b_conv3',[32])

			w_conv4_t = self.conv_weight_variable('_w_conv4', [3,3,32,32])
			b_conv4_t = self.bias_variable('_b_conv4',[32])

		# Feature Vector
		conv1 = tf.nn.elu(self.conv2d(s_next, w_conv1, 2) + b_conv1)
		conv2 = tf.nn.elu(self.conv2d(conv1, w_conv2, 2) + b_conv2)
		conv3 = tf.nn.elu(self.conv2d(conv2, w_conv3, 2) + b_conv3)
		conv4 = tf.nn.elu(self.conv2d(conv3, w_conv4, 2) + b_conv4)
			
		conv4_flat = tf.layers.flatten(conv4)

		conv1_t = tf.nn.elu(self.conv2d(s_next, w_conv1_t, 2) + b_conv1_t)
		conv2_t = tf.nn.elu(self.conv2d(conv1_t, w_conv2_t, 2) + b_conv2_t)
		conv3_t = tf.nn.elu(self.conv2d(conv2_t, w_conv3_t, 2) + b_conv3_t)
		conv4_t = tf.nn.elu(self.conv2d(conv3_t, w_conv4_t, 2) + b_conv4_t)

		conv4_t_flat = tf.layers.flatten(conv4_t)

		# Intrinsic Reward
		r_i = (self.eta * 0.5) * tf.reduce_sum(tf.square(tf.subtract(conv4_flat, conv4_t_flat)), axis = 1)

		Lf = tf.losses.mean_squared_error(conv4_flat, conv4_t_flat)

		return s_next, r_i, Lf

	def loss_and_train(self):
		# Loss function and Train
		action_target = tf.placeholder(tf.float32, shape = [None, self.Num_action])
		y_target = tf.placeholder(tf.float32, shape = [None])

		y_prediction = tf.reduce_sum(tf.multiply(self.output, action_target), reduction_indices = 1)

		Loss = self.lamb * tf.reduce_mean(tf.square(y_prediction - y_target)) + (self.beta * self.Lf)

		trainable_variables = tf.trainable_variables()
		trainable_var_network = [var for var in trainable_variables if var.name.startswith('network')]
		trainable_var_predict = [var for var in trainable_variables if var.name.startswith('curiosity_predict')]

		train_vars = trainable_var_network + trainable_var_predict

		train_step = tf.train.AdamOptimizer(learning_rate = self.learning_rate, epsilon = 1e-02).minimize(Loss, var_list=train_vars)

		return train_step, action_target, y_target, Loss

	def select_action(self, stacked_state):
		action = np.zeros([self.Num_action])
		action_index = 0

		# Choose action
		if self.progress == 'Exploring':
			# Choose random action
			action_index = random.randint(0, self.Num_action-1)
			action[action_index] = 1

		elif self.progress == 'Training':
			# Choose greedy action
			Q_value = self.output.eval(feed_dict={self.input: [stacked_state]})
			action_index = np.argmax(Q_value)
			action[action_index] = 1
			self.maxQ = np.max(Q_value)

		elif self.progress == 'Testing':
			# Choose greedy action
			Q_value = self.output.eval(feed_dict={self.input: [stacked_state]})
			action_index = np.argmax(Q_value)
			action[action_index] = 1
			self.maxQ = np.max(Q_value)

		return action

	def experience_replay(self, state, action, reward, next_state, terminal):
		# If Replay memory is longer than Num_replay_memory, delete the oldest one
		if len(self.replay_memory) >= self.Num_replay_memory:
			del self.replay_memory[0]

		self.replay_memory.append([state, action, reward, next_state, terminal])

	def update_target(self):
		# Get trainable variables
		trainable_variables = tf.trainable_variables()
		# network variables
		trainable_variables_network = [var for var in trainable_variables if var.name.startswith('network')]

		# target variables
		trainable_variables_target = [var for var in trainable_variables if var.name.startswith('target')]

		for i in range(len(trainable_variables_network)):
			self.sess.run(tf.assign(trainable_variables_target[i], trainable_variables_network[i]))

	def train(self, replay_memory):
		# Select minibatch
		minibatch =  random.sample(replay_memory, self.Num_batch)

		# Save the each batch data
		state_batch      = [batch[0] for batch in minibatch]
		action_batch     = [batch[1] for batch in minibatch]
		reward_batch     = [batch[2] for batch in minibatch]
		next_state_batch = [batch[3] for batch in minibatch]
		terminal_batch   = [batch[4] for batch in minibatch]

		# Get y_prediction
		y_batch = []
		Q_batch = self.output_target.eval(feed_dict = {self.input_target: next_state_batch})

		r_i = self.sess.run(self.r_i, feed_dict = {self.s_next: next_state_batch})
		
		for i in range(len(reward_batch)):
			reward_batch[i] = (self.extrinsic_coeff * reward_batch[i]) + (self.intrinsic_coeff * r_i[i])

		# Get target values
		for i in range(len(minibatch)):
			if terminal_batch[i] == True:
				y_batch.append(reward_batch[i])
			else:
				y_batch.append(reward_batch[i] + self.gamma * np.max(Q_batch[i]))

		_, self.loss = self.sess.run([self.train_step, self.loss_train], feed_dict = {self.action_target: action_batch,
																					  self.y_target: y_batch,
																					  self.input: state_batch,
																					  self.s_next: next_state_batch})

		return np.mean(self.intrinsic_coeff * r_i)

	def save_model(self):
		# Save the variables to disk.
		if self.step == self.Num_Exploration + self.Num_Training:
		    save_path = self.saver.save(self.sess, 'saved_networks/' + self.game_name + '/' + self.date_time + '_' + self.algorithm + "/model.ckpt")
		    print("Model saved in file: %s" % save_path)

	def plotting(self, terminal, r_i):
		if self.progress != 'Exploring':
			if terminal:
				self.score_board += self.score

			self.maxQ_board += self.maxQ
			self.loss_board += self.loss
			self.r_i_board  += r_i

			if self.episode % self.Num_plot_episode == 0 and self.episode != 0 and terminal:
				diff_step = self.step - self.step_old
				diff_episode = self.episode - self.episode_old

				tensorboard_info = [self.score_board / diff_episode, self.maxQ_board / diff_step, self.loss_board / diff_step, self.r_i_board / diff_step]

				for i in range(len(tensorboard_info)):
				    self.sess.run(self.update_ops[i], feed_dict = {self.summary_placeholders[i]: float(tensorboard_info[i])})
				summary_str = self.sess.run(self.summary_op)
				self.summary_writer.add_summary(summary_str, self.step)

				self.score_board = 0
				self.maxQ_board  = 0
				self.loss_board  = 0
				self.r_i_board   = 0

				self.step_old = self.step
				self.episode_old = self.episode
		else:
			self.step_old = self.step

	def if_terminal(self, game_state):
		# Show Progress
		print('Step: ' + str(self.step) + ' / ' +
		      'Episode: ' + str(self.episode) + ' / ' +
			  'Progress: ' + self.progress + ' / ' +
			  'Score: ' + str(self.score))

		if self.progress != 'Exploring':
			self.episode += 1
		self.score = 0

		# If game is finished, initialize the state
		state = self.initialization(game_state)
		stacked_state = self.skip_and_stack_frame(state)

		return stacked_state

if __name__ == '__main__':
	agent = RND_DQN()
	agent.main()
