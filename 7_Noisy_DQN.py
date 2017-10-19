# Import modules
import sys
import pygame
import tensorflow as tf
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
import datetime
import time

# Parameter Setting
import Deep_Parameters
game = Deep_Parameters.game

algorithm = 'Noisy_DQN'

Num_action = game.Return_Num_Action()
game_name = game.ReturnName()

Gamma = Deep_Parameters.Gamma
Learning_rate = Deep_Parameters.Learning_rate

Num_replay_memory = Deep_Parameters.Num_replay_memory
Num_start_training = Deep_Parameters.Num_start_training
Num_training = Deep_Parameters.Num_training
Num_update = Deep_Parameters.Num_update
Num_batch = Deep_Parameters.Num_batch
Num_test = Deep_Parameters.Num_test
Num_skipFrame = Deep_Parameters.Num_skipFrame
Num_stackFrame = Deep_Parameters.Num_stackFrame
Num_colorChannel = Deep_Parameters.Num_colorChannel

Num_plot_episode = Deep_Parameters.Num_plot_episode
Num_step_save = Deep_Parameters.Num_step_save

GPU_fraction = Deep_Parameters.GPU_fraction
Is_train = Deep_Parameters.Is_train

# Parametwrs for Network
img_size = Deep_Parameters.img_size

first_conv   = Deep_Parameters.first_conv
second_conv  = Deep_Parameters.second_conv
third_conv   = Deep_Parameters.third_conv
first_dense  = Deep_Parameters.first_dense
second_dense = Deep_Parameters.second_dense

# If is train is false then immediately start testing
if Is_train == False:
	Num_start_training = 0
	Num_training = 0

# Initialize weights and bias
def weight_variable(shape):
    return tf.Variable(xavier_initializer(shape))

def bias_variable(shape):
	return tf.Variable(xavier_initializer(shape))

def mu_variable(shape):
    return tf.Variable(tf.random_uniform(shape, minval = -tf.sqrt(3/shape[0]), maxval = tf.sqrt(3/shape[0])))

def sigma_variable(shape):
	return tf.Variable(tf.constant(0.017, shape = shape))

# Xavier Weights initializer
def xavier_initializer(shape):
	dim_sum = np.sum(shape)
	if len(shape) == 1:
		dim_sum += 1
	bound = np.sqrt(2.0 / dim_sum)
	return tf.random_uniform(shape, minval=-bound, maxval=bound)

# Convolution and pooling
def conv2d(x,w, stride):
	return tf.nn.conv2d(x,w,strides=[1, stride, stride, 1], padding='SAME')

def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

def assign_network_to_target():
	# Get trainable variables
	trainable_variables = tf.trainable_variables()
	# network lstm variables
	trainable_variables_network = [var for var in trainable_variables if var.name.startswith('network')]

	# target lstm variables
	trainable_variables_target = [var for var in trainable_variables if var.name.startswith('target')]

	for i in range(len(trainable_variables_network)):
		sess.run(tf.assign(trainable_variables_target[i], trainable_variables_network[i]))

def resize_input(observation):
	observation_out = cv2.resize(observation, (img_size, img_size))
	if Num_colorChannel == 1:
		observation_out = cv2.cvtColor(observation_out, cv2.COLOR_BGR2GRAY)
		observation_out = np.reshape(observation_out, (img_size, img_size))

	observation_out = np.uint8(observation_out)
	return observation_out

########################################### Noisy Network ###########################################
def noisy_dense(input_, input_shape, mu_w, sig_w, mu_b, sig_b, is_train_process):
	eps_w = tf.cond(is_train_process, lambda: tf.random_normal(input_shape), lambda: tf.zeros(input_shape))
	eps_b = tf.cond(is_train_process, lambda: tf.random_normal([input_shape[1]]), lambda: tf.zeros([input_shape[1]]))

	# if is_train_process == True:
	# 	eps_w = tf.random_normal(input_shape)
	# 	eps_b = tf.random_normal([input_shape[1]])
	# else:
	# 	eps_w = tf.zeros(input_shape)
	# 	eps_b = tf.zeros([input_shape[1]])

	w_fc = tf.add(mu_w, tf.multiply(sig_w, eps_w))
	b_fc = tf.add(mu_b, tf.multiply(sig_b, eps_b))

	return tf.matmul(input_, w_fc) + b_fc
#####################################################################################################

# Input
x_image = tf.placeholder(tf.float32, shape = [None, img_size, img_size, Num_colorChannel * Num_stackFrame])
x_normalize = (x_image - (255.0/2)) / (255.0/2)

########################################### Noisy Network ###########################################
train_process = tf.placeholder(tf.bool)
#####################################################################################################

with tf.variable_scope('network'):
	# Convolution variables
	w_conv1 = weight_variable(first_conv)
	b_conv1 = bias_variable([first_conv[3]])

	w_conv2 = weight_variable(second_conv)
	b_conv2 = bias_variable([second_conv[3]])

	w_conv3 = weight_variable(third_conv)
	b_conv3 = bias_variable([third_conv[3]])

########################################### Noisy Network ###########################################
	# Densely connect layer variables (Noisy)
	mu_w1  = mu_variable(first_dense)
	sig_w1 = sigma_variable(first_dense)
	mu_b1  = mu_variable([first_dense[1]])
	sig_b1 = sigma_variable([first_dense[1]])
	# eps_w1 = tf.random_normal(first_dense)
	# eps_b1 = tf.random_normal([first_dense[1]])
	#
	# w_fc1 = tf.add(mu_w1, tf.multiply(sig_w1, eps_w1))
	# b_fc1 = tf.add(mu_b1, tf.multiply(sig_b1, eps_b1))

	mu_w2  = mu_variable(second_dense)
	sig_w2 = sigma_variable(second_dense)
	mu_b2  = mu_variable([second_dense[1]])
	sig_b2 = sigma_variable([second_dense[1]])
	# eps_w2 = tf.random_normal(second_dense)
	# eps_b2 = tf.random_normal([second_dense[1]])
	#
	# w_fc2 = tf.add(mu_w2, tf.multiply(sig_w2, eps_w2))
	# b_fc2 = tf.add(mu_b2, tf.multiply(sig_b2, eps_b2))
#####################################################################################################

# Network
h_conv1 = tf.nn.relu(conv2d(x_normalize, w_conv1, 4) + b_conv1)
h_conv2 = tf.nn.relu(conv2d(h_conv1, w_conv2, 2) + b_conv2)
h_conv3 = tf.nn.relu(conv2d(h_conv2, w_conv3, 1) + b_conv3)

h_pool3_flat = tf.reshape(h_conv3, [-1, first_dense[0]])

# h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, w_fc1)+b_fc1)
# h_fc2 = tf.nn.relu(tf.matmul(h_fc1, w_fc2)+b_fc2)
#
# output = tf.matmul(h_fc2, w_fc3) + b_fc3
########################################### Noisy Network ###########################################
h_fc1 = tf.nn.relu(noisy_dense(h_pool3_flat, first_dense, mu_w1, sig_w1, mu_b1, sig_b1, train_process))
output = noisy_dense(h_fc1, second_dense, mu_w2, sig_w2, mu_b2, sig_b2, train_process)
#####################################################################################################

with tf.variable_scope('target'):
	# Convolution variables target
	w_conv1_target = weight_variable(first_conv)
	b_conv1_target = bias_variable([first_conv[3]])

	w_conv2_target = weight_variable(second_conv)
	b_conv2_target = bias_variable([second_conv[3]])

	w_conv3_target = weight_variable(third_conv)
	b_conv3_target = bias_variable([third_conv[3]])

########################################### Noisy Network ###########################################
	# Densely connect layer variables target (Noisy)
	mu_w1_target  = mu_variable(first_dense)
	sig_w1_target = sigma_variable(first_dense)
	mu_b1_target  = mu_variable([first_dense[1]])
	sig_b1_target = sigma_variable([first_dense[1]])
	# eps_w1_target = tf.random_normal(first_dense)
	# eps_b1_target = tf.random_normal([first_dense[1]])
	#
	# w_fc1_target = tf.add(mu_w1_target, tf.multiply(sig_w1_target, eps_w1_target))
	# b_fc1_target = tf.add(mu_b1_target, tf.multiply(sig_b1_target, eps_b1_target))

	mu_w2_target  = mu_variable(second_dense)
	sig_w2_target = sigma_variable(second_dense)
	mu_b2_target  = mu_variable([second_dense[1]])
	sig_b2_target = sigma_variable([second_dense[1]])
	# eps_w2_target = tf.random_normal(second_dense)
	# eps_b2_target = tf.random_normal([second_dense[1]])
	#
	# w_fc2_target = tf.add(mu_w2_target, tf.multiply(sig_w2_target, eps_w2_target))
	# b_fc2_target = tf.add(mu_b2_target, tf.multiply(sig_b2_target, eps_b2_target))
#####################################################################################################

# Target Network
h_conv1_target = tf.nn.relu(conv2d(x_normalize, w_conv1_target, 4) + b_conv1_target)
h_conv2_target = tf.nn.relu(conv2d(h_conv1_target, w_conv2_target, 2) + b_conv2_target)
h_conv3_target = tf.nn.relu(conv2d(h_conv2_target, w_conv3_target, 1) + b_conv3_target)

h_pool3_flat_target = tf.reshape(h_conv3_target, [-1, first_dense[0]])
# h_fc1_target = tf.nn.relu(tf.matmul(h_pool3_flat_target, w_fc1_target)+b_fc1_target)
# h_fc2_target = tf.nn.relu(tf.matmul(h_fc1_target, w_fc2_target)+b_fc2_target)
#
# output_target = tf.matmul(h_fc2_target, w_fc3_target) + b_fc3_target

########################################### Noisy Network ###########################################
h_fc1_target = tf.nn.relu(noisy_dense(h_pool3_flat_target, first_dense, mu_w1_target, sig_w1_target, mu_b1_target, sig_b1_target, train_process))
output_target = noisy_dense(h_fc1_target, second_dense, mu_w2_target, sig_w2_target, mu_b2_target, sig_b2_target, train_process)
#####################################################################################################

# Loss function and Train
action_target = tf.placeholder(tf.float32, shape = [None, Num_action])
y_prediction = tf.placeholder(tf.float32, shape = [None])

y_target = tf.reduce_sum(tf.multiply(output, action_target), reduction_indices = 1)
Loss = tf.reduce_mean(tf.square(y_prediction - y_target))
train_step = tf.train.AdamOptimizer(learning_rate = Learning_rate, epsilon = 1e-02).minimize(Loss)

# Initialize variables
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = GPU_fraction

sess = tf.InteractiveSession(config=config)
init = tf.global_variables_initializer()
sess.run(init)

# Load the file if the saved file exists
saver = tf.train.Saver()
check_save = input('Is there any saved data?(1=y/2=n): ')

if check_save == 1:
    checkpoint = tf.train.get_checkpoint_state("7_saved_networks_Noisy_DQN")
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print("Successfully loaded:", checkpoint.model_checkpoint_path)
    else:
        print("Could not find old network weights")

# Initial parameters
Replay_memory = []
step = 1
score = 0
episode = 0

# date - hour - minute of training time
date_time = str(datetime.date.today()) + '_' + str(datetime.datetime.now().hour) + '_' + str(datetime.datetime.now().minute)

game_state = game.GameState()
action = np.zeros([Num_action])
observation, _, _ = game_state.frame_step(action)
observation = resize_input(observation)

observation_in = np.zeros([img_size, img_size, Num_colorChannel * Num_stackFrame])
observation_next_in = np.zeros([img_size, img_size, Num_colorChannel * Num_stackFrame])

observation_set = []

start_time = time.time()

for i in range(Num_skipFrame * Num_stackFrame):
	observation_set.append(observation)

# Figure and figure data setting
plt.figure(1)
plot_x = []
plot_y = []

test_score = []

check_plot = 0
# Training & Testing
while True:
	if step <= Num_start_training:
		# Observation
		progress = 'Observing'

		action = np.zeros([Num_action])
		action[random.randint(0, Num_action - 1)] = 1.0

		observation_next, reward, terminal = game_state.frame_step(action)
		observation_next = resize_input(observation_next)

		observation_set.append(observation_next)

		observation_next_in = np.zeros((img_size, img_size, Num_colorChannel * Num_stackFrame))

		# Stack the frame according to the number of skipping frame
		for stack_frame in range(Num_stackFrame):
			observation_next_in[:,:,stack_frame] = observation_set[-1 - (Num_skipFrame * stack_frame)]

		del observation_set[0]

		observation_next_in = np.uint8(observation_next_in)

	elif step <= Num_start_training + Num_training:
		# Training
		progress = 'Training'

		########################################### Noisy Network ###########################################
		# Select action with max Q from Noisy network
		Q_value = output.eval(feed_dict={x_image: [observation_in], train_process: True})
		action = np.zeros([Num_action])
		action[np.argmax(Q_value)] = 1
		#####################################################################################################

		observation_next, reward, terminal = game_state.frame_step(action)
		observation_next = resize_input(observation_next)

		observation_set.append(observation_next)

		observation_next_in = np.zeros((img_size, img_size, Num_colorChannel * Num_stackFrame))

		# Stack the frame according to the number of skipping frame
		for stack_frame in range(Num_stackFrame):
			observation_next_in[:,:,stack_frame] = observation_set[-1 - (Num_skipFrame * stack_frame)]

		del observation_set[0]

		observation_next_in = np.uint8(observation_next_in)

		# Select minibatch
		minibatch =  random.sample(Replay_memory, Num_batch)

		# Save the each batch data
		observation_batch      = [batch[0] for batch in minibatch]
		action_batch           = [batch[1] for batch in minibatch]
		reward_batch           = [batch[2] for batch in minibatch]
		observation_next_batch = [batch[3] for batch in minibatch]
		terminal_batch 	       = [batch[4] for batch in minibatch]

		# Update target network according to the Num_update value
		if step % Num_update == 0:
			assign_network_to_target()

		########################################### Noisy Network ###########################################
		# Get y_prediction
		y_batch = []
		Q_batch = output_target.eval(feed_dict = {x_image: observation_next_batch, train_process: True})
		for i in range(len(minibatch)):
			if terminal_batch[i] == True:
				y_batch.append(reward_batch[i])
			else:
				y_batch.append(reward_batch[i] + Gamma * np.max(Q_batch[i]))

		train_step.run(feed_dict = {action_target: action_batch, y_prediction: y_batch, x_image: observation_batch, train_process: True})
		#####################################################################################################

	    # save progress every 10000 iterations
		if step % Num_step_save == 0:
			saver.save(sess, '7_saved_networks_Noisy_DQN/' + game_name)
			print('Model is saved!!!')

	elif step < Num_start_training + Num_training + Num_test:
		# Testing
		progress = 'Testing'

		# Choose the action of testing state
		########################################### Noisy Network ###########################################
		Q_value = output.eval(feed_dict={x_image: [observation_in], train_process: False})
		#####################################################################################################
		action = np.zeros([Num_action])
		action[np.argmax(Q_value)] = 1

		# Get game state
		observation_next, reward, terminal = game_state.frame_step(action)
		observation_next = resize_input(observation_next)

		observation_set.append(observation_next)

		observation_next_in = np.zeros((img_size, img_size, Num_colorChannel * Num_stackFrame))

		# Stack the frame according to the number of skipping frame
		for stack_frame in range(Num_stackFrame):
			observation_next_in[:,:,stack_frame] = observation_set[-1 - (Num_skipFrame * stack_frame)]

		del observation_set[0]

		observation_next_in = np.uint8(observation_next_in)

	else:
		mean_score_test = np.average(test_score)
		print(game_name + str(mean_score_test))
		plt.savefig('./Plot/' + date_time + '_' + algorithm + '_' + game_name + str(mean_score_test) + '.png')

		# Finish the Code
		print('It takes ' + str(time.time() - start_time) + ' seconds to finish this algorithm!')
		break

	# If length of replay memeory is more than the setting value then remove the first one
	if len(Replay_memory) > Num_replay_memory:
		del Replay_memory[0]

	# Save experience to the Replay memory
	if progress != 'Testing':
		Replay_memory.append([observation_in, action, reward, observation_next_in, terminal])

	step += 1
	score += reward

	observation_in = observation_next_in

	# If terminal is True
	if terminal == True:
		# Print informations
		print('step: ' + str(step) + ' / '  + 'episode: ' + str(episode) + ' / ' + 'progress: ' + progress  + ' / '  + 'score: ' + str(score))

		# Add data for plotting
		plot_x.append(episode)
		plot_y.append(score)

		check_plot = 1

		# If progress is testing then add score for calculating test score
		if progress == 'Testing':
			test_score.append(score)

		# Initialize score and add 1 to episode number
		score = 0

		if progress != 'Observing':
			episode += 1

		# Initialize game state
		action = np.zeros([Num_action])
		observation, _, _ = game_state.frame_step(action)
		observation = resize_input(observation)

		observation_set = []

		for i in range(Num_skipFrame * Num_stackFrame):
				observation_set.append(observation)

	if episode % Num_plot_episode == 0 and episode != 0 and check_plot == 1:
		plt.xlabel('Episode')
		plt.ylabel('Score')
		plt.title('Noisy Network DQN')
		plt.grid(True)

		plt.plot(np.average(plot_x), np.average(plot_y), hold = True, marker = '*', ms = 5)
		plt.draw()
		plt.pause(0.000001)

		plot_x = []
		plot_y = []

		check_plot = 0
