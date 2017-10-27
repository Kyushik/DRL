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

algorithm = 'DRQN'

Num_action = game.Return_Num_Action()
game_name = game.ReturnName()

Gamma = Deep_Parameters.Gamma
Learning_rate = Deep_Parameters.Learning_rate
Epsilon = Deep_Parameters.Epsilon
Final_epsilon = Deep_Parameters.Final_epsilon

Num_replay_episode = 500
Num_start_training = Deep_Parameters.Num_start_training
Num_training = Deep_Parameters.Num_training
Num_update = Deep_Parameters.Num_update
Num_batch = 32
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

step_size = 6
lstm_size = 400
flatten_size = 10*10*64

first_conv   = [8,8,Num_colorChannel,32]
second_conv  = Deep_Parameters.second_conv
third_conv   = Deep_Parameters.third_conv
first_dense  = [lstm_size, Num_action]

# If is train is false then immediately start testing
if Is_train == False:
	Num_start_training = 0
	Num_training = 0
	check_save = 1
else:
	check_save = input('Is there any saved data?(1=y/2=n): ')

# Initialize weights and bias
def weight_variable(shape):
    return tf.Variable(xavier_initializer(shape))

def bias_variable(shape):
	return tf.Variable(xavier_initializer(shape))

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
		observation_out = np.reshape(observation_out, (img_size, img_size, 1))

	observation_out = np.uint8(observation_out)
	return observation_out

# Input
x_image = tf.placeholder(tf.float32, shape = [None, img_size, img_size, Num_colorChannel])
x_normalize = (x_image - (255.0/2)) / (255.0/2)

with tf.variable_scope('network'):
	# Convolution variables
	w_conv1 = weight_variable(first_conv)
	b_conv1 = bias_variable([first_conv[3]])

	w_conv2 = weight_variable(second_conv)
	b_conv2 = bias_variable([second_conv[3]])

	w_conv3 = weight_variable(third_conv)
	b_conv3 = bias_variable([third_conv[3]])

	# Densely connect layer variables
	w_fc1 = weight_variable(first_dense)
	b_fc1 = bias_variable([first_dense[1]])

	# Network
	h_conv1 = tf.nn.relu(conv2d(x_normalize, w_conv1, 4) + b_conv1)
	h_conv2 = tf.nn.relu(conv2d(h_conv1, w_conv2, 2) + b_conv2)
	h_conv3 = tf.nn.relu(conv2d(h_conv2, w_conv3, 1) + b_conv3)

	rnn_batch_size = tf.placeholder(dtype = tf.int32)
	rnn_step_size  = tf.placeholder(dtype = tf.int32)

	convFlat = tf.reshape(h_conv3,[rnn_batch_size, rnn_step_size , flatten_size])

	cell = tf.contrib.rnn.BasicLSTMCell(num_units = lstm_size, state_is_tuple = True)
	rnn_out, rnn_state = tf.nn.dynamic_rnn(inputs = convFlat, cell = cell, dtype = tf.float32)

# Vectorization
rnn_out = rnn_out[:, -1, :]
rnn_out =tf.reshape(rnn_out, shape = [rnn_batch_size, -1])

output = tf.matmul(rnn_out, w_fc1) + b_fc1

with tf.variable_scope('target'):
	# Convolution variables target
	w_conv1_target = weight_variable(first_conv)
	b_conv1_target = bias_variable([first_conv[3]])

	w_conv2_target = weight_variable(second_conv)
	b_conv2_target = bias_variable([second_conv[3]])

	w_conv3_target = weight_variable(third_conv)
	b_conv3_target = bias_variable([third_conv[3]])

	# Densely connect layer variables target
	w_fc1_target = weight_variable(first_dense)
	b_fc1_target = bias_variable([first_dense[1]])

	# Target Network
	h_conv1_target = tf.nn.relu(conv2d(x_normalize, w_conv1_target, 4) + b_conv1_target)
	h_conv2_target = tf.nn.relu(conv2d(h_conv1_target, w_conv2_target, 2) + b_conv2_target)
	h_conv3_target = tf.nn.relu(conv2d(h_conv2_target, w_conv3_target, 1) + b_conv3_target)

	convFlat_target = tf.reshape(h_conv3_target,[rnn_batch_size, rnn_step_size , flatten_size])

	cell_target = tf.contrib.rnn.BasicLSTMCell(num_units = lstm_size)
	rnn_out_target, rnn_state_target = tf.nn.dynamic_rnn(inputs = convFlat_target, cell = cell_target, dtype = tf.float32)

# Vectorization
rnn_out_target = rnn_out_target[:, -1, :]
rnn_out_target = tf.reshape(rnn_out_target, shape = [rnn_batch_size , -1])

output_target = tf.matmul(rnn_out_target, w_fc1_target) + b_fc1_target

# Loss function and Train
action_target = tf.placeholder(tf.float32, shape = [None, Num_action])
y_prediction = tf.placeholder(tf.float32, shape = [None])

y_target = tf.reduce_sum(tf.multiply(output, action_target), reduction_indices = 1)
Loss = tf.reduce_mean(tf.square(y_prediction - y_target))
train_step = tf.train.AdamOptimizer(learning_rate = Learning_rate, epsilon = 0.0001).minimize(Loss)

# Initialize variables
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = GPU_fraction

sess = tf.InteractiveSession(config=config)
init = tf.global_variables_initializer()
sess.run(init)

# Load the file if the saved file exists
saver = tf.train.Saver()
if check_save == 1:
    checkpoint = tf.train.get_checkpoint_state("6_saved_networks_DRQN")
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

episode_memory = []
observation_set = []

game_state = game.GameState()
action = np.zeros([Num_action])
observation, _, _ = game_state.frame_step(action)
observation = resize_input(observation)

# Initialize observation set
for i in range(step_size):
	observation_set.append(observation)

start_time = time.time()

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

	elif step <= Num_start_training + Num_training:
		# Training
		progress = 'Training'

		# if random value(0 - 1) is smaller than Epsilon, action is random. Otherwise, action is the one which has the largest Q value
		if random.random() < Epsilon:
			action = np.zeros([Num_action])
			action[random.randint(0, Num_action - 1)] = 1
		else:
			Q_value = output.eval(feed_dict={x_image: observation_set, rnn_batch_size: 1, rnn_step_size: step_size})[0]
			action = np.zeros([Num_action])
			action[np.argmax(Q_value)] = 1

		observation_next, reward, terminal = game_state.frame_step(action)
		observation_next = resize_input(observation_next)

		# Decrease the epsilon value
		if Epsilon > Final_epsilon:
			Epsilon -= 1.0/Num_training

		# Select minibatch
		episode_batch = random.sample(Replay_memory, Num_batch)

		minibatch = []
		batch_end_index = []
		count_minibatch = 0

		for episode_ in episode_batch:
			episode_start = np.random.randint(0, len(episode_) + 1 - step_size)
			for step_ in range(step_size):
				minibatch.append(episode_[episode_start + step_])
				if step_ == step_size - 1:
					batch_end_index.append(count_minibatch)

				count_minibatch += 1

		# Save the each batch data
		observation_batch      = [batch[0] for batch in minibatch]
		action_batch           = [batch[1] for batch in minibatch]
		reward_batch           = [batch[2] for batch in minibatch]
		observation_next_batch = [batch[3] for batch in minibatch]
		terminal_batch 	       = [batch[4] for batch in minibatch]

		# Update target network according to the Num_update value
		if step % Num_update == 0:
			assign_network_to_target()

		# Get y_prediction
		y_batch = []
		action_in = []

		Q_batch = output_target.eval(feed_dict = {x_image: observation_next_batch, rnn_batch_size: Num_batch, rnn_step_size: step_size})

		for count, i in enumerate(batch_end_index):
			action_in.append(action_batch[i])
			if terminal_batch[i] == True:
				y_batch.append(reward_batch[i])
			else:
				y_batch.append(reward_batch[i] + Gamma * np.max(Q_batch[count]))

		train_step.run(feed_dict = {action_target: action_in, y_prediction: y_batch, x_image: observation_batch, rnn_batch_size: Num_batch, rnn_step_size: step_size})

	    # save progress every 10000 iterations
		if step % Num_step_save == 0:
			saver.save(sess, '6_saved_networks_DRQN/' + game_name)
			print('Model is saved!!!')

	elif step < Num_start_training + Num_training + Num_test:
		# Testing
		progress = 'Testing'
		Epsilon = 0

		# Choose the action of testing state
		Q_value = output.eval(feed_dict={x_image: observation_set, rnn_batch_size: 1, rnn_step_size: step_size})[0]
		action = np.zeros([Num_action])
		action[np.argmax(Q_value)] = 1

		# Get game state
		observation_next, reward, terminal = game_state.frame_step(action)
		observation_next = resize_input(observation_next)

	else:
		mean_score_test = np.average(test_score)
		print(game_name + str(mean_score_test))
		plt.savefig('./Plot/' + date_time + '_' + algorithm + '_' + game_name + str(mean_score_test) + '.png')

		# Finish the Code
		print('It takes ' + str(time.time() - start_time) + ' seconds to finish this algorithm!')
		break

	# If length of replay memeory is more than the setting value then remove the first one
	if len(Replay_memory) > Num_replay_episode:
		del Replay_memory[0]

	# Save experience to the Replay memory
	if progress != 'Testing':
		# Save experience to the Replay memory
		episode_memory.append([observation, action, reward, observation_next, terminal])

	step += 1
	score += reward

	observation = observation_next
	observation_set.append(observation)

	if len(observation_set) > step_size:
		del observation_set[0]

	# If terminal is True
	if terminal == True:
		# Print informations
		print('step: ' + str(step) + ' / '  + 'episode: ' + str(episode) + ' / ' + 'progress: ' + progress  + ' / '  + 'epsilon: ' + str(Epsilon) + ' / '  + 'score: ' + str(score))

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

		if len(episode_memory) > step_size:
			Replay_memory.append(episode_memory)
		episode_memory = []

		# Initialize game state
		action = np.zeros([Num_action])
		observation, _, _ = game_state.frame_step(action)
		observation = resize_input(observation)

		observation_set = []
		for i in range(step_size):
			observation_set.append(observation)

	if episode % Num_plot_episode == 0 and episode != 0 and check_plot == 1:
		plt.xlabel('Episode')
		plt.ylabel('Score')
		plt.title('Deep Recurrent Q Network')
		plt.grid(True)

		plt.plot(np.average(plot_x), np.average(plot_y), hold = True, marker = '*', ms = 5)
		plt.draw()
		plt.pause(0.000001)

		plot_x = []
		plot_y = []

		check_plot = 0
