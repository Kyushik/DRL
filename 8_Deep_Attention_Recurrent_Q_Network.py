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

algorithm = 'DARQN'

Num_action = game.Return_Num_Action()
game_name = game.ReturnName()

Gamma = Deep_Parameters.Gamma
Learning_rate = Deep_Parameters.Learning_rate
Epsilon = Deep_Parameters.Epsilon
Final_epsilon = Deep_Parameters.Final_epsilon

Num_replay_episode = 500
Num_start_training = Deep_Parameters.Num_start_training
# Num_start_training = 5000

Num_training = Deep_Parameters.Num_training
Num_update = Deep_Parameters.Num_update
Num_batch = 32
Num_test = Deep_Parameters.Num_test
Num_skipFrame = Deep_Parameters.Num_skipFrame
Num_stackFrame = Deep_Parameters.Num_stackFrame
Num_colorChannel = Deep_Parameters.Num_colorChannel

Num_plot_episode = Deep_Parameters.Num_plot_episode
# Num_plot_episode = 2
Num_step_save = Deep_Parameters.Num_step_save

GPU_fraction = Deep_Parameters.GPU_fraction
Is_train = Deep_Parameters.Is_train

# Parametwrs for Network
img_size = Deep_Parameters.img_size

step_size = 4
lstm_size = 256
flatten_size = 5 * 5 * 32

first_conv   = [5,5,Num_colorChannel,8]
second_conv  = [3,3,8,16]
third_conv   = [2,2,16,32]
first_dense  = [lstm_size, Num_action]

# parameter for attention
img_fraction_size = 40
stride = 20

len_horizontal = int((img_size - img_fraction_size) / stride + 1)
len_vertical   = int((img_size - img_fraction_size) / stride + 1)
len_stack = len_horizontal * len_vertical

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
		observation_out = np.reshape(observation_out, (img_size, img_size))

	# Making fractions
	# Initialize fraction of test images and heatmap
	img_fraction = np.zeros([img_fraction_size, img_fraction_size, len_stack])

	index_fraction = 0
	for m in range(len_vertical):
		start_v = stride * m
		for n in range(len_horizontal):
			start_h = stride * n

			img_fraction[:,:,index_fraction] = observation_out[start_v : start_v + img_fraction_size, start_h : start_h + img_fraction_size]

			index_fraction += 1

	img_fraction = np.uint8(img_fraction)

	return observation_out, img_fraction

# LSTM function
def LSTM_cell(C_prev, h_prev, x_lstm, Wf, Wi, Wc, Wo, bf, bi, bc, bo):
    # C_prev: Cell state from lstm of previous time step (shape: [batch_size, lstm_size])
    # h_prev: output from lstm of previous time step (shape: [batch_size, lstm_size])
    # x_lstm: input of lstm (shape: [batch_size, data_flatten_size])

    input_concat = tf.concat([x_lstm, h_prev], 1)
    f = tf.sigmoid(tf.matmul(input_concat, Wf) + bf)
    i = tf.sigmoid(tf.matmul(input_concat, Wi) + bi)
    c = tf.tanh(tf.matmul(input_concat, Wc) + bc)
    o = tf.sigmoid(tf.matmul(input_concat, Wo) + bo)

    C_t = tf.multiply(f, C_prev) + tf.multiply(i, c)
    h_t = tf.multiply(o, tf.tanh(C_t))

    return C_t, h_t # Cell state, Output

# Soft Attention function
def soft_attention(h_prev, a, Wa, Wh):
    # h_prev: output from lstm of previous time step (shape: [batch_size, lstm_size])
    # a: Image windows after CNN. List of convolution window images
    # (List len: number of windows, element shape: [batch_size, convolution flatten size])

	m_list = [tf.tanh(tf.matmul(a[i], Wa) + tf.matmul(h_prev, Wh)) for i in range(len(a))]
	m_concat = tf.concat([m_list[i] for i in range(len(a))], axis = 1)
	alpha = tf.nn.softmax(m_concat)
	z_list = [tf.multiply(a[i], tf.slice(alpha, (0, i), (-1, 1))) for i in range(len(a))]
	z_stack = tf.stack(z_list, axis = 2)
	z = tf.reduce_sum(z_stack, axis = 2)

	return alpha, z

# Input
x_image = tf.placeholder(tf.float32, shape = [None, img_fraction_size, img_fraction_size, len_stack])
x_normalize = (x_image - (255.0/2)) / (255.0/2)

x_unstack = tf.unstack(x_normalize, axis = 3)
rnn_batch_size = tf.cast(tf.shape(x_image)[0] / step_size, tf.int32)

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

	conv_list = []
	for i in range(len_stack):
		x_conv = tf.reshape(x_unstack[i], (-1, img_fraction_size, img_fraction_size, 1))
		conv1 = tf.nn.relu(conv2d(x_conv, w_conv1, 2) + b_conv1)
		conv2 = tf.nn.relu(conv2d(conv1, w_conv2, 2) + b_conv2)
		conv3 = tf.nn.relu(conv2d(conv2, w_conv3, 2) + b_conv3)
		conv_result_flat = tf.reshape(conv3,[-1, step_size , flatten_size])
		conv_list.append(conv_result_flat)

	# len_conv = width * height * num_conv_feature_map
	len_conv = flatten_size
	conv_stack = tf.stack(conv_list, axis = 3)
	conv_unstack_step = tf.unstack(conv_stack, axis = 1)

	#LSTM Variables
	Wf = weight_variable([len_conv + lstm_size, lstm_size])
	Wi = weight_variable([len_conv + lstm_size, lstm_size])
	Wc = weight_variable([len_conv + lstm_size, lstm_size])
	Wo = weight_variable([len_conv + lstm_size, lstm_size])

	bf = bias_variable([lstm_size])
	bi = bias_variable([lstm_size])
	bc = bias_variable([lstm_size])
	bo = bias_variable([lstm_size])

	# Attention Variables
	Wa = weight_variable([len_conv, 1])
	Wh = weight_variable([lstm_size, 1])

# Initial lstm cell state and output
rnn_state = tf.zeros([rnn_batch_size, lstm_size], tf.float32)
rnn_out = tf.zeros([rnn_batch_size, lstm_size], tf.float32)

#################################### Attention!!! ####################################
for i in range(step_size):
	att_input = tf.unstack(conv_unstack_step[i], axis = 2)
	alpha, z = soft_attention(rnn_out, att_input, Wa, Wh)
	rnn_state, rnn_out = LSTM_cell(rnn_state, rnn_out, z, Wf, Wi, Wc, Wo, bf, bi, bc, bo)
######################################################################################

output = tf.matmul(rnn_out, w_fc1)+b_fc1

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

	conv_list_target = []
	for i in range(len_stack):
		x_conv_target = tf.reshape(x_unstack[i], (-1, img_fraction_size, img_fraction_size, 1))
		conv1_target = tf.nn.relu(conv2d(x_conv_target, w_conv1_target, 2) + b_conv1_target)
		conv2_target = tf.nn.relu(conv2d(conv1_target, w_conv2_target, 2) + b_conv2_target)
		conv3_target = tf.nn.relu(conv2d(conv2_target, w_conv3_target, 2) + b_conv3_target)
		conv_result_flat_target = tf.reshape(conv3_target,[-1, step_size , flatten_size])
		conv_list_target.append(conv_result_flat_target)

	len_conv_target = flatten_size
	conv_stack_target = tf.stack(conv_list_target, axis = 3)
	conv_unstack_step_target = tf.unstack(conv_stack_target, axis = 1)

	#LSTM Variables
	Wf_target = weight_variable([len_conv_target + lstm_size, lstm_size])
	Wi_target = weight_variable([len_conv_target + lstm_size, lstm_size])
	Wc_target = weight_variable([len_conv_target + lstm_size, lstm_size])
	Wo_target = weight_variable([len_conv_target + lstm_size, lstm_size])

	bf_target = bias_variable([lstm_size])
	bi_target = bias_variable([lstm_size])
	bc_target = bias_variable([lstm_size])
	bo_target = bias_variable([lstm_size])

	# Attention Variables
	Wa_target = weight_variable([len_conv_target, 1])
	Wh_target = weight_variable([lstm_size, 1])

# Initial lstm cell state and output
rnn_state_target = tf.zeros([rnn_batch_size, lstm_size], tf.float32)
rnn_out_target = tf.zeros([rnn_batch_size, lstm_size], tf.float32)

#################################### Attention!!! ####################################
for i in range(step_size):
	att_input_target = tf.unstack(conv_unstack_step_target[i], axis = 2)
	alpha_target, z_target = soft_attention(rnn_out_target, att_input_target, Wa_target, Wh_target)
	rnn_state_target, rnn_out_target = LSTM_cell(rnn_state_target, rnn_out_target, z_target, Wf_target, Wi_target, Wc_target, Wo_target, bf_target, bi_target, bc_target, bo_target)
######################################################################################

output_target = tf.matmul(rnn_out_target, w_fc1_target)+b_fc1_target

# Loss function and Train
action_target = tf.placeholder(tf.float32, shape = [None, Num_action])
y_prediction = tf.placeholder(tf.float32, shape = [None])

y_target = tf.reduce_sum(tf.multiply(output, action_target), reduction_indices = 1)
Loss = tf.reduce_mean(tf.square(y_prediction - y_target))
train_step = tf.train.AdamOptimizer(learning_rate = Learning_rate, epsilon = 1e-2 / Num_batch).minimize(Loss)

# Initialize variables
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = GPU_fraction

sess = tf.InteractiveSession(config=config)
init = tf.global_variables_initializer()
sess.run(init)

# Load the file if the saved file exists
saver = tf.train.Saver()
if check_save == 1:
    checkpoint = tf.train.get_checkpoint_state("8_saved_networks_DARQN")
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
obs_original, observation = resize_input(observation)

# Initialize observation set
for i in range(step_size):
	observation_set.append(observation)

start_time = time.time()

# Figure and figure data setting
plt.figure(1)
plot_x = []
plot_y = []

test_score = []

plot_y_maxQ = []
maxQ_list = []

check_plot = 0

f, ax = plt.subplots(2,2, sharex=False)

# Training & Testing
while True:
	if step <= Num_start_training:
		# Observation
		progress = 'Observing'

		action = np.zeros([Num_action])
		action[random.randint(0, Num_action - 1)] = 1.0

		observation_next, reward, terminal = game_state.frame_step(action)
		obs_original, observation_next = resize_input(observation_next)

	elif step <= Num_start_training + Num_training:
		# Training
		progress = 'Training'

		# if random value(0 - 1) is smaller than Epsilon, action is random. Otherwise, action is the one which has the largest Q value
		if random.random() < Epsilon:
			action = np.zeros([Num_action])
			action[random.randint(0, Num_action - 1)] = 1
		else:
			Q_value = output.eval(feed_dict={x_image: observation_set})[0]
			action = np.zeros([Num_action])
			action[np.argmax(Q_value)] = 1

		observation_next, reward, terminal = game_state.frame_step(action)
		obs_original, observation_next = resize_input(observation_next)

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


		Q_batch = output_target.eval(feed_dict = {x_image: observation_next_batch})

		for count, i in enumerate(batch_end_index):
			action_in.append(action_batch[i])
			if terminal_batch[i] == True:
				y_batch.append(reward_batch[i])
			else:
				y_batch.append(reward_batch[i] + Gamma * np.max(Q_batch[count]))

		train_step.run(feed_dict = {action_target: action_in, y_prediction: y_batch, x_image: observation_batch})

		maxQ_list.append(np.max(Q_batch))

	    #save progress every 10000 iterations
		if step % Num_step_save == 0:
			saver.save(sess, '8_saved_networks_DARQN/' + game_name)
			print('Model is saved!!!')

	elif step < Num_start_training + Num_training + Num_test:
		# Testing
		progress = 'Testing'
		Epsilon = 0

		# Choose the action of testing state
		Q_value = output.eval(feed_dict={x_image: observation_set})[0]
		action = np.zeros([Num_action])
		action[np.argmax(Q_value)] = 1

		# Get game state
		observation_next, reward, terminal = game_state.frame_step(action)
		obs_original, observation_next = resize_input(observation_next)

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

	if step % 1000 == 0:
		obs_plot = obs_original
		obs_set_plot = observation_set

	if len(observation_set) > step_size:
		del observation_set[0]

	# If terminal is True
	if terminal == True:
		# Print informations
		print('step: ' + str(step) + ' / '  + 'episode: ' + str(episode) + ' / ' + 'progress: ' + progress  + ' / '  + 'epsilon: ' + str(Epsilon) + ' / '  + 'score: ' + str(score))

		# Add data for plotting
		plot_x.append(episode)
		plot_y.append(score)

		plot_y_maxQ.append(np.mean(maxQ_list))

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
		_, observation = resize_input(observation)

		observation_set = []
		for i in range(step_size):
			observation_set.append(observation)

	if episode % Num_plot_episode == 0 and episode != 0 and check_plot == 1:
		plt.xlabel('Episode')
		plt.ylabel('Score')
		plt.title('Deep Attention Recurrent Q Network')
		plt.grid(True)

		alpha_ = sess.run(alpha, feed_dict = {x_image: obs_set_plot})[0]
		obs_original = np.reshape(obs_original, (img_size, img_size))
		alpha_reshape = np.reshape(alpha_, (len_vertical, len_horizontal))

		ax[0,0].plot(np.average(plot_x), np.average(plot_y),'*')
		ax[0,0].set_title('Score')
		ax[0,0].set_ylabel('Score')
		ax[0,0].hold(True)

		ax[0,1].plot(np.average(plot_x), np.average(plot_y_maxQ),'*')
		ax[0,1].set_title('Max Q')
		ax[0,1].set_ylabel('Max Q')
		ax[0,1].hold(True)

		ax[1,0].clear()
		ax[1,0].imshow(obs_plot, cmap = 'gray')
		ax[1,0].set_title('Original Image')
		# ax[1,0].hold(True)

		ax[1,1].clear()
		ax[1,1].imshow(alpha_reshape, cmap = 'gray')
		ax[1,1].set_title('Attention')
		# ax[1,1].set_xlabel('Episode')
		# ax[1,1].hold(True)

		plt.draw()
		plt.pause(0.000001)

		plot_x = []
		plot_y = []

		plot_y_maxQ = []

		check_plot = 0
                #
				# ax[0,0].plot(np.average(plot_x), np.average(plot_y_loss), '*')
				# ax[0,0].set_title('Categorical DQN C51')
				# ax[0,0].set_ylabel('Mean Loss')
				# ax[0,0].hold(True)
                #
				# ax[1,0].plot(np.average(plot_x), np.average(plot_y),'*')
				# ax[1,0].set_ylabel('Mean score')
				# ax[1,0].hold(True)
                #
				# ax[2,0].plot(np.average(plot_x), np.average(plot_y_maxQ),'*')
				# ax[2,0].set_ylabel('Mean Max Q')
				# ax[2,0].set_xlabel('Episode')
				# ax[2,0].hold(True)
