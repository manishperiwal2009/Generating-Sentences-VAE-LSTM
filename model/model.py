import os

import numpy as np
import tensorflow as tf

import utils
from datahelper import DataHelper

try:
	from tensorflow.models.rnn import rnn, rnn_cell, seq2seq
except ImportError:
	from tensorflow.python.ops import rnn, rnn_cell, seq2seq

class VAELSTM():
	def __init__(self, args):
		self.args = args
		self.activation = tf.nn.softplus
		self.build_graph()
		self.dataloader = DataHelper("data/input.txt")
		self.dataloader.create_batch()
		if self.args.train:
			self.run_graph()
		else:
			self.test_model()

	def build_graph(self):
		print("Starting building graph.")
		self.input_seq = tf.placeholder(tf.int32, [None, self.args.num_steps])
		# self.embeddings = tf.Variable(tf.constant(0.0, shape=[self.args.vocabulary_size, self.args.embedding_size]),
		# 						trainable=False, name="embeddings")
		with tf.device('/cpu:0'), tf.name_scope("embedding"):
			self.embeddings = tf.get_variable("embeddings", 
								[self.args.vocabulary_size, self.args.embedding_size], dtype=tf.float32)
			encoder_inputs = tf.nn.embedding_lookup(self.embeddings, self.input_seq)
			# self.embedding_placeholder = tf.placeholder(tf.float32, [self.args.vocabulary_size, self.args.embedding_size])
			# self.embedding_init = self.embeddings.assign(self.embedding_placeholder)
			# encoder_inputs = tf.nn.embedding_lookup(self.embeddings, self.input_seq)

		# XT -> num_steps * batch_size * embedding_size
		XT = tf.transpose(encoder_inputs, [1, 0, 2])
		XR = tf.reshape(XT, [-1, self.args.embedding_size])

		# encoder_inputs_split -> list(length) of 2d array
		encoder_inputs_split = tf.split(0, self.args.num_steps, XR)

		# basic LSTM seq2seq model
		cell = tf.nn.rnn_cell.LSTMCell(self.args.num_hidden, state_is_tuple=True)

		if self.args.train:
			cell = tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=0.7, output_keep_prob=0.5)
		else:
			cell = tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=1.0, output_keep_prob=1.0)

		decoCell = cell
		# cell = tf.nn.rnn_cell.MultiRNNCell([cell] * self.args.num_layers, state_is_tuple=True)

		self.output_rnn, self.rnn_encoder_state = tf.nn.rnn(cell, encoder_inputs_split, dtype=tf.float32)
		print(self.rnn_encoder_state)
		XT_new = tf.transpose(self.input_seq, [1, 0])
		XT_new_split = tf.split(0, self.args.num_steps, XT_new)

		W_mu = utils.weight_variable([self.args.n_components_encoder, self.args.num_hidden])
		b_mu = utils.bias_variable([self.args.num_hidden])

		W_log_sigma = utils.weight_variable([self.args.n_components_encoder, self.args.num_hidden])
		b_log_sigma = utils.bias_variable([self.args.num_hidden])

		z_mu = tf.matmul(self.rnn_encoder_state[0], W_mu) + b_mu
		z_log_sigma = 0.5 * (tf.matmul(self.rnn_encoder_state[0], W_log_sigma) + b_log_sigma)

		debug = False

		# Sample from noise distribution p(eps) ~ N(0, 1)
		if debug:
			epsilon = tf.random_normal(
						[dims[0], self.args.num_hidden])

		else:
			epsilon = tf.random_normal(
						tf.pack([tf.shape(self.input_seq)[0], self.args.num_hidden]))

		# Sample from posterior
		z = z_mu + tf.exp(z_log_sigma) * epsilon

		W_dec1 = utils.weight_variable([self.args.num_hidden, self.args.n_components_decoder])
		b_dec1 = utils.bias_variable([self.args.n_components_decoder])
		h_dec1 = self.activation(tf.add(tf.matmul(z, W_dec1), b_dec1))
		h_dec1_tensor_shaped = tf.reshape(h_dec1, [-1, self.args.num_hidden])
		decoder_inputs = ([tf.zeros_like(encoder_inputs_split[0], name="GO")] + encoder_inputs_split[:-1])
		self.decoder_outputs, self.decoder_state = tf.nn.seq2seq.rnn_decoder(decoder_inputs, (h_dec1_tensor_shaped, self.rnn_encoder_state[1]), decoCell)
		
		self.decoder_weights = [tf.ones_like(targets_t, dtype=tf.float32) for targets_t in XT_new_split]

		with tf.variable_scope('softmax'):
			W = tf.get_variable('W', [self.args.num_hidden, self.args.vocabulary_size])
			b = tf.get_variable('b', [self.args.vocabulary_size], initializer=tf.constant_initializer(0.0))

		self.logits = [tf.matmul(dec_output, W) + b for dec_output in self.decoder_outputs]

		# flatten the prediction and target to compute squared error loss(Not used as of now)
		self.y_true = [tf.reshape(encoder_inputs_split_, [-1]) for encoder_inputs_split_ in encoder_inputs_split]
		y_pred = [tf.reshape(dec_output, [-1]) for dec_output in self.decoder_outputs]

		self.loss = 0
		log_px_given_z = 0
		log_px_given_z = tf.nn.seq2seq.sequence_loss(
								self.logits,
								XT_new_split,
								self.decoder_weights,
								self.args.vocabulary_size
							)
		kl_div = -0.5 * tf.reduce_sum(
					1.0 + 2.0 * z_log_sigma - tf.square(z_mu) - tf.exp(2.0 * z_log_sigma),
					1)

		self.loss = tf.reduce_mean(log_px_given_z + kl_div)
		gradients, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tf.trainable_variables()), 5)
		optimizer = tf.train.AdamOptimizer(self.args.learning_rate)
		self.optimizerOp = optimizer.apply_gradients(zip(gradients, tf.trainable_variables()))
		print("Finished building graph.")

	def run_graph(self):
		# Initializing the variables
		init = tf.initialize_all_variables()
		
		saver = tf.train.Saver()

		# Launch the graph
		with tf.Session() as sess:
			sess.run(init)
			# sess.run(self.embedding_init, feed_dict={self.embedding_placeholder: self.dataloader.embeddings})
			if os.path.isfile(self.args.save_dir + self.args.checkpoint_file) and self.args.restore and self.args.train:
				print("Restroing saved checkpoint file.")
				saver.restore(sess, self.args.save_dir + self.args.checkpoint_file)

			for epoch in range(self.args.training_epochs):
				encoder_inputs_x, decoder_inputs_x, target_weights = self.train_batch(self.args.batch_size)
				feed = {self.input_seq: encoder_inputs_x}

				# Fit training using batch data
				_, cost_value = sess.run([self.optimizerOp, self.loss], feed_dict=feed)

				# Display logs per epoch step
				if epoch % self.args.save_every == 0:
					encoder_inputs_x_test, decoder_inputs_x_test, target_weights_test = self.train_batch(1)
					feed2 = {self.input_seq: encoder_inputs_x_test}
					decoder_outputs_new, true_output = sess.run([self.logits, self.y_true], feed_dict=feed2)
					saver.save(sess, self.args.save_dir + self.args.checkpoint_file)
					self.dataloader.similarity(decoder_outputs_new, true_output, decoder_inputs_x_test, self.args.train)
					print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(cost_value))

	def train_batch(self, batch_size):
		encoder_inputs_x, decoder_inputs_x, target_weights = self.dataloader.get_batch(batch_size)
		return encoder_inputs_x, decoder_inputs_x, target_weights

	def test_model(self):

		# Initializing the variables
		init = tf.initialize_all_variables()
		
		saver = tf.train.Saver()

		# Launch the graph
		with tf.Session() as sess:
			sess.run(init)
			# sess.run(self.embedding_init, feed_dict={self.embedding_placeholder: self.dataloader.embeddings})
			if os.path.isfile(self.args.save_dir + self.args.checkpoint_file) and self.args.restore:
				print("Restroing saved checkpoint file.")
				saver.restore(sess, self.args.save_dir + self.args.checkpoint_file)

			while True:
				sentence = raw_input("Input Sentence:\n")
				tokens = self.dataloader.sent2tokens(sentence)
				feed = {self.input_seq: tokens}
				decoder_outputs_new, true_output = sess.run([self.logits, self.y_true], feed_dict=feed)
				decoder_inputs_x_test = sentence
				self.dataloader.similarity(decoder_outputs_new, true_output, decoder_inputs_x_test, self.args.train)

