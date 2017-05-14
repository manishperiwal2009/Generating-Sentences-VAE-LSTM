from __future__ import unicode_literals

import json
import re
import pickle
import os.path
import random
import sys

reload(sys)  
sys.setdefaultencoding('utf8')

import numpy as np
from spacy.en import English
parser = English()

def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\'", " \' ", string)
    string = re.sub(r"]", " ] ", string)
    string = re.sub(r"\(", " ( ", string)
    string = re.sub(r"\)", " ) ", string)
    string = re.sub(r"\?", " ? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

class DataHelper(object):
	def __init__(self, filename):
		self.dictionary = {"_PAD": 0, "_GO": 1, "_EOS": 2, "_UNK": 3}
		self.filename = filename
		self.trainingset = []
		self.encoder_size, self.decoder_size = (10, 10)

		if os.path.isfile("data/reverse_dictionary.txt"):
			reverse_dictionaryfile = open("data/reverse_dictionary.txt", "r+")
			self.reverse_dictionary = json.load(reverse_dictionaryfile)
			print("Reverse Dictionary already exists")
		if os.path.isfile("data/dictionary.txt"):
			dictionaryfile = open("data/dictionary.txt", "r+")
			self.dictionary = json.load(dictionaryfile)
			print("Dictionary already exists")
		else:
			print("Creating Dictionary")
			self.create_dictionary(filename)

	def create_dictionary(self, filename):
		file_connection = open(filename, "r+")
		for line in file_connection.readlines():
			format_line = clean_str(line)
			for word in format_line.split():
				if word not in self.dictionary:
					self.dictionary[word] = len(self.dictionary.keys())

		self.reverse_dictionary = {v: k for k, v in self.dictionary.iteritems()}
		dictionaryfile = open("data/dictionary.txt", "w+")
		reverse_dictionaryfile = open("data/reverse_dictionary.txt", "w+")
		json.dump(self.dictionary, dictionaryfile)
		json.dump(self.reverse_dictionary, reverse_dictionaryfile)

	def sent2tokens(self, sentence):
		batch_size = 1
		tokens = []
		batch_encoder_inputs = []
		sentence = clean_str(sentence)
		for word in sentence.split():
			if word in self.dictionary:
				tokens.append(self.dictionary[word])
			else:
				tokens.append(self.dictionary["_UNK"])
		tokens_padded = [self.dictionary["_PAD"]] * (self.encoder_size - len(tokens))
		encoder_inputs = [tokens + tokens_padded]
		return encoder_inputs

	def extract_sents(self, parsedData):
		sents = []
			# the "sents" property returns spans
			# spans have indices into the original string
			# where each index value represents a token
		for span in parsedData.sents:
			# go from the start to the end of each span, returning each token in the sentence
			# combine each token using join()
			sent = ''.join(parsedData[i].string for i in range(span.start, span.end)).strip()
			sents.append(sent)
		return sents

	def create_batch(self):
		file_connection = open(self.filename, "r+")
		all_lines = file_connection.readlines()
		if os.path.isfile("data/trainingset.pkl"):
			with open('data/trainingset.pkl', 'rb') as in_file:
				print("Loading from training pickle file")
				self.trainingset = pickle.load(in_file)
				in_file.close()
				return
		self.trainingset = []
		for index, line in enumerate(all_lines):
			if index >= len(all_lines) - 1:
				break
			# source_line = parser(line.decode('utf-8'))
			# target_line = parser(line.decode('utf-8'))
			source_line = parser(line.decode('windows-1252'))
			target_line = parser(line.decode('windows-1252'))

			# Let's look at the sentences
			source_line_sent = self.extract_sents(source_line)
			target_line_sent = self.extract_sents(target_line)
			source_line_sent[-1] = source_line_sent[-1].split(":")[-1]
			target_line_sent[-1] = target_line_sent[-1].split(":")[-1]
			source_line_sent[-1] = clean_str(source_line_sent[-1])
			target_line_sent[-1] = clean_str(target_line_sent[-1])
			source_line_words = []
			target_line_words = []

			for word in source_line_sent[-1].split():
				if word in self.dictionary:
					source_line_words.append(self.dictionary[word])
					target_line_words.append(self.dictionary[word])
				else:
					source_line_words.append(self.dictionary["_UNK"])
					target_line_words.append(self.dictionary["_UNK"])

			if len(source_line_words) < self.encoder_size and len(source_line_words) > 5:
				encoder_pad = [self.dictionary["_PAD"]] * (self.encoder_size - len(source_line_words))
				decoder_pad = [self.dictionary["_PAD"]] * (self.encoder_size - len(target_line_words))
				self.trainingset.append([source_line_words + encoder_pad, target_line_words + decoder_pad])
		with open('data/trainingset.pkl', 'wb') as out_file:
			pickle.dump(self.trainingset, out_file)
			out_file.close()

	def get_batch(self, batch_size):
		#just a hack
		
		encoder_inputs, decoder_inputs = [], []

		# Get a random batch of encoder and decoder inputs from data,
		for _ in xrange(batch_size):
			encoder_input, decoder_input = random.choice(self.trainingset)
			encoder_inputs.append(encoder_input)
			decoder_inputs.append(decoder_input)

			# Now we create batch-major vectors from the data selected above.
		batch_encoder_inputs, batch_decoder_inputs, batch_weights = [], [], []
		# Batch encoder inputs are just re-indexed encoder_inputs.
		for length_idx in xrange(self.encoder_size):
			batch_encoder_inputs.append(
				np.array([encoder_inputs[batch_idx][length_idx]
					for batch_idx in xrange(batch_size)], dtype=np.float32))

		# Create target_weights to be 0 for targets that are padding.
		for length_idx in xrange(self.decoder_size):
			batch_weight = np.ones(batch_size, dtype=np.float32)
			for batch_idx in xrange(batch_size):
				# We set weight to 0 if the corresponding target is a PAD symbol.
				# The corresponding target is decoder_input shifted by 1 forward.
				if length_idx < self.decoder_size - 1:
					target = encoder_inputs[batch_idx][length_idx + 1]
				if length_idx == self.decoder_size - 1 or target == 0:
					batch_weight[batch_idx] = 0.0
				batch_weights.append(batch_weight)
			# print("Batch creation completed")
		return encoder_inputs, decoder_inputs, batch_weights

	def similarity(self, y_pred, y_true, decoder_input, train):
		if train and isinstance(decoder_input, list):
			sentence = [self.reverse_dictionary[str(word)] for word in decoder_input[0]]
			print("Input Sentence", " ".join(sentence))
		else:
			print("Input Sentence", decoder_input)

		outputs = []
		outputs = [int(np.argmax(logit, axis=1)) for logit in y_pred]
		# for out in y_pred:
		# 	outputs.append(np.argmax(out))
		output_sentence = [self.reverse_dictionary[str(word)] for word in outputs[:20]]
		print("Output Sentence", " ".join(output_sentence))