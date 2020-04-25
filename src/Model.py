from __future__ import division
from __future__ import print_function

import sys
import numpy as np
import tensorflow as b
#import backend as b
import os


class DecoderType:
	BestPath = 0


class Model:
	#model
	batchSize = 50
	imgSize = (128, 32)
	maxTextLen = 32

	def __init__(self, charList, decoderType=DecoderType.BestPath, mustRestore=False, dump=False):
		self.dump = dump
		self.charList = charList
		self.decoderType = decoderType
		self.mustRestore = mustRestore
		self.snapID = 0

		#normalisasi dari batch
		self.is_train = b.placeholder(b.bool, name='is_train')

		#input image batch
		self.inputImgs = b.placeholder(b.float32, shape=(None, Model.imgSize[0], Model.imgSize[1]))

		#setup CNN, RNN and CTC
		self.setupCNN()
		self.setupRNN()
		self.setupCTC()

		#setup optimizer to train NN
		self.batchesTrained = 0
		self.learningRate = b.placeholder(b.float32, shape=[])
		self.update_ops = b.get_collection(b.GraphKeys.UPDATE_OPS) 
		with b.control_dependencies(self.update_ops):
			self.optimizer = b.train.RMSPropOptimizer(self.learningRate).minimize(self.loss)

		#inisialisasi backend
		(self.sess, self.saver) = self.setupb()

			
	def setupCNN(self):
		cnnIn4d = b.expand_dims(input=self.inputImgs, axis=3)

		#parameter
		kernelVals = [5, 5, 3, 3, 3]
		featureVals = [1, 32, 64, 128, 128, 256]
		strideVals = poolVals = [(2,2), (2,2), (1,2), (1,2), (1,2)]
		numLayers = len(strideVals)

		#create layers
		#layer pertama
		pool = cnnIn4d
		for i in range(numLayers):
			kernel = b.Variable(b.truncated_normal([kernelVals[i], kernelVals[i], featureVals[i], featureVals[i + 1]], stddev=0.1))
			conv = b.nn.conv2d(pool, kernel, padding='SAME',  strides=(1,1,1,1))
			conv_norm = b.layers.batch_normalization(conv, training=self.is_train)
			relu = b.nn.relu(conv_norm)
			pool = b.nn.max_pool(relu, (1, poolVals[i][0], poolVals[i][1], 1), (1, strideVals[i][0], strideVals[i][1], 1), 'VALID')

		self.cnnOut4d = pool


	def setupRNN(self):
		rnnIn3d = b.squeeze(self.cnnOut4d, axis=[2])

		numHidden = 256
		cells = [b.rnn.LSTMCell(num_units=numHidden, state_is_tuple=True) for _ in range(2)] # 2 layers

		#stack cell
		stacked = b.rnn.MultiRNNCell(cells, state_is_tuple=True)

		#bidirectional RNN
		#BxTxF -> BxTx2H
		((fw, bw), _) = b.nn.bidirectional_dynamic_rnn(cell_fw=stacked, cell_bw=stacked, inputs=rnnIn3d, dtype=rnnIn3d.dtype)
									
		#BxTxH + BxTxH -> BxTx2H -> BxTx1X2H
		concat = b.expand_dims(b.concat([fw, bw], 2), 2)
									
		#output to char inc blank BxTx1x2H -> BxTx1xC -> BxTxC
		kernel = b.Variable(b.truncated_normal([1, 1, numHidden * 2, len(self.charList) + 1], stddev=0.1))
		self.rnnOut3d = b.squeeze(b.nn.atrous_conv2d(value=concat, filters=kernel, rate=1, padding='SAME'), axis=[2])
		

	def setupCTC(self):
		#BxTxC -> TxBxC
		self.ctcIn3dTBC = b.transpose(self.rnnOut3d, [1, 0, 2])
		#ground truth text as sparse tensor
		self.gtTexts = b.SparseTensor(b.placeholder(b.int64, shape=[None, 2]) , b.placeholder(b.int32, [None]), b.placeholder(b.int64, [2]))

		#calc loss batch
		self.seqLen = b.placeholder(b.int32, [None])
		self.loss = b.reduce_mean(b.nn.ctc_loss(labels=self.gtTexts, inputs=self.ctcIn3dTBC, sequence_length=self.seqLen, ctc_merge_repeated=True))

		#calc loss prob
		self.savedCtcInput = b.placeholder(b.float32, shape=[Model.maxTextLen, None, len(self.charList) + 1])
		self.lossPerElement = b.nn.ctc_loss(labels=self.gtTexts, inputs=self.savedCtcInput, sequence_length=self.seqLen, ctc_merge_repeated=True)

		# decoder: either best path decoding or beam search decoding
		if self.decoderType == DecoderType.BestPath:
			self.decoder = b.nn.ctc_greedy_decoder(inputs=self.ctcIn3dTBC, sequence_length=self.seqLen)
			chars = str().join(self.charList)
			wordChars = open('../model/wordCharList.txt').read().splitlines()[0]
			corpus = open('../data/corpus.txt').read()


	def setupb(self):
		sess=b.Session()

		saver = b.train.Saver(max_to_keep=1) # saver saves model to file
		modelDir = '../model/'
		latestSnapshot = b.train.latest_checkpoint(modelDir)
		if latestSnapshot:
			print('Snap Terakhir: ' + latestSnapshot)
			saver.restore(sess, latestSnapshot)
		else:
			print('Snap Baru')
			sess.run(b.global_variables_initializer())
		return (sess,saver)


	def toSparse(self, texts):
		indices = []
		values = []
		shape = [len(texts), 0] # last entry must be max(labelList[i])

		#go over all texts
		for (batchElement, text) in enumerate(texts):
			#convert to string of label (i.e. class-ids)
			labelStr = [self.charList.index(c) for c in text]
			if len(labelStr) > shape[1]:
				shape[1] = len(labelStr)
			#put each label into sparse tensor
			for (i, label) in enumerate(labelStr):
				indices.append([batchElement, i])
				values.append(label)

		return (indices, values, shape)


	def decoderOutputToText(self, ctcOutput, batchSize):
		#label dari batch
		encodedLabelStrs = [[] for i in range(batchSize)]

		if self.decoderType == DecoderType.BestPath:
			decoded=ctcOutput[0][0] 

			#batch -> values
			idxDict = { b : [] for b in range(batchSize) }
			for (idx, idx2d) in enumerate(decoded.indices):
				label = decoded.values[idx]
				#index b t
				batchElement = idx2d[0]
				encodedLabelStrs[batchElement].append(label)
		return [str().join([self.charList[c] for c in labelStr]) for labelStr in encodedLabelStrs]


	def trainBatch(self, batch):
		numBatchElements = len(batch.imgs)
		sparse = self.toSparse(batch.gtTexts)
		rate = 0.01 if self.batchesTrained < 10 else (0.001 if self.batchesTrained < 10000 else 0.0001) # decay learning rate
		evalList = [self.optimizer, self.loss]
		feedDict = {self.inputImgs : batch.imgs, self.gtTexts : sparse , self.seqLen : [Model.maxTextLen] * numBatchElements, self.learningRate : rate, self.is_train: True}
		(_, lossVal) = self.sess.run(evalList, feedDict)
		self.batchesTrained += 1
		return lossVal


	
		"dump the output of the NN to CSV file(s)"
		dumpDir = '../dump/'
		if not os.path.isdir(dumpDir):
			os.mkdir(dumpDir)

		# iterate over all batch elements and create a CSV file for each one
		maxT, maxB, maxC = rnnOutput.shape
		for b in range(maxB):
			csv = ''
			for t in range(maxT):
				for c in range(maxC):
					csv += str(rnnOutput[t, b, c]) + ';'
				csv += '\n'
			fn = dumpDir + 'rnnOutput_'+str(b)+'.csv'
			print('Write dump of NN to file: ' + fn)
			with open(fn, 'w') as f:
				f.write(csv)


	def inferBatch(self, batch, calcProbability=False, probabilityOfGT=False):
		#decode
		numBatchElements = len(batch.imgs)
		evalRnnOutput = self.dump or calcProbability
		evalList = [self.decoder] + ([self.ctcIn3dTBC] if evalRnnOutput else [])
		feedDict = {self.inputImgs : batch.imgs, self.seqLen : [Model.maxTextLen] * numBatchElements, self.is_train: False}
		evalRes = self.sess.run(evalList, feedDict)
		decoded = evalRes[0]
		texts = self.decoderOutputToText(decoded, numBatchElements)
		
		probs = None
		if calcProbability:
			sparse = self.toSparse(batch.gtTexts) if probabilityOfGT else self.toSparse(texts)
			ctcInput = evalRes[1]
			evalList = self.lossPerElement
			feedDict = {self.savedCtcInput : ctcInput, self.gtTexts : sparse, self.seqLen : [Model.maxTextLen] * numBatchElements, self.is_train: False}
			lossVals = self.sess.run(evalList, feedDict)
			probs = np.exp(-lossVals)
		return (texts, probs)
	

	def save(self):
		self.snapID += 1
		self.saver.save(self.sess, '../model/snapshot', global_step=self.snapID)
 
