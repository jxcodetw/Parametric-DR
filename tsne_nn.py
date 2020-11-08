import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from openTSNE import TSNE
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import SpectralEmbedding
from scipy.sparse import save_npz, load_npz
import random
from functools import partial
import timeit
import math

EPS = 1e-12
D_GRAD_CLIP = 1e14

class TSNE_NN():
	def __init__(self, device, network, n_epochs, batch_size=256):
		self.device = device
		self.network = network
		self.n_epochs = n_epochs
		self.batch_size = batch_size
		self.perplexity = 15
		self.test_data = None
		self.grads = []
	
	def fit(self, data, callback):
		encoder = self.network
		batch_size = self.batch_size
		device = self.device
		print('perplexity:', self.perplexity)
		
		print('calc P')
		pre_embedding = TSNE(perplexity=self.perplexity).prepare_initial(data)
		P_csc = pre_embedding.affinities.P
			
		print('Trying to put X into GPU')
		X = torch.from_numpy(data).float()
		X = X.to(device)
		self.X = X

		encoder = encoder.to(device)
		init_lr = 1e-3
		optimizer = optim.RMSprop(encoder.parameters(), lr=init_lr)
		lr_sched = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.n_epochs * math.ceil(len(X)/batch_size), eta_min=1e-7)
		
		def neg_squared_euc_dists(X):
			sum_X = X.pow(2).sum(dim=1)
			D = (-2 * X @ X.transpose(1, 0) + sum_X).transpose(1, 0) + sum_X
			return -D

		def w_tsne(Y):
			distances = neg_squared_euc_dists(Y)
			inv_distances = 1. / (1. - (distances)) #1 / (1+d^2)
			return inv_distances
		
		def KLD(P, Q):
			x = P/Q
			if x.requires_grad:
				def hook(grad):
					clipped_grad = grad.clamp(min=-D_GRAD_CLIP, max=D_GRAD_CLIP)
					return clipped_grad
				x.register_hook(hook)
			return P * torch.log(x)
		
		iteration = 0
		print('optimizing...')
		for epoch in range(self.n_epochs):
			iteration += 1

			idxs = torch.randperm(len(X))
			
			loss_total = []
			update_time = []
			for i in range(0, len(X), batch_size):
				start_time = timeit.default_timer()
				idx = idxs[i:i+batch_size]
				_p = torch.Tensor(P_csc[idx][:, idx].toarray()).float()
				if iteration < 250:
					_p *= 4
				p = (_p+EPS).to(device)
				optimizer.zero_grad()
				y = encoder(X[idx])
				w = w_tsne(y)
				q = w / torch.sum(w)
				loss = KLD(p, q).sum()
				loss.backward()
				loss_total.append(loss.item())
				torch.nn.utils.clip_grad_value_(encoder.parameters(), 4)
				optimizer.step()
				elapsed = timeit.default_timer() - start_time
				update_time.append(elapsed)
			
				lr_sched.step()


			callback(self, np.mean(update_time), epoch, np.mean(loss_total))

