import os
import sys
import time
import pathlib
from argparse import ArgumentParser

from sklearn.decomposition import PCA
from joblib import dump as job_dump
import numpy as np
import torch

from network import FCEncoder

parser = ArgumentParser()
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--verbose', type=int, default=1)
parser.add_argument('--shuffle_data', type=int, default=0)
parser.add_argument('--cuda', type=int, default=1)
parser.add_argument('--perplexity', type=int, default=15)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--hidden_dim', type=int, default=256)
parser.add_argument('--low_dim', type=int, default=2)
parser.add_argument('--num_layers', type=int, default=3)
parser.add_argument('--method', type=str, default='ae', choices=[
  'tsne-nn',
  'largevis-nn',
  'umap-nn'
])
parser.add_argument('--dataset', type=str, default='coil20.npz')
parser.add_argument('--output_dir', type=str, default='results')
parser.add_argument('--max_dataset_size', type=int, default=10000)
args = parser.parse_args()

start_time = time.time()

if args.max_dataset_size == -1:
	args.max_dataset_size = None
print('log saved at:', args.output_dir)
pathlib.Path(args.output_dir).mkdir(parents=True, exist_ok=True)
_P = lambda p: os.path.join(args.output_dir, p)
np.savez_compressed(_P('args'), args=args)

# load data
print('load data')
dataset = np.load(args.dataset)
if args.shuffle_data:
	print('shuffle data')
	perm = np.random.permutation(len(dataset['data']))
else:
	perm = np.arange(len(dataset['data']))
data = dataset['data'][perm][:args.max_dataset_size]
print('data.shape', data.shape)

# create device
device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')
print('Device:', device)

# create encoder
encoder = FCEncoder(data.shape[1], num_layers=args.num_layers, hidden_dim=args.hidden_dim, low_dim=args.low_dim)
print('Encoder: ', encoder)

def default_logger(dr, elapsed, epoch, loss):
	if args.verbose:
		print('{:03d}/{:03d}'.format(epoch, args.epochs), '{:.5f}'.format(loss), '{:.5f}s'.format(elapsed))

if args.method == 'tsne-nn':
	from tsne_nn import TSNE_NN
	dr = TSNE_NN(device, encoder, n_epochs=args.epochs, batch_size=args.batch_size)
	dr.perplexity = args.perplexity
	dr.fit(data, default_logger)
elif args.method == 'largevis-nn':
	from largevis_nn import LARGEVIS_NN
	dr = LARGEVIS_NN(device, encoder, n_epochs=args.epochs, batch_size=args.batch_size)
	dr.fit(data, default_logger)
elif args.method == 'umap-nn':
	from umap_nn import UMAP_NN
	dr = UMAP_NN(device, encoder, n_epochs=args.epochs, batch_size=args.batch_size)
	dr.fit(data, default_logger)

np.savez_compressed(_P('time'.format(args.method)), elapsed=time.time()-start_time)
torch.save(encoder, _P('encoder.pth'))
with torch.no_grad():
	embedding = encoder(dr.X)
	np.savez_compressed(_P('Y'), Y=embedding.detach().cpu().numpy())
print('done', args.output_dir)