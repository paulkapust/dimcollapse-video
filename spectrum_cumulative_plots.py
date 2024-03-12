import numpy as np
import matplotlib.pyplot as plt
import torch
import argparse
from sklearn.decomposition import PCA
import sys

parser = argparse.ArgumentParser(description='Visualize graphs of singular values')
parser.add_argument('--methods', type=str, nargs='+', required=True, help='list of method names')
parser.add_argument('--representations', type=str, nargs='+', help='path to stored representations')
parser.add_argument('--embeddings', type=str, nargs='+', help='path to stored embeddings')

def load_normalize(paths):
	z = torch.load(paths)
	z = torch.nn.functional.normalize(z, dim=1)			
	z = z.cpu().detach().numpy()
	return z

# Following https://github.com/facebookresearch/directclr/blob/main/spectrum.py
def compute_spectrum(features):
	h = np.transpose(features)
	c = np.cov(h)
	_, d, _ = np.linalg.svd(c)
	print(d.shape)
	return np.log(d)

def compute_cumulative(features):
	pca = PCA()
	pca.fit_transform(features)
	s = pca.singular_values_
	
	cum_singular = np.cumsum(s) / s.sum()

	auc = np.cumsum(s).sum()/s.shape[0]/ s.sum()

	return cum_singular, auc

def compute_spectrum_cumulative(paths):
	spectrum_list = []
	cumulative_list = []
	auc_list = []

	for _, feature_path in enumerate(paths):

		# Load features of shape (number of samples) x (number of dimensions)
		features = load_normalize(feature_path)

		# Compute logarithmic singular value spectrum and cumulative explained variance of features
		spectrum = compute_spectrum(features)
		cumulative, auc = compute_cumulative(features)

		spectrum_list.append(spectrum)
		cumulative_list.append(cumulative)
		auc_list.append(auc)

	return spectrum_list, cumulative_list, auc_list

def main():
	args = parser.parse_args()

	if not args.embeddings and not args.representations:
		sys.exit("No representations or embeddings provided!")

	print("Computing singular value spectrum and cumulative explained variance of features...")

	if args.representations:
		reps_spectrum, reps_cumulative, reps_auc = compute_spectrum_cumulative(args.representations)

	if args.embeddings:
		embed_spectrum, embed_cumulative, embed_auc = compute_spectrum_cumulative(args.embeddings)

	print("Plotting singular value spectrum and cumulative explained variance of features...")

	if not args.representations:
		fig, (ax3, ax4) = plt.subplots(1, 2, figsize=(12,3))
	elif not args.embeddings:
		fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,3))
	else:
		fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12,6))
		
	plt.subplots_adjust(left=0.125,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.25, 
                    hspace=0.55)

	plt.rcParams.update({'font.size': 14})

	if args.representations:
		for i, s in enumerate(reps_spectrum):
			ax1.plot(np.arange(s.shape[0]), s, label=args.methods[i], linewidth=2)
			ax1.set_title("Representation Space")
			ax1.set_xlabel('Singular Value Rank Index', fontsize=12)
			ax1.set_ylabel("Log of Singular Values", fontsize=12)
			ax1.set_ylim(top=0)
			ax1.set_ylim(bottom=-25)
			ax1.set_yticks(np.arange(0, -26, step=-5))
			ax1.legend(fontsize=10, loc="lower left")

		for i, s in enumerate(reps_cumulative):
			ax2.plot(np.arange(s.shape[0]), s, label=args.methods[i]+ " " + f"(AUC={round(reps_auc[i], 2)})", linewidth=2)
			ax2.set_title("Representation Space")
			ax2.set_ylabel('Cumulative Explained Variance', fontsize=12)
			ax2.set_xlabel('Singular Value Rank Index', fontsize=12)
			ax2.legend(fontsize=10, loc="lower right")

	if args.embeddings:
		for i, s in enumerate(embed_spectrum):
			ax3.plot(np.arange(s.shape[0]), s, label=args.methods[i], linewidth=2)
			ax3.set_title("Embedding Space")
			ax3.set_xlabel('Singular Value Rank Index', fontsize=12)
			ax3.set_ylabel("Log of Singular Values", fontsize=12)
			ax3.set_ylim(top=0)
			ax3.set_ylim(bottom=-25)
			ax3.set_yticks(np.arange(0, -26, step=-5))
			ax3.legend(fontsize=10, loc="lower left")

		for i, s in enumerate(embed_cumulative):
			ax4.plot(np.arange(s.shape[0]), s, label=args.methods[i]+ " " + f"(AUC={round(embed_auc[i], 2)})", linewidth=2)
			ax4.set_title("Embedding Space")
			ax4.set_ylabel('Cumulative Explained Variance', fontsize=12)
			ax4.set_xlabel('Singular Value Rank Index', fontsize=12)
			ax4.legend(fontsize=10, loc="lower right")
		
	plt.savefig("_".join(args.methods) + ".pdf", format='pdf', bbox_inches='tight')
	plt.show()

if __name__ == '__main__':
    main()
