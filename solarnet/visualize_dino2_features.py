import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

sys.path.append('./')

from solarnet.config import UK_1M_FEATS_PATH
from solarnet.datasets.uk_feat_dataset import _get_subsample

# --------------------------------------------------------------------------

pkl_file = UK_1M_FEATS_PATH
per_class_sample_size = 10_000
feature_size = 3072

with open(pkl_file, 'rb') as handle:
    all_paths = pickle.load(handle)
all_paths = _get_subsample(all_paths, per_class_sample_size)

positive_feats = []
negative_feats = []
for i, (name, array) in enumerate(all_paths.items()):
    if name.endswith('-P.tif'): positive_feats.append(array)
    if name.endswith('-N.tif'): negative_feats.append(array)
positive_feats = np.stack(positive_feats, 0)
negative_feats = np.stack(negative_feats, 0)

# reducer = PCA(n_components=2)
reducer = TSNE(n_components=2, perplexity=40, n_iter=300)
pca_result = reducer.fit_transform(np.concatenate([positive_feats, negative_feats], axis=0))
positive_points = pca_result[:per_class_sample_size, :]
negative_points = pca_result[per_class_sample_size:, :]

plt.figure(figsize=(16,10))
plt.scatter(positive_points[:, 0], positive_points[:, 1], s=10, alpha=0.7, color='red')
plt.scatter(negative_points[:, 0], negative_points[:, 1], s=10, alpha=0.7, color='blue')
plt.axis('off')
plt.savefig('sample.png')
plt.close()