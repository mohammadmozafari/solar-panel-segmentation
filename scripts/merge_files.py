import glob
import pickle
import numpy as np

# general_dict = {}
# files = glob.glob('./data/temp/*')
# for f in files:
#     x = np.load(f, allow_pickle=True)
#     x = x.flatten()
#     d = x[0]
    
#     print(len(d))
#     general_dict.update(d)

# with open('dinov2_features.pickle', 'wb') as handle:
#     pickle.dump(general_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
with open('../Datasets/london300_features_dinov2_giant_3072.pickle', 'rb') as handle:
    general_dict = pickle.load(handle)
    
print(len(general_dict))

for k, v in general_dict.items():
    print(k)
    print(v.shape)
    # exit()