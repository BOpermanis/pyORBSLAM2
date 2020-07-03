import pickle
import numpy as np

with open("/home/slam_data/data_sets1/temp.pickle", 'rb') as conn:
    kf_ids_from_mps, kf_ids, mp_3dpts, kf_3dpts = pickle.load(conn)

kf_ids_from_mps = kf_ids_from_mps[:, 0]
kf_ids = kf_ids[:, 0]

s = set(kf_ids)
s1 = set(kf_ids_from_mps)

print(kf_ids_from_mps[:4])
print(kf_ids[:4])
print(len(kf_ids_from_mps), len(set(kf_ids_from_mps)))
print(len(kf_ids), len(set(kf_ids)))
print(kf_ids.dtype, kf_ids_from_mps.dtype)
print(mp_3dpts.dtype, kf_3dpts.dtype)

print(len(s.intersection(s1)))