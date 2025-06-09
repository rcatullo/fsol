from pathlib import Path         # or: import path
import pickle
import numpy as np

fname = Path("ckpt/test/checkpoint.ckpt") # original checkpoint
fname2 = Path("ckpt/trained/checkpoint.ckpt") # new checkpoint

with fname.open("rb") as f:
    data1 = pickle.load(f)

with fname2.open("rb") as f:
	data2 = pickle.load(f)

print(list(data1.keys()))
print(data1["step"])
print(data2["step"])

data2_keys = list(data2["agent"].keys())
data1_keys = list(data1["agent"].keys())

for k, v in data2["agent"].items():
	if k not in data1_keys:
		data1["agent"][k] = v
	else:
		if isinstance(v, np.ndarray):
			assert v.shape == data1["agent"][k].shape, f"Shape mismatch for key {k}: {v.shape} vs {data1['agent'][k].shape}"
	
data1["step"] = 0

with open("ckpt/merged_checkpoint.ckpt", "wb") as f:
	pickle.dump(data1, f)
