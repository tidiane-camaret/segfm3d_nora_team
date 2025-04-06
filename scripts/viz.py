import numpy as np
img = np.load("/work/dlclarge2/ndirt-SegFM3D/data/3D_val_npz/CT_AMOS_amos_0018.npz", allow_pickle=True)
print(list(img.keys()))
print(img['imgs'].shape)

print(img['boxes'])

print(img['spacing'])

print(img['text_prompts'])