import os

import numpy as np
import torch
from src.nninteractive import nnInteractivePredictor


predictor = nnInteractivePredictor(
            checkpoint_path="model/nnInteractive_v1.0",
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

for filename in os.listdir('inputs'): # the eval script copies each volume to this directory sequentially
    if filename.endswith('.npz'):
        # Load input
        data = np.load(f'inputs/{filename}', allow_pickle=True)
        image = data['imgs']
        
        # Get clicks if available
        clicks = data.get('clicks', None)  
        
        # Get bounding box if available
        bboxs = data.get('boxes', None)
        
        # Get previous prediction if available
        prev_pred = data.get('prev_pred', None)

        # Get spacing 
        spacing = data.get('spacing', None)
        
        # Run your segmentation model
        segmentation, infer_time = predictor.predict(
                    image=image,
                    spacing=spacing,
                    bboxs=bboxs,
                    clicks=clicks,
                    prev_pred=prev_pred,  # Pass previous prediction
                )
        
        # Save output with the same filename
        np.savez_compressed(
            f'outputs/{filename}',
            segs=segmentation
        )