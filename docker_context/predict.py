import os
from tabnanny import verbose

import numpy as np
import torch
from src.method import nnInteractivePredictor


predictor = nnInteractivePredictor(
            checkpoint_path="model/nnInteractive_v1.0",
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            verbose=False,)

for filename in os.listdir('inputs'): # the eval script copies each volume to this directory sequentially
    if filename.endswith('.npz'):
        # Load input
        data = np.load(f'inputs/{filename}', allow_pickle=True)
        image = data['imgs']
        
        # Get clicks if available
        clicks_cls = data.get('clicks', None)  # CAREFUL : clicks_cls is named 'clicks' in the npz !
        clicks_order = data.get('clicks_order', None)


        clicks = (clicks_cls, clicks_order) if clicks_cls is not None else None

        is_bbox_iteration = True
        if clicks is not None:
            if len(clicks_order[0]) > 0:
                is_bbox_iteration = False

        
        
        # Get bounding box if available
        bboxs = data.get('boxes', None)
        
        # Get previous prediction if available
        prev_pred = data.get('prev_pred', None)

        # Get spacing 
        spacing = data.get('spacing', None)
        """
        print(f'Processing {filename}...')
        print(f'clicks: {clicks}')
        print(f'bboxs: {bboxs}')
        print(f'is_bbox_iteration: {is_bbox_iteration}')
        print(f'prev_pred: {prev_pred}')
        """
        # Run your segmentation model
        segmentation, infer_time = predictor.predict(
                    image=image,
                    spacing=spacing,
                    bboxs=bboxs,
                    clicks=clicks,  # Pass clicks
                    is_bbox_iteration=is_bbox_iteration,  # Set to False for non-iterative prediction
                    prev_pred=prev_pred,  # Pass previous prediction
                )
        
        # Save output with the same filename
        np.savez_compressed(
            f'outputs/{filename}',
            segs=segmentation
        )