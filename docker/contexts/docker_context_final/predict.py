import os
import numpy as np
import torch
from src.method import nnInteractiveOrigPredictor


os.environ['nnUNet_raw'] = '/workspace/nnUNet_raw'
os.environ['nnUNet_preprocessed'] = '/workspace/nnUNet_preprocessed'
os.environ['nnUNet_results'] = '/workspace/nnUNet_results'

predictor = nnInteractiveOrigPredictor(
            checkpoint_path="model/finetuned_20250529_191827_v4i5izr1",
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

        clicks = (clicks_cls, clicks_order) if clicks_cls is not None else (None, None)

        # Get bounding box if available
        bboxs = data.get('boxes', None)
        
        # Get previous prediction if available
        prev_pred = data.get('prev_pred', None)

        """
        print(f'Processing {filename}...')
        print(f'clicks: {clicks}')
        print(f'bboxs: {bboxs}')
        print(f'is_bbox_iteration: {is_bbox_iteration}')
        print(f'prev_pred: {prev_pred}')
        """
        # Run your segmentation model
        segmentation = predictor.predict(
                    image=image,
                    bboxs=bboxs,
                    clicks=clicks,  # Pass clicks
                    prev_pred=prev_pred,  # Pass previous prediction
                )
        
        # Save output with the same filename
        np.savez_compressed(
            f'outputs/{filename}',
            segs=segmentation
        )