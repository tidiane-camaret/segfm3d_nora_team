import os
import numpy as np
import torch
from src.method import SimplePredictor



predictor = SimplePredictor(
            checkpoint_path="model/checkpoint_11_2025-05-30_17-56-57-428.pth",
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            include_previous_clicks=True,
            n_pred_iters= 1)

for filename in os.listdir('inputs'): # the eval script copies each volume to this directory sequentially
    if filename.endswith('.npz'):
        # Load input
        data = np.load(f'inputs/{filename}', allow_pickle=True)
        image = data['imgs']

        # Get bounding box if available
        bboxs = data.get('boxes', None)

        n_classes = len(bboxs) if bboxs is not None else 0
        # Get clicks if available
        clicks_cls = data.get('clicks', [
                {"fg": [], "bg": []} for _ in range(n_classes)
            ])  # CAREFUL : clicks_cls is named 'clicks' in the npz !
        clicks_order = data.get('clicks_order', [[] for _ in range(n_classes)])

        clicks = (clicks_cls, clicks_order) #if clicks_cls is not None else (None, None)

        
        # Get previous prediction if available
        prev_pred = data.get('prev_pred', np.zeros_like(image, dtype=np.uint8))

        """
        print(f'Processing {filename}...')
        print(f'clicks: {clicks}')
        print(f'bboxs: {bboxs}')
        print(f'is_bbox_iteration: {is_bbox_iteration}')
        print(f'prev_pred: {prev_pred}')
        """
        # Run your segmentation model
        segmentation, _ = predictor.predict(
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