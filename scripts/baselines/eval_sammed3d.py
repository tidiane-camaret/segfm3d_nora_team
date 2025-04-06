"""
SAM Med 3d evaluation

original command : 
cd SAM-Med3D/
python medim_infer.py 
"""
import os 
import sys

import yaml
config = yaml.safe_load(open("config.yaml"))

sys.path.append(config["SAM_REPO_DIR"])
from medim_infer import *

if __name__ == "__main__":
    ''' 1. prepare the pre-trained model with local path or huggingface url '''
    # ckpt_path = "https://huggingface.co/blueyo0/SAM-Med3D/blob/main/sam_med3d_turbo.pth"
    # or you can use the local path like: 
    ckpt_path = config["SAM_CKPT_PATH"]
    model = medim.create_model("SAM-Med3D",
                               pretrained=True,
                               checkpoint_path=ckpt_path)

    ''' 2. read and pre-process your input data '''
    npz_file = glob(os.path.join(config["VAL_DIR"],"3D_val_npz","*.npz"))[0]
    out_dir = os.path.join(config["RESULTS_DIR"], "sammed3d")
    imgs, spacing, all_clicks, prev_pred = read_data_from_npz(npz_file)
    final_pred = np.zeros_like(imgs, dtype=np.uint8)
    for idx, cls_clicks in all_clicks.items():
        category_index = idx + 1
        # import pdb; pdb.set_trace()
        pred_ori = prev_pred==category_index
        final_pred[pred_ori!=0] = category_index
        if (cls_clicks[-1][1][0] == 1):
            cls_gt = create_gt_arr(imgs.shape, cls_clicks[-1][0], category_index=category_index)
            # print(category_index, imgs.shape, spacing, cls_clicks, (cls_gt==category_index).sum())
            # continue
            cls_prev_seg = prev_pred==category_index
            roi_image, roi_label, roi_prev_seg, meta_info = data_preprocess(imgs, cls_gt, cls_prev_seg,
                                                            orig_spacing=spacing, 
                                                            category_index=category_index)

            # import pdb; pdb.set_trace()
            ''' 3. infer with the pre-trained SAM-Med3D model '''
            
            roi_pred = sam_model_infer(model, roi_image, roi_gt=roi_label, prev_low_res_mask=roi_prev_seg)

            # import pdb; pdb.set_trace()
            ''' 4. post-process and save the result '''
            pred_ori = data_postprocess(roi_pred, meta_info, out_dir)
            final_pred[pred_ori!=0] = category_index

    output_path = osp.join(out_dir, osp.basename(npz_file))
    np.savez_compressed(output_path, segs=final_pred)
    print("result saved to", output_path)

    # import pdb; pdb.set_trace()
    from surface_distance import compute_surface_distances, compute_surface_dice_at_tolerance, compute_dice_coefficient
    def compute_multi_class_dsc(gt, seg):
        dsc = []
        for i in np.unique(gt)[1:]: # skip bg
            gt_i = gt == i
            seg_i = seg == i
            dsc.append(compute_dice_coefficient(gt_i, seg_i))
            print("dsc", dsc[-1])
        return np.mean(dsc)

    # # compute_multi_class_dsc(roi_label, roi_label)
    # img_data = np.load('/mnt/sh_flex_storage/home/wanghaoy/code/SAM_Med3D_debug/raw_data/biomed_val/3D_val_gt/CT_AbdomenAtlas_BDMAP_00000006.npz')
    # gt = img_data['gts']

    # # gt = (gt==2)
    # # final_pred = (final_pred==1)
    # dsc = compute_multi_class_dsc(gt, final_pred)
    # print('all dice:', dsc)
    # compute_multi_class_dsc(gt==3, np.flip(pred_ori.transpose(2, 1, 0), axes=1))
