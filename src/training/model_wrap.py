from torch import nn
import torch
import numpy as np
from src.eval_metrics import (
    compute_edt,
    sample_coord,
)
import cc3d


class ModelPrevSegAndClickWrapper(nn.Module):
    def __init__(self, orig_network, n_max_clicks=5):
        print("with custom forward print shape")
        super().__init__()
        self.orig_network = orig_network
        # just for trainer necessary
        self.decoder = self.orig_network.decoder
        self.click_radius = 4  # that was from nninteractivev1.0
        self.n_max_clicks = n_max_clicks

    def forward(self, *args, **kwargs):
        with torch.no_grad():
            x_and_gts = args[0]
            x = x_and_gts[:, :-1]
            # gts must be binary mask!
            gts = x_and_gts[:, -1:]
            n_clicks = np.random.choice(
                range(self.n_max_clicks + 1),
                size=1,
                p=[0.5] + [0.5 / max(1, self.n_max_clicks)] * self.n_max_clicks,
            ).item()
            # print("n_clicks", n_clicks)

            if n_clicks > 0:
                seg_outputs = self.orig_network(x)
                final_seg_output = seg_outputs[0]  # check this more properly
                if final_seg_output.ndim < 5:
                    final_seg_output = final_seg_output[None]
                assert final_seg_output.ndim == 5
                final_seg_mask = (torch.diff(final_seg_output, dim=1) > 0).float()
                assert final_seg_mask.shape[1] == 1
                assert final_seg_mask.ndim == x.ndim
                x[:, 1] = final_seg_mask.squeeze(1)

                #  positive means adding to the proposed segmentation,
                # negative means subtracting from the proposed segmentation
                complete_pos_click_mask = torch.zeros_like(gts.bool())
                complete_neg_click_mask = torch.zeros_like(gts.bool())
                for _ in range(n_clicks):
                    mistakes_mask = (gts > 0) != (final_seg_mask > 0)

                    if torch.sum(mistakes_mask).item() == 0:
                        # no mistakes at all, nothing to fix... very unlikely
                        break

                    coord_per_image = []
                    positive_clicks = []
                    for i_image in range(len(mistakes_mask)):
                        error_mask = mistakes_mask[i_image, 0].cpu().numpy()
                        assert gts.shape[1] == 1
                        gts_this = gts[i_image, 0]
                        n_errors = np.sum(error_mask)
                        if n_errors > 0:
                            errors = cc3d.connected_components(
                                error_mask, connectivity=26
                            )  # 26 for 3D connectivity
                            # Calculate the sizes of connected error components
                            component_sizes = np.bincount(errors.flat)
                            # Ignore non-error regions
                            component_sizes[0] = 0
                            # Find the largest error component
                            largest_component_error = np.argmax(component_sizes)
                            # Find the voxel coordinates of the largest error component
                            largest_component = errors == largest_component_error
                            edt = compute_edt(largest_component)
                            center = sample_coord(edt)
                            positive = (gts_this[center[0], center[1], center[2]] > 0).item()
                            if positive:
                                assert (
                                    final_seg_mask[
                                        i_image, 0, center[0], center[1], center[2]
                                    ].item()
                                    == 0
                                )
                            else:
                                assert (
                                    final_seg_mask[
                                        i_image, 0, center[0], center[1], center[2]
                                    ].item()
                                    == 1
                                )
                        else:
                            positive = False
                            center = [
                                -self.click_radius * 2,
                                -self.click_radius * 2,
                                -self.click_radius * 2,
                            ]  # will be ignored later anyways through any_mistakes
                        coord_per_image.append(center)
                        positive_clicks.append(positive)
                    coords_z_y_x_per_im = torch.tensor(
                        coord_per_image, device=mistakes_mask.device
                    )
                    positive_clicks = torch.tensor(
                        positive_clicks, device=mistakes_mask.device
                    )

                    # Create indices
                    z_indices, y_indices, x_indices = torch.meshgrid(
                        torch.arange(mistakes_mask.shape[-3], device=mistakes_mask.device),
                        torch.arange(mistakes_mask.shape[-2], device=mistakes_mask.device),
                        torch.arange(mistakes_mask.shape[-1], device=mistakes_mask.device),
                        indexing="ij",
                    )
                    diffs = (
                        torch.stack((z_indices, y_indices, x_indices))[None]
                        - coords_z_y_x_per_im[:, :, None, None, None]
                    )
                    # set to zero if no errors, just for safety... should be 0 anyways
                    any_mistakes = torch.sum(mistakes_mask, dim=(1, 2, 3, 4)) > 0
                    clicks_mask = (
                        torch.sqrt(torch.sum(diffs**2, dim=1, keepdim=True))
                        < self.click_radius
                    ) * any_mistakes[:, None, None, None, None]

                    positive_clicks_mask = (
                        clicks_mask & positive_clicks[:, None, None, None, None]
                    )
                    negative_clicks_mask = clicks_mask & (
                        ~positive_clicks[:, None, None, None, None]
                    )
                    complete_pos_click_mask = complete_pos_click_mask | positive_clicks_mask
                    complete_neg_click_mask = complete_neg_click_mask | negative_clicks_mask

                # - Channel -4: Positive point interaction
                # - Channel -3: Negative point interaction

                assert complete_pos_click_mask.shape[1] == 1
                assert complete_neg_click_mask.shape[1] == 1
                x[:, -4] = complete_pos_click_mask.squeeze(1).float()
                x[:, -3] = complete_neg_click_mask.squeeze(1).float()

        return self.orig_network(x, *args[1:], **kwargs)
