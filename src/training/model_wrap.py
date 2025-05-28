from torch import nn
import torch
import numpy as np


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
                range(self.n_max_clicks + 1), size=1, p=[0.75] + [0.25 / max(1, self.n_max_clicks)] * self.n_max_clicks,
            ).item()
            # print("n_clicks", n_clicks)

            if n_clicks > 0:
                seg_outputs = self.orig_network(x)
                final_seg_output = seg_outputs[0] # check this more properly
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
                    mistakes_coords = torch.argwhere(mistakes_mask)
                    if len(mistakes_coords) == 0:
                        # no mistakes at all, nothing to fix... very unlikely
                        break

                    coord_per_image = []
                    positive_clicks = []
                    for i_image in range(mistakes_mask.shape[0]):
                        this_coords = mistakes_coords[mistakes_coords[:, 0] == i_image]
                        if len(this_coords) > 0:
                            i_wanted_coord = np.random.choice(len(this_coords))
                            wanted_coord = this_coords[i_wanted_coord]
                            positive = gts[*wanted_coord] > 0
                        else:
                            # add a click outside the image to simulate no click
                            # first just take first mistake coord as some random coord
                            wanted_coord = mistakes_coords[0]
                            # doesn't matter if positive or not
                            positive = gts[*wanted_coord] > 0
                            # move outside image
                            wanted_coord = wanted_coord * 0 - self.click_radius * 2
                        coord_per_image.append(wanted_coord[2:])
                        positive_clicks.append(positive)
                    coords_z_y_x_per_im = torch.stack(coord_per_image)
                    # Create indices
                    z_indices, y_indices, x_indices = torch.meshgrid(
                        torch.arange(
                            mistakes_mask.shape[-3], device=mistakes_mask.device
                        ),
                        torch.arange(
                            mistakes_mask.shape[-2], device=mistakes_mask.device
                        ),
                        torch.arange(
                            mistakes_mask.shape[-1], device=mistakes_mask.device
                        ),
                        indexing="ij",
                    )
                    diffs = (
                        torch.stack((z_indices, y_indices, x_indices))[None]
                        - coords_z_y_x_per_im[:, :, None, None, None]
                    )
                    clicks_mask = (
                        torch.sqrt(torch.sum(diffs**2, dim=1, keepdim=True))
                        < self.click_radius
                    )
                    positive_clicks_mask = (
                        clicks_mask
                        & torch.stack(positive_clicks)[:, None, None, None, None]
                    )
                    negative_clicks_mask = clicks_mask & (
                        ~torch.stack(positive_clicks)[:, None, None, None, None]
                    )
                    complete_pos_click_mask = (
                        complete_pos_click_mask | positive_clicks_mask
                    )
                    complete_neg_click_mask = (
                        complete_neg_click_mask | negative_clicks_mask
                    )

                # - Channel -4: Positive point interaction
                # - Channel -3: Negative point interaction

                assert complete_pos_click_mask.shape[1] == 1
                assert complete_neg_click_mask.shape[1] == 1
                x[:, -4] = complete_pos_click_mask.squeeze(1).float()
                x[:, -3] = complete_neg_click_mask.squeeze(1).float()

        return self.orig_network(x, *args[1:], **kwargs)