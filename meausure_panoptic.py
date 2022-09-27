import argparse
import os
from os import path
import pickle

import torch
import torch.utils.data as data

from seamseg.data import (
    ISSTransform,
    ISSDataset
)
from seamseg.utils.panoptic import panoptic_stats, PanopticPreprocessing

parser = argparse.ArgumentParser(
    description="Simple Script to measure panotic quality between a ground truth set and a testset"
)
parser.add_argument("--gt", help="Path to the ground truth root data dir", type=str)
parser.add_argument(
    "--target", help="Path the results of the source folder of the results", type=str
)
parser.add_argument(
    "--result", help="Path to the config file to read in the configs for panoptic", type=str
)

def func_mapper(self, eval_mode: str, val: int):
    if eval_mode == "things":
        converted_value = val + self.vistas_stuff
    elif eval_mode == "stuff":
        converted_value = val
    else:
        raise argparse.ArgumentError(
            "eval_mode must be either 'things' or 'stuff', got {}".format(eval_mode)
        )

    if converted_value in self.lookup_dict:
        return self.lookup_dict[converted_value]
    else:
        return self.void_value


def main(args):

    transform = ISSTransform(shortest_size=512, longest_max_size=2661)
    gt_dataset = ISSDataset(args.gt, "val", transform)
    num_stuff = gt_dataset.num_stuff
    num_classes = gt_dataset.num_categories

    vistas_num_stuff = 28
    city_num_stuff = 11

    res_dataset = ResultDataset(
        args.target,
        transform=None,
        path_modifier=lambda x: x.split("_")[0],
        file_suffix="_leftImg8bit",
        dtype="pth.tar",
    )

    panoptic_buffer = torch.zeros(4, num_classes, dtype=torch.double)

    # Iterate over the entire dataset
    for i in range(len(gt_dataset)):

        if i % 10 == 0:
            print(f"Processing image {i} of {len(gt_dataset)}")

        gt_out = gt_dataset[i]
        msk_gt = gt_out["msk"].cpu()
        cat_gt = gt_out["cat"].cpu()
        iscrowd = gt_out["iscrowd"].cpu()

        msk_gt = msk_gt.squeeze(0)
        sem_gt = cat_gt[msk_gt]
        cmap = msk_gt.new_zeros(cat_gt.numel())
        cmap[~iscrowd] = torch.arange(
            0, (~iscrowd).long().sum().item(), dtype=cmap.dtype, device=cmap.device
        )
        msk_gt = cmap[msk_gt]
        cat_gt = cat_gt[~iscrowd]

        res_out = res_dataset[gt_out["idx"]]
        panoptic_merge = PanopticPreprocessing()

        panoptic_result = panoptic_merge(
            res_out["sem_pred"],
            res_out["bbx_pred"],
            res_out["cls_pred"],
            res_out["obj_pred"],
            res_out["msk_pred"],
            city_num_stuff,
        )

        # Convert results to common ground truth --> Compute panoptic quality

        stats = panoptic_stats(msk_gt, cat_gt, panoptic_result, num_classes, num_stuff)
        panoptic_buffer += torch.stack(stats, dim=0)
        # We receive IOU, TP, FP, FN

    denom = panoptic_buffer[1] + 0.5 * (panoptic_buffer[2] + panoptic_buffer[3])
    denom[denom == 0] = 1.0
    scores = panoptic_buffer[0] / denom

    pan_score = scores.mean().item()
    pan_score_stuff = scores[:num_stuff].mean().item()
    pan_score_thing = scores[num_stuff:].mean().item()

    with open(path.join(args.result, "panoptic.txt"), "w") as f:
        f.write(f"panoptic: {pan_score}\n")
        f.write(f"panoptic_stuff: {pan_score_stuff}\n")
        f.write(f"panoptic_thing: {pan_score_thing}\n")
        f.write("\n\n\n")
        for score in scores:
            f.write(f"{score}\n")

    return pan_score, pan_score_stuff, pan_score_thing


class ResultDataset(data.Dataset):
    """Small Dataset which iterates over results"""

    def __init__(
        self, root_dir, transform=None, path_modifier=None, file_suffix=None, dtype=None
    ) -> None:
        super().__init__()
        self.root_dir = root_dir
        self.transform = transform
        self.element_list = os.listdir(root_dir)
        self.nof_elements = len(self.element_list)
        self.dtype = dtype or ".".join(self.element_list[0].split(".")[1:])
        if path_modifier is not None:
            if type(path_modifier) is str:
                self.path_modifier = lambda x: path_modifier
            elif callable(path_modifier):
                self.path_modifier = path_modifier
            else:
                raise argparse.ArgumentError("Expect file suffix to be either string or callable")
        else:
            self.path_modifier = None
        self.file_suffix = file_suffix

    def __getitem__(self, idx, pkl=False):
        # Indicate if file has to be read using standard pickle
        if type(idx) is str:
            file_name = path.join(self.root_dir, self.path_modifier(idx), idx)\
                + (self.file_suffix or "")\
                + "."\
                + self.dtype
            if not pkl:
                file = torch.load(file_name,
                    map_location=torch.device("cpu")
                )
            else:
                with open(file_name,"rb") as f:
                    file = pickle.load(f)
        elif type(idx) is int:
            file = torch.load(path.join(self.root_dir, self.element_list[idx]),
                                map_location=torch.device("cpu"))
        else:
            raise IndexError("Expect type string or int as index, got {}".format(type(idx)))

        if self.transform is not None:
            return self.transform(file)
        else:
            return file

    def __len__(self) -> int:
        return self.nof_elements


if __name__ == "__main__":
    main(parser.parse_args())
