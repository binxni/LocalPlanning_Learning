import numpy as np
import torch


def preprocess(scan_msg):
    ranges = np.array(scan_msg.ranges, dtype=np.float32)
    ranges = np.nan_to_num(ranges, nan=scan_msg.range_max,
                           posinf=scan_msg.range_max,
                           neginf=scan_msg.range_min)
    ranges = ranges / scan_msg.range_max
    n = 360
    if ranges.size != n:
        idx = np.linspace(0, ranges.size - 1, n)
        ranges = np.interp(idx, np.arange(ranges.size), ranges)
    tensor = torch.from_numpy(ranges).unsqueeze(0)
    return tensor
