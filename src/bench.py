"""Measure training throughput on this GPU so the run budget can be planned."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C
from src import model as M


def bench(n_channels: int, batch: int, mcfg: dict, amp: bool, steps: int = 25) -> dict:
    dev = torch.device("cuda")
    net = M.build(n_channels, mcfg).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-4)
    lossf = nn.HuberLoss(delta=1.0)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    x = torch.randn(batch, mcfg["context_len"], n_channels, device=dev)
    y = torch.randn(batch, device=dev)

    for i in range(steps + 5):
        if i == 5:
            torch.cuda.synchronize(); t0 = time.time()
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=amp):
            loss = lossf(net(x).squeeze(-1), y)
        scaler.scale(loss).backward()
        scaler.step(opt); scaler.update()
    torch.cuda.synchronize()
    dt = (time.time() - t0) / steps
    peak = torch.cuda.max_memory_allocated() / 1e9
    torch.cuda.reset_peak_memory_stats()
    del net, opt, x, y
    torch.cuda.empty_cache()
    return {"s_per_step": dt, "samples_per_s": batch / dt, "peak_gb": peak}


if __name__ == "__main__":
    base = dict(C.MODEL)
    print(f"{'config':46s} {'s/step':>8s} {'samp/s':>9s} {'peak GB':>8s}")
    trials = [
        ("F2(35ch) b128 p16/s8 fp32", 35, 128, {}, False),
        ("F2(35ch) b128 p16/s8 amp", 35, 128, {}, True),
        ("F2(35ch) b192 p16/s8 fp32", 35, 192, {}, False),
        ("F2(35ch) b128 p16/s16 fp32", 35, 128, {"stride": 16}, False),
        ("F1(16ch) b128 p16/s8 fp32", 16, 128, {}, False),
        ("F2(35ch) b128 mean-head fp32", 35, 128, {"head": "mean"}, False),
    ]
    for name, ch, b, over, amp in trials:
        cfg = {**base, **over}
        try:
            r = bench(ch, b, cfg, amp)
            print(f"{name:46s} {r['s_per_step']:8.4f} {r['samples_per_s']:9.1f} {r['peak_gb']:8.2f}")
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"{name:46s} {'OOM':>8s}")
