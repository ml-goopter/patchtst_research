"""Stage-one expected-return training (PLAN.md sections 12, 24).

One run = one (feature_set, fold, seed). Early stopping on validation loss; the
selected checkpoint is scored on the fold's test window and predictions are saved
in raw log-return units for downstream validation.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C
from src import model as M
from src.datamodule import GPUWindows, Normalizer, Panel
from src.metrics import decile_monotonicity, expected_return_metrics
from src.splits import make_folds


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_loss(name: str, delta: float):
    if name == "huber":
        return nn.HuberLoss(delta=delta)
    if name == "mse":
        return nn.MSELoss()
    raise ValueError(name)


@torch.no_grad()
def predict(net, gw: GPUWindows, idx: np.ndarray, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    net.eval()
    P, Y = [], []
    t = torch.from_numpy(idx).to(gw.device)
    for i in range(0, len(t), batch_size):
        xb, yb = gw.gather(t[i: i + batch_size])
        P.append(net(xb).squeeze(-1).float().cpu())
        Y.append(yb.float().cpu())
    return torch.cat(P).numpy(), torch.cat(Y).numpy()


def run_one(panel_df: pd.DataFrame, fold: dict, seed: int, feature_set: str,
            loss_name: str = "huber", model_over: dict | None = None,
            train_over: dict | None = None, tag: str = "stage1",
            device: str = "cuda", verbose: bool = True) -> dict:
    mcfg = {**C.MODEL, **(model_over or {})}
    tcfg = {**C.TRAIN, **(train_over or {})}
    dev = torch.device(device)
    set_seed(seed)

    pan = Panel(panel_df, feature_set)
    idxs = pan.fold_indices(fold)
    if min(len(v) for v in idxs.values()) == 0:
        raise RuntimeError(f"empty split in fold {fold['fold']}: "
                           f"{ {k: len(v) for k, v in idxs.items()} }")

    norm = Normalizer(pan.X, pan.y, idxs["train"])
    gw = GPUWindows(norm.transform_X(pan.X), norm.transform_y(pan.y), dev)

    net = M.build(len(pan.features), mcfg).to(dev)
    n_params = sum(p.numel() for p in net.parameters())

    stride = max(1, int(tcfg["sample_stride"]))
    per_epoch = max(1, int(np.ceil(len(idxs["train"]) / stride / tcfg["batch_size"])))

    opt = torch.optim.AdamW(net.parameters(), lr=tcfg["lr"], weight_decay=tcfg["weight_decay"])
    steps = per_epoch * tcfg["max_epochs"]
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=tcfg["lr"], total_steps=steps, pct_start=0.15)
    lossf = make_loss(loss_name, delta=1.0)  # targets are robust-scaled, so delta=1 is ~1 IQR
    scaler = torch.amp.GradScaler("cuda", enabled=tcfg["amp"])
    gen = torch.Generator(device=dev).manual_seed(seed)

    best = {"val": np.inf, "epoch": -1, "state": None}
    history = []
    t_start = time.time()

    for epoch in range(tcfg["max_epochs"]):
        net.train()
        tot, nb = 0.0, 0
        # rotate which windows this epoch sees, so all of them are used over time
        epoch_idx = idxs["train"][epoch % stride:: stride]
        for xb, yb in gw.iterate(epoch_idx, tcfg["batch_size"], shuffle=True, generator=gen):
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=tcfg["amp"]):
                loss = lossf(net(xb).squeeze(-1), yb)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(net.parameters(), tcfg["grad_clip"])
            scaler.step(opt)
            scaler.update()
            if sched.last_epoch < steps - 1:
                sched.step()
            tot += loss.item(); nb += 1

        vp, vy = predict(net, gw, idxs["val"], tcfg["eval_batch_size"])
        val_loss = float(np.mean(np.abs(vp - vy)))  # scaled-space MAE, monotone in raw MAE
        history.append({"epoch": epoch, "train_loss": tot / nb, "val_mae_scaled": val_loss,
                        "minutes": round((time.time() - t_start) / 60, 2)})
        if verbose:
            print(f"    epoch {epoch:2d}  train {tot/nb:.5f}  val_mae {val_loss:.5f}  "
                  f"[{(time.time()-t_start)/60:.1f}m]", flush=True)

        if val_loss < best["val"] - 1e-6:
            best = {"val": val_loss, "epoch": epoch,
                    "state": {k: v.detach().clone() for k, v in net.state_dict().items()}}
        elif epoch - best["epoch"] >= tcfg["patience"]:
            break

    net.load_state_dict(best["state"])

    result = {
        "tag": tag, "fold": fold["fold"], "seed": seed, "feature_set": feature_set,
        "loss": loss_name, "n_features": len(pan.features), "n_params": n_params,
        "best_epoch": best["epoch"], "train_minutes": round((time.time() - t_start) / 60, 2),
        "n_train": len(idxs["train"]), "n_val": len(idxs["val"]), "n_test": len(idxs["test"]),
        "model": mcfg, "train_cfg": {k: v for k, v in tcfg.items()},
        "history": history,
    }

    preds = {}
    for split in ("val", "test"):
        p_s, y_s = predict(net, gw, idxs[split], tcfg["eval_batch_size"])
        p, y = norm.inverse_y(p_s), norm.inverse_y(y_s)
        result[split] = expected_return_metrics(p, y, C.COST_THRESHOLD)
        result[split].update(decile_monotonicity(p, y))
        preds[split] = pd.DataFrame({
            "timestamp": pan.timestamp[idxs[split]],
            "symbol": pan.symbol[idxs[split]],
            "predicted_log_return": p,
            "actual_log_return": y,
            "split": split, "fold": fold["fold"], "seed": seed,
            "feature_set": feature_set, "tag": tag,
        })

    out_dir = C.RUNS / tag / f"{feature_set}_{loss_name}_fold{fold['fold']}_seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.concat(preds.values(), ignore_index=True).to_parquet(out_dir / "predictions.parquet", index=False)
    (out_dir / "result.json").write_text(json.dumps(result, indent=2, default=str))
    torch.save(best["state"], out_dir / "model.pt")
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature-set", default="F2")
    ap.add_argument("--loss", default="huber")
    ap.add_argument("--folds", default="1,2,3,4,5")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--tag", default="stage1")
    ap.add_argument("--max-epochs", type=int, default=None)
    ap.add_argument("--context-len", type=int, default=None)
    ap.add_argument("--patch", default=None, help="patch_len:stride")
    ap.add_argument("--revin", default=None, choices=["on", "off"])
    ap.add_argument("--channel-attention", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()

    panel_df = pd.read_parquet(C.DATA_PROC / "panel.parquet")
    folds = {f["fold"]: f for f in make_folds(panel_df)}

    m_over, t_over = {}, {}
    if a.max_epochs:
        t_over["max_epochs"] = a.max_epochs
    if a.context_len:
        m_over["context_len"] = a.context_len
    if a.patch:
        p, s = a.patch.split(":")
        m_over.update(patch_len=int(p), stride=int(s))
    if a.revin:
        m_over["revin"] = a.revin == "on"
    if a.channel_attention:
        m_over["channel_attention"] = True

    if m_over.get("context_len") and m_over["context_len"] != C.CONTEXT_LEN:
        raise SystemExit("context_len ablation needs CONTEXT_LEN changed in config.py "
                         "(the window gather is sized from it)")

    rows = []
    for fold_id in [int(x) for x in a.folds.split(",")]:
        for seed in [int(x) for x in a.seeds.split(",")]:
            print(f"\n=== {a.tag} | {a.feature_set} | {a.loss} | fold {fold_id} | seed {seed} ===",
                  flush=True)
            r = run_one(panel_df, folds[fold_id], seed, a.feature_set, a.loss,
                        m_over, t_over, a.tag, verbose=not a.quiet)
            t = r["test"]
            print(f"  -> test  rmse {t['rmse']:.6f}  (zero {t['baseline_zero_rmse']:.6f})  "
                  f"pearson {t['pearson']:+.4f}  spearman {t['spearman']:+.4f}  "
                  f"dir {t['directional_accuracy']:.4f}  bias {t['prediction_bias']:+.2e}", flush=True)
            rows.append(r)

    summary = C.RUNS / a.tag / "summary.jsonl"
    summary.parent.mkdir(parents=True, exist_ok=True)
    with summary.open("a") as fh:
        for r in rows:
            fh.write(json.dumps(r, default=str) + "\n")


if __name__ == "__main__":
    main()
