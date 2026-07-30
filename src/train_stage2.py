"""Stage two: distributional and multi-task PatchTST (PLAN.md sections 14, 20).

Two modes, sharing everything else with stage one -- same folds, same purging,
same train-only normalization, same encoder config:

  studentt   the return head only, emitting (loc, scale, df) trained by NLL.
             Eligibility matches stage one exactly, so its return predictions are
             directly comparable to the Huber runs.
  multitask  the same encoder with five heads (PLAN.md section 20):
             Student-t return, volatility, MAE, MFE, regime, combined as
             L = 1.0 L_ret + 0.25 L_vol + 0.15 L_mae + 0.15 L_mfe + 0.20 L_reg

Both early-stop on validation *return* NLL rather than the combined loss. PLAN.md
section 20 keeps auxiliary tasks only when they do not degrade the return output,
so the checkpoint selected must be the best return model, not the best compromise.
The combined loss is recorded alongside it so the trade-off stays visible.

The five heads are one Linear over the shared representation, sliced -- which is
exactly equivalent to five separate Linear heads on the same input, with fewer
moving parts. Note this makes the head 9 outputs wide: at 138,880 features that is
1.25M head parameters against the encoder's ~600k, so the multi-task model is
roughly 2.5x the size of the stage-one model. That is a property of the flatten
head, not of the extra tasks, and is the strongest argument for the deferred
head="mean" ablation.
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
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C
from src import dist
from src import model as M
from src.datamodule import GPUWindows, MultiPanel, TargetNormalizer
from src.splits import make_folds
from src.train import set_seed, shuffle_target

# name -> (slice width, loss weight). Order fixes the output layout.
# The return width depends on --df-mode, so it is filled in by layout().
TASK_WEIGHT = {"return": 1.00, "volatility": 0.25, "mae": 0.15, "mfe": 0.15, "regime": 0.20}
TASK_WIDTH = {"volatility": 1, "mae": 1, "mfe": 1, "regime": len(C.REGIME_CLASSES)}
TARGET_COL = {"return": "y_return", "volatility": "y_volatility", "mae": "y_mae",
              "mfe": "y_mfe", "regime": "y_regime"}
QUANTILES = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95]


def layout(mode: str, df_mode: str) -> dict[str, slice]:
    names = ["return"] if mode == "studentt" else list(TASK_WEIGHT)
    out, i = {}, 0
    for n in names:
        w = dist.N_PARAMS[df_mode] if n == "return" else TASK_WIDTH[n]
        out[n] = slice(i, i + w)
        i += w
    return out


def compute_losses(raw: torch.Tensor, y: torch.Tensor, sl: dict[str, slice],
                   tgts: list[str], df_raw: torch.Tensor | None) -> dict[str, torch.Tensor]:
    col = {t: i for i, t in enumerate(tgts)}
    out = {"return": dist.nll(raw[:, sl["return"]], y[:, col["y_return"]], df_raw)}
    for name in ("volatility", "mae", "mfe"):
        if name in sl:
            out[name] = F.huber_loss(raw[:, sl[name]].squeeze(-1), y[:, col[TARGET_COL[name]]],
                                     delta=1.0)
    if "regime" in sl:
        out["regime"] = F.cross_entropy(raw[:, sl["regime"]],
                                        y[:, col["y_regime"]].long())
    return out


@torch.no_grad()
def predict(net, gw: GPUWindows, idx: np.ndarray, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    net.eval()
    P, Y = [], []
    t = torch.from_numpy(idx).to(gw.device)
    for i in range(0, len(t), batch_size):
        xb, yb = gw.gather(t[i: i + batch_size])
        P.append(net(xb).float().cpu())
        Y.append(yb.float().cpu())
    return torch.cat(P).numpy(), torch.cat(Y).numpy()


def run_one(panel_df: pd.DataFrame, fold: dict, seed: int, feature_set: str, mode: str,
            model_over: dict | None = None, train_over: dict | None = None,
            tag: str = "stage2", device: str = "cuda", verbose: bool = True,
            df_mode: str = "global") -> dict:
    mcfg = {**C.MODEL, **(model_over or {})}
    tcfg = {**C.TRAIN, **(train_over or {})}
    dev = torch.device(device)
    set_seed(seed)

    sl = layout(mode, df_mode)
    tgts = [TARGET_COL[n] for n in sl]
    pan = MultiPanel(panel_df, feature_set, tgts)
    idxs = pan.fold_indices(fold)
    if min(len(v) for v in idxs.values()) == 0:
        raise RuntimeError(f"empty split in fold {fold['fold']}")

    from src.datamodule import Normalizer
    xnorm = Normalizer(pan.X, pan.y[:, 0], idxs["train"])  # feature stats only
    tnorm = TargetNormalizer(pan.y, idxs["train"], tgts)
    gw = GPUWindows(xnorm.transform_X(pan.X), tnorm.transform(pan.y), dev)

    n_out = sl[list(sl)[-1]].stop
    net = M.build(len(pan.features), mcfg, n_outputs=n_out).to(dev)
    width = sl["return"].stop - sl["return"].start
    # start the distribution at scale ~1 rather than at the softplus floor
    with torch.no_grad():
        net.head[-1].bias[sl["return"]] = torch.tensor(dist.head_bias_init(width), device=dev)
    # shared tail index, learned jointly; see src/dist.py for why it is not per-sample
    df_raw = None
    if df_mode == "global":
        df_raw = torch.nn.Parameter(torch.tensor(dist.df_param_init(), device=dev))
    n_params = sum(p.numel() for p in net.parameters()) + (0 if df_raw is None else 1)

    stride = max(1, int(tcfg["sample_stride"]))
    per_epoch = max(1, int(np.ceil(len(idxs["train"]) / stride / tcfg["batch_size"])))
    # df gets no weight decay -- decaying it pulls the tail index toward the floor
    groups = [{"params": list(net.parameters()), "weight_decay": tcfg["weight_decay"]}]
    if df_raw is not None:
        groups.append({"params": [df_raw], "weight_decay": 0.0})
    opt = torch.optim.AdamW(groups, lr=tcfg["lr"])
    steps = per_epoch * tcfg["max_epochs"]
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=tcfg["lr"], total_steps=steps,
                                                pct_start=0.15)
    scaler = torch.amp.GradScaler("cuda", enabled=tcfg["amp"])
    gen = torch.Generator(device=dev).manual_seed(seed)

    best = {"val": np.inf, "epoch": -1, "state": None}
    history = []
    t_start = time.time()

    for epoch in range(tcfg["max_epochs"]):
        net.train()
        tot, nb = 0.0, 0
        epoch_idx = idxs["train"][epoch % stride:: stride]
        for xb, yb in gw.iterate(epoch_idx, tcfg["batch_size"], shuffle=True, generator=gen):
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=tcfg["amp"]):
                losses = compute_losses(net(xb), yb, sl, tgts, df_raw)
                loss = sum(TASK_WEIGHT[k] * v for k, v in losses.items())
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            clip_params = list(net.parameters()) + ([df_raw] if df_raw is not None else [])
            nn.utils.clip_grad_norm_(clip_params, tcfg["grad_clip"])
            scaler.step(opt)
            scaler.update()
            if sched.last_epoch < steps - 1:
                sched.step()
            tot += loss.item(); nb += 1

        vp, vy = predict(net, gw, idxs["val"], tcfg["eval_batch_size"])
        df_cpu = None if df_raw is None else df_raw.detach().cpu()
        vl = compute_losses(torch.from_numpy(vp), torch.from_numpy(vy), sl, tgts, df_cpu)
        val_nll = float(vl["return"])           # selection criterion
        val_total = float(sum(TASK_WEIGHT[k] * v for k, v in vl.items()))
        df_val = None if df_raw is None else float(df_raw.detach().cpu())
        _, vs, vdf = dist.params_numpy(vp[:, sl["return"]], df_val)
        row = {"epoch": epoch, "train_loss": tot / nb, "val_return_nll": val_nll,
               "val_total": val_total, "df_median": float(np.median(vdf)),
               "df_at_floor": float(np.mean(vdf <= dist.DF_MIN + 1e-3)),
               "scale_median": float(np.median(vs)),
               "minutes": round((time.time() - t_start) / 60, 2)}
        row.update({f"val_{k}": float(v) for k, v in vl.items()})
        history.append(row)
        if verbose:
            extra = "  ".join(f"{k} {float(v):.4f}" for k, v in vl.items() if k != "return")
            print(f"    epoch {epoch:2d}  train {tot/nb:.5f}  val_nll {val_nll:.5f}  "
                  f"df {row['df_median']:.2f} (floor {100*row['df_at_floor']:.0f}%)  "
                  f"sig {row['scale_median']:.3f}  {extra} "
                  f"[{(time.time()-t_start)/60:.1f}m]", flush=True)

        if val_nll < best["val"] - 1e-6:
            best = {"val": val_nll, "epoch": epoch,
                    "state": {k: v.detach().clone() for k, v in net.state_dict().items()},
                    "df_raw": None if df_raw is None else df_raw.detach().clone()}
        elif epoch - best["epoch"] >= tcfg["patience"]:
            break

    net.load_state_dict(best["state"])
    if df_raw is not None:
        with torch.no_grad():
            df_raw.copy_(best["df_raw"])

    result = {
        "tag": tag, "mode": mode, "fold": fold["fold"], "seed": seed,
        "feature_set": feature_set, "n_features": len(pan.features), "n_params": n_params,
        "n_outputs": n_out, "df_mode": df_mode, "best_epoch": best["epoch"],
        "train_minutes": round((time.time() - t_start) / 60, 2),
        "n_train": len(idxs["train"]), "n_val": len(idxs["val"]), "n_test": len(idxs["test"]),
        "model": mcfg, "train_cfg": dict(tcfg), "history": history,
    }

    # P(y > cost) is evaluated in scaled space, so the threshold moves with it.
    # y_return is robust-scaled, not log-transformed, so this is a plain affine map.
    cost_scaled = float((C.COST_THRESHOLD - tnorm.med[0]) / tnorm.scale[0])
    frames = {}
    for split in ("val", "test"):
        raw, y = predict(net, gw, idxs[split], tcfg["eval_batch_size"])
        loc, scale, df = dist.params_numpy(raw[:, sl["return"]], df_val)
        d = dist.derive(loc, scale, df, QUANTILES, cost_scaled)

        out = pd.DataFrame({
            "timestamp": pan.timestamp[idxs[split]],
            "symbol": pan.symbol[idxs[split]],
            "predicted_log_return": tnorm.inverse(d["expected_log_return"], "y_return"),
            "actual_log_return": tnorm.inverse(y[:, 0], "y_return"),
            "pred_scale": tnorm.inverse_scale(d["scale"], "y_return"),
            "pred_df": d["df"],
            "pred_iqr": tnorm.inverse_scale(d["iqr"], "y_return"),
            "p_above_cost": d["prob_above_cost"],
        })
        for q in QUANTILES:
            k = f"p{int(round(q * 100)):02d}"
            out[f"q{int(round(q * 100)):02d}"] = tnorm.inverse(d[k], "y_return")

        col = {t: i for i, t in enumerate(tgts)}
        for name in ("volatility", "mae", "mfe"):
            if name in sl:
                t = TARGET_COL[name]
                out[f"pred_{name}"] = tnorm.inverse(raw[:, sl[name]].squeeze(-1), t)
                out[f"actual_{name}"] = tnorm.inverse(y[:, col[t]], t)
        if "regime" in sl:
            p = torch.softmax(torch.from_numpy(raw[:, sl["regime"]]), dim=1).numpy()
            for i, cname in enumerate(C.REGIME_CLASSES):
                out[f"p_{cname}"] = p[:, i]
            out["actual_regime"] = y[:, col["y_regime"]].astype(int)

        out["split"], out["fold"], out["seed"] = split, fold["fold"], seed
        out["mode"], out["feature_set"], out["tag"] = mode, feature_set, tag
        frames[split] = out

        result[split] = {
            "n": len(out),
            "return_nll": float(compute_losses(torch.from_numpy(raw), torch.from_numpy(y),
                                               sl, tgts, df_cpu)["return"]),
            "df_median": float(np.median(df)),
            "df_at_floor": float(np.mean(df <= dist.DF_MIN + 1e-3)),
            "df_at_cap": float(np.mean(df >= dist.DF_MAX - 1e-3)),
            "scale_at_floor": float(np.mean(scale <= dist.SCALE_FLOOR * 1.01)),
        }

    out_dir = C.RUNS / tag / f"{feature_set}_{mode}_fold{fold['fold']}_seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.concat(frames.values(), ignore_index=True).to_parquet(
        out_dir / "predictions.parquet", index=False)
    (out_dir / "result.json").write_text(json.dumps(result, indent=2, default=str))
    torch.save(best["state"], out_dir / "model.pt")
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="studentt", choices=["studentt", "multitask"])
    ap.add_argument("--feature-set", default="F2")
    ap.add_argument("--folds", default="1,2,3,4,5")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--tag", default=None, help="defaults to the mode name")
    ap.add_argument("--max-epochs", type=int, default=None)
    ap.add_argument("--shuffle-target", default="none", choices=["none", "iid", "shift"])
    ap.add_argument("--shuffle-seeds", default="0")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    tag = a.tag or a.mode

    panel_df = pd.read_parquet(C.DATA_PROC / "panel.parquet")
    folds = {f["fold"]: f for f in make_folds(panel_df)}
    t_over = {"max_epochs": a.max_epochs} if a.max_epochs else {}
    shuf = [int(x) for x in a.shuffle_seeds.split(",")] if a.shuffle_target != "none" else [None]

    rows = []
    for s in shuf:
        df = panel_df if s is None else shuffle_target(panel_df, a.shuffle_target, s)
        for fold_id in [int(x) for x in a.folds.split(",")]:
            for seed in [int(x) for x in a.seeds.split(",")]:
                lbl = "" if s is None else f" | shuffle {a.shuffle_target} draw {s}"
                print(f"\n=== {tag} | {a.mode} | {a.feature_set} | fold {fold_id} | "
                      f"seed {seed}{lbl} ===", flush=True)
                t = tag if s is None else f"{tag}/{a.shuffle_target}_draw{s}"
                r = run_one(df, folds[fold_id], seed, a.feature_set, a.mode,
                            train_over=t_over, tag=t, verbose=not a.quiet)
                print(f"  -> test nll {r['test']['return_nll']:.5f}  "
                      f"df_med {r['test']['df_median']:.2f}  "
                      f"df_floor {100*r['test']['df_at_floor']:.1f}%  "
                      f"df_cap {100*r['test']['df_at_cap']:.1f}%  "
                      f"[{r['train_minutes']:.1f}m]", flush=True)
                rows.append(r)

    summary = C.RUNS / tag / "summary.jsonl"
    summary.parent.mkdir(parents=True, exist_ok=True)
    with summary.open("a") as fh:
        for r in rows:
            fh.write(json.dumps(r, default=str) + "\n")


if __name__ == "__main__":
    main()
