"""Student-t output head (PLAN.md section 14).

The model emits three raw numbers per sample; this module turns them into a
Student-t and back out into the quantities PLAN.md section 3 asks for.

Parameterisation, chosen against the failure mode STATUS.md flagged -- `df`
diverging on fat-tailed 4h crypto returns:

    loc   = raw0                          unconstrained, per sample
    scale = softplus(raw1) + 1e-3         strictly positive, floored, per sample
    df    = 1.1 + softplus(raw_df), <= 100

`df` is a single learned parameter shared by every sample by default, not a
per-sample output. That is a deliberate change from the obvious design, forced by
measurement:

  * A per-sample df collapsed onto its floor for 100% of validation samples
    through five epochs, at whatever floor was set (tried 2.0 and 1.1), while the
    NLL kept improving. Once softplus is driven negative its gradient vanishes, so
    the floor is absorbing -- df cannot climb back out.
  * The data does not want the floor. The unconditional MLE over all 556,910
    scaled targets is df = 2.123, and the NLL is flat between df 1.5 and 3.0
    (1.486 / 1.478 / 1.484). The collapse was an optimisation artifact, not a fit.
  * With one observation per row, a per-sample df is only identifiable through
    smoothness across samples. loc and scale carry the conditional information;
    the tail index is a property of the panel.

`--df-mode per_sample` restores the three-output version for anyone who wants to
re-test it. It is not the default because it does not work here.

The floor of 1.1 keeps the *mean* defined, which `expected_log_return` needs, and
allows the variance not to exist -- for df below 2 it does not. So nothing here
derives an `expected_volatility`: PLAN.md section 17 validates predicted
volatility against realized 4h volatility, a separate target with its own head in
multitask mode. Dispersion here is the predicted interquartile range, which exists
for any df. Capping at 100 stops drift toward the Gaussian limit where the NLL
surface is flat in df. Both bounds are recorded per run so it stays visible when
they bind.

Everything here operates in the normalizer's scaled space. The scaling is affine
and increasing, so loc, scale and every quantile invert with `inverse_y`; df is
scale-free and needs no inversion.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats

DF_MIN, DF_MAX = 1.1, 100.0
SCALE_FLOOR = 1e-3
DF_INIT = 2.5  # near the unconditional MLE of 2.12, inside the flat NLL region
N_PARAMS = {"global": 2, "per_sample": 3}


def inv_softplus(y: float) -> float:
    return math.log(math.expm1(y))


def df_param_init() -> float:
    return inv_softplus(DF_INIT - DF_MIN)


def unpack(raw: torch.Tensor, df_raw: torch.Tensor | None = None
           ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """raw [B, 2 or 3] -> (loc, scale, df), all [B].

    Three columns means a per-sample df. Two columns means df comes from the
    shared `df_raw` scalar parameter, broadcast."""
    loc = raw[:, 0]
    scale = F.softplus(raw[:, 1]) + SCALE_FLOOR
    src = raw[:, 2] if raw.shape[1] >= 3 else df_raw
    if src is None:
        raise ValueError("2-column output needs df_raw")
    df = (DF_MIN + F.softplus(src)).clamp(max=DF_MAX).expand_as(loc)
    return loc, scale, df


def head_bias_init(n: int) -> list[float]:
    """Bias so an untrained head starts at scale ~1 in scaled units.

    Targets are robust-scaled to roughly unit IQR, so scale ~1 is the right
    starting magnitude. Starting from softplus(0)=0.69 would put scale near its
    floor, where the log-likelihood gradient is steep and the first steps are
    unstable."""
    b = [0.0, inv_softplus(1.0 - SCALE_FLOOR)]
    return b + [df_param_init()] if n >= 3 else b


def nll(raw: torch.Tensor, y: torch.Tensor,
        df_raw: torch.Tensor | None = None) -> torch.Tensor:
    """Mean negative log likelihood. Written out rather than via
    torch.distributions so the lgamma terms stay in one graph and fp16 autocast
    does not silently promote them."""
    loc, scale, df = unpack(raw, df_raw)
    z = (y - loc) / scale
    log_p = (torch.lgamma((df + 1) / 2) - torch.lgamma(df / 2)
             - 0.5 * torch.log(df * math.pi) - torch.log(scale)
             - (df + 1) / 2 * torch.log1p(z * z / df))
    return -log_p.mean()


# CRPS is scored downstream from the predicted quantile grid, the same way the
# quantile GBM is scored, so the two are directly comparable. A closed form for
# the t would be faster but would not be measuring the same quantity.

# --------------------------------------------------------------- numpy side
def params_numpy(raw: np.ndarray, df_raw: float | None = None
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    softplus = lambda x: np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)
    loc = raw[:, 0].astype(np.float64)
    scale = softplus(raw[:, 1].astype(np.float64)) + SCALE_FLOOR
    src = raw[:, 2].astype(np.float64) if raw.shape[1] >= 3 else np.float64(df_raw)
    df = np.minimum(DF_MIN + softplus(src), DF_MAX) * np.ones_like(loc)
    return loc, scale, df


def derive(loc: np.ndarray, scale: np.ndarray, df: np.ndarray, quantiles, cost: float) -> dict:
    """PLAN.md section 3 outputs, all in whatever units loc/scale are given in.

    `cost` must already be in those same units."""
    out = {"expected_log_return": loc, "scale": scale, "df": df}
    for q in quantiles:
        out[f"p{int(round(q * 100)):02d}"] = loc + scale * stats.t.ppf(q, df)
    # no variance below df=2, so dispersion is the IQR, which always exists
    out["iqr"] = scale * (stats.t.ppf(0.75, df) - stats.t.ppf(0.25, df))
    out["prob_above_cost"] = stats.t.sf((cost - loc) / scale, df)
    return out
