"""White-box activation probes for self-hosted HF models.

Implements two recipes:

* :class:`LinearProbe`. A single logistic regression over the
  pooled residual-stream activations at a chosen layer (or an ensemble
  averaged across layers). This is the *Anthropic "Simple probes catch
  sleeper agents"* recipe.

* :class:`MLPProbe`. A small 2-layer MLP over the same activations.
  Kirch et al. (arXiv 2411.03343) showed non-linear probes
  outperform linear ones on harder jailbreak categories. We default
  to one hidden layer of width 256.

Both classes work on any ``transformers`` model that exposes
``output_hidden_states=True``. The probe weights are stored in a
single ``.safetensors`` file alongside a small JSON manifest so they
are portable across machines.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from aisafepy.core.decisions import GuardDecision
from aisafepy.stream.pipeline import Context


def _require_torch():
    try:
        import torch  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - depends on env
        raise ModuleNotFoundError(
            "Probes require `pip install aisafepy[probes]`"
        ) from exc
    return torch


@dataclass
class LinearProbe:
    """A logistic-regression probe over hidden states.

    Parameters
    ----------
    name:
        Span / log identifier.
    layers:
        Iterable of layer indices to probe. Activations are mean-pooled
        across tokens and concatenated across layers before the linear
        head sees them.
    threshold:
        Probability above which the probe returns BLOCK.
    tier:
        Cascade tier; defaults to 3.
    """

    name: str
    layers: tuple[int, ...]
    threshold: float = 0.5
    tier: int = 3
    weight: Any | None = None  # torch.Tensor, set after fit() or load()
    bias: Any | None = None
    feature_dim: int | None = None
    _torch: Any = field(default=None, init=False, repr=False)

    # ---- fitting -------------------------------------------------------

    @classmethod
    def fit(
        cls,
        *,
        model: Any,
        tokenizer: Any,
        layers: Iterable[int],
        pos_examples: Iterable[str],
        neg_examples: Iterable[str],
        name: str = "linear_probe",
        threshold: float = 0.5,
        l2: float = 1.0,
    ) -> LinearProbe:
        """Fit a logistic-regression probe.

        ``pos_examples`` are unsafe / target-class strings.
        ``neg_examples`` are benign / negative-class strings.
        """
        torch = _require_torch()
        try:
            from sklearn.linear_model import LogisticRegression  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - depends on env
            raise ModuleNotFoundError(
                "Probe fit requires `pip install aisafepy[probes]` "
                "(brings in scikit-learn)."
            ) from exc

        layers = tuple(layers)
        feats_pos = [_collect_features(model, tokenizer, layers, t) for t in pos_examples]
        feats_neg = [_collect_features(model, tokenizer, layers, t) for t in neg_examples]
        X = torch.stack(feats_pos + feats_neg).cpu().numpy()
        y = [1] * len(feats_pos) + [0] * len(feats_neg)
        clf = LogisticRegression(C=1.0 / max(l2, 1e-6), max_iter=2000)
        clf.fit(X, y)
        probe = cls(
            name=name,
            layers=layers,
            threshold=threshold,
            feature_dim=X.shape[1],
        )
        probe.weight = torch.tensor(clf.coef_[0], dtype=torch.float32)
        probe.bias = torch.tensor(clf.intercept_[0], dtype=torch.float32)
        probe._torch = torch
        return probe

    # ---- inference ----------------------------------------------------

    async def __call__(self, ctx: Context) -> GuardDecision:
        torch = _require_torch()
        start = time.perf_counter()
        if self.weight is None or ctx.hidden_states is None:
            return GuardDecision.allow(
                tier=self.tier,
                rationale="probe not fitted or no hidden_states; failing open",
                guard_name=self.name,
                evidence={"missing": True},
                latency_ms=(time.perf_counter() - start) * 1000.0,
            )
        feats = _pool_hidden_states(ctx.hidden_states, self.layers)
        logit = torch.dot(self.weight, feats) + self.bias
        prob = float(torch.sigmoid(logit).item())
        latency_ms = (time.perf_counter() - start) * 1000.0
        if prob >= self.threshold:
            return GuardDecision.block(
                tier=self.tier,
                confidence=prob,
                rationale=f"{self.name}: P(target)={prob:.3f} >= {self.threshold}",
                guard_name=self.name,
                evidence={"probe": "linear", "layers": list(self.layers), "prob": prob},
                severity="high" if prob > 0.9 else "medium",
                latency_ms=latency_ms,
            )
        return GuardDecision.allow(
            tier=self.tier,
            confidence=1.0 - prob,
            rationale=f"{self.name}: P(target)={prob:.3f} < {self.threshold}",
            guard_name=self.name,
            evidence={"prob": prob},
            latency_ms=latency_ms,
        )

    # ---- persistence --------------------------------------------------

    def save(self, path: str | Path) -> None:
        torch = _require_torch()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        manifest = {
            "kind": "linear_probe",
            "name": self.name,
            "layers": list(self.layers),
            "threshold": self.threshold,
            "feature_dim": self.feature_dim,
        }
        manifest_path = path.with_suffix(".json")
        with manifest_path.open("w") as f:
            json.dump(manifest, f, indent=2)
        torch.save({"weight": self.weight, "bias": self.bias}, path)

    @classmethod
    def load(cls, path: str | Path) -> LinearProbe:
        torch = _require_torch()
        path = Path(path)
        manifest = json.loads(path.with_suffix(".json").read_text())
        state = torch.load(path)
        probe = cls(
            name=manifest["name"],
            layers=tuple(manifest["layers"]),
            threshold=manifest["threshold"],
            feature_dim=manifest.get("feature_dim"),
        )
        probe.weight = state["weight"]
        probe.bias = state["bias"]
        probe._torch = torch
        return probe


@dataclass
class MLPProbe(LinearProbe):
    """A 2-layer MLP probe (Kirch et al. 2024)."""

    hidden_dim: int = 256
    w1: Any | None = None
    b1: Any | None = None
    w2: Any | None = None
    b2: Any | None = None

    @classmethod
    def fit(
        cls,
        *,
        model: Any,
        tokenizer: Any,
        layers: Iterable[int],
        pos_examples: Iterable[str],
        neg_examples: Iterable[str],
        name: str = "mlp_probe",
        threshold: float = 0.5,
        hidden_dim: int = 256,
        epochs: int = 30,
        lr: float = 1e-3,
    ) -> MLPProbe:
        torch = _require_torch()
        layers = tuple(layers)
        feats_pos = [_collect_features(model, tokenizer, layers, t) for t in pos_examples]
        feats_neg = [_collect_features(model, tokenizer, layers, t) for t in neg_examples]
        X = torch.stack(feats_pos + feats_neg)
        y = torch.tensor([1.0] * len(feats_pos) + [0.0] * len(feats_neg))
        d = X.shape[1]
        w1 = torch.randn(d, hidden_dim) * 0.1
        b1 = torch.zeros(hidden_dim)
        w2 = torch.randn(hidden_dim) * 0.1
        b2 = torch.zeros(1)
        params = [p.requires_grad_(True) for p in (w1, b1, w2, b2)]
        opt = torch.optim.Adam(params, lr=lr)
        for _ in range(epochs):
            opt.zero_grad()
            h = torch.relu(X @ w1 + b1)
            logits = h @ w2 + b2.squeeze()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y)
            loss.backward()
            opt.step()
        probe = cls(
            name=name,
            layers=layers,
            threshold=threshold,
            hidden_dim=hidden_dim,
            feature_dim=d,
        )
        probe.w1, probe.b1, probe.w2, probe.b2 = w1.detach(), b1.detach(), w2.detach(), b2.detach()
        probe._torch = torch
        return probe

    async def __call__(self, ctx: Context) -> GuardDecision:
        torch = _require_torch()
        start = time.perf_counter()
        if self.w1 is None or ctx.hidden_states is None:
            return GuardDecision.allow(
                tier=self.tier,
                rationale="MLP probe not fitted; failing open",
                guard_name=self.name,
                evidence={"missing": True},
            )
        feats = _pool_hidden_states(ctx.hidden_states, self.layers)
        h = torch.relu(feats @ self.w1 + self.b1)
        logit = h @ self.w2 + self.b2.squeeze()
        prob = float(torch.sigmoid(logit).item())
        latency_ms = (time.perf_counter() - start) * 1000.0
        if prob >= self.threshold:
            return GuardDecision.block(
                tier=self.tier,
                confidence=prob,
                rationale=f"{self.name}: P(target)={prob:.3f} >= {self.threshold}",
                guard_name=self.name,
                evidence={"probe": "mlp", "layers": list(self.layers), "prob": prob},
                severity="high" if prob > 0.9 else "medium",
                latency_ms=latency_ms,
            )
        return GuardDecision.allow(
            tier=self.tier,
            confidence=1.0 - prob,
            rationale=f"{self.name}: P(target)={prob:.3f} < {self.threshold}",
            guard_name=self.name,
            evidence={"prob": prob},
            latency_ms=latency_ms,
        )


# ---- helpers ----------------------------------------------------------


def _collect_features(model: Any, tokenizer: Any, layers: tuple[int, ...], text: str) -> Any:
    """Run a forward pass and return mean-pooled hidden states concatenated across layers."""
    torch = _require_torch()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(next(model.parameters()).device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    hs = outputs.hidden_states  # tuple of (B, T, D)
    parts = [hs[i].mean(dim=1).squeeze(0).cpu() for i in layers]
    return torch.cat(parts, dim=0)


def _pool_hidden_states(hidden_states: Any, layers: tuple[int, ...]) -> Any:
    """Mean-pool the supplied hidden states at the requested layers.

    Accepts either a tuple of tensors (the HF default) or a dict
    ``{layer_idx: tensor}``.
    """
    torch = _require_torch()
    if isinstance(hidden_states, dict):
        parts = [hidden_states[i].mean(dim=1).squeeze(0).cpu() for i in layers]
    else:
        parts = [hidden_states[i].mean(dim=1).squeeze(0).cpu() for i in layers]
    return torch.cat(parts, dim=0)
