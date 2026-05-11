"""Steering-vector target (self-hosted models only).

Trains a contrastive activation steering (CAA) or conditional
activation steering (ICLR 2025) vector that, when added to the
residual stream at the specified layers, *steers* the model away
from the cluster's failure mode. The resulting ``.safetensors``
payload can be loaded by the inference layer (vLLM, HF Transformers
with a custom hook, llama.cpp).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aisafepy.adapt.cluster import Cluster
from aisafepy.adapt.compile import CompiledArtifact


@dataclass
class _SteeringTarget:
    model: str
    layers: tuple[int, ...] = (16, 18)
    method: str = "conditional_activation_steering"
    out_dir: str = "guards/steering"
    kind: str = "steering"

    def compile_for_cluster(self, cluster: Cluster) -> CompiledArtifact | None:
        positives = [r.input + " " + r.output for r in cluster.records if r.was_violation]
        negatives = [r.input + " " + r.output for r in cluster.records if not r.was_violation]
        if len(positives) < 8:
            return None

        path: Any
        try:
            path = _train_steering_vector(
                model_id=self.model,
                layers=self.layers,
                positives=positives,
                negatives=negatives or positives[:4],  # fall back to self-contrast
                method=self.method,
                out_dir=Path(self.out_dir),
                cluster_label=cluster.label,
            )
        except ModuleNotFoundError as exc:
            return CompiledArtifact(
                kind=self.kind,
                name=f"steering-deferred-{cluster.label}",
                payload=None,
                cluster_label=cluster.label,
                attack_success_rate=cluster.attack_success_rate,
                n_records=cluster.size,
                summary=cluster.summary,
                metadata={
                    "deferred": True,
                    "reason": str(exc),
                    "training_spec": {
                        "model": self.model,
                        "layers": list(self.layers),
                        "method": self.method,
                        "positives": positives,
                        "negatives": negatives,
                    },
                },
            )

        return CompiledArtifact(
            kind=self.kind,
            name=f"steering-{cluster.label}",
            payload=str(path),
            cluster_label=cluster.label,
            attack_success_rate=cluster.attack_success_rate,
            n_records=cluster.size,
            summary=cluster.summary,
            metadata={"model": self.model, "layers": list(self.layers), "method": self.method},
        )


def _train_steering_vector(
    *,
    model_id: str,
    layers: tuple[int, ...],
    positives: list[str],
    negatives: list[str],
    method: str,
    out_dir: Path,
    cluster_label: int,
) -> Path:
    """Compute the difference of mean activations between positives and
    negatives (CAA-style), or fall back to a normalized contrast for
    "conditional" steering. The implementation mirrors the
    ``repeng`` / IBM activation-steering recipes; both have richer
    training loops, but the core math is just a mean-diff."""
    try:
        import torch  # type: ignore[import-not-found]
        from transformers import (  # type: ignore[import-not-found]
            AutoModelForCausalLM,
            AutoTokenizer,
        )
    except ImportError as exc:  # pragma: no cover - depends on env
        raise ModuleNotFoundError(
            "Steering requires `pip install aisafepy[adapt,probes]`."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, output_hidden_states=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()

    def mean_activations(texts: list[str]) -> torch.Tensor:
        all_layers = []
        for t in texts[:32]:  # cap for memory
            with torch.no_grad():
                enc = tokenizer(t, return_tensors="pt", truncation=True, max_length=256).to(device)
                out = model(**enc, output_hidden_states=True)
                per_layer = torch.stack(
                    [out.hidden_states[i].mean(dim=1).squeeze(0) for i in layers]
                )
                all_layers.append(per_layer.cpu())
        return torch.stack(all_layers).mean(dim=0)

    pos_means = mean_activations(positives)
    neg_means = mean_activations(negatives)
    vector = neg_means - pos_means  # subtract harmful direction
    if method == "conditional_activation_steering":
        # Normalize per-layer to enable conditional (gated) application.
        norms = vector.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        vector = vector / norms

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"steering_cluster_{cluster_label}.safetensors"
    try:
        from safetensors.torch import save_file  # type: ignore[import-not-found]

        save_file(
            {f"layer_{i}": vector[idx] for idx, i in enumerate(layers)},
            str(out_path),
        )
    except ImportError:
        out_path = out_dir / f"steering_cluster_{cluster_label}.pt"
        torch.save(
            {f"layer_{i}": vector[idx] for idx, i in enumerate(layers)},
            out_path,
        )
    return out_path
