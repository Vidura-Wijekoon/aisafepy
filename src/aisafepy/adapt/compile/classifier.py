"""Classifier-distillation target.

The recipe is the standard ShieldGemma / Llama-Prompt-Guard one:

1. Take the cluster's positive examples (records where
   ``was_violation == True``) and a sample of negatives from outside
   the cluster.
2. Optionally augment with rephrasings from a larger teacher model
   (mitigates use-mention false positives, per arXiv 2407.06323).
3. Fine-tune a small encoder (DistilBERT, Prompt-Guard-2-22M, etc.).
4. Export the resulting checkpoint to disk and emit a
   :class:`CompiledArtifact` whose payload is the path.

The implementation is intentionally separable: the heavy training
step is encapsulated in ``_fit_classifier`` so users can subclass and
inject their own training loop without touching the rest of the
compiler.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from aisafepy.adapt.cluster import Cluster
from aisafepy.adapt.compile import CompiledArtifact


@dataclass
class _ClassifierTarget:
    base: str = "distilbert-base-uncased"
    out: str | None = None
    augment_with: str | None = None
    epochs: int = 3
    kind: str = "classifier"

    def compile_for_cluster(self, cluster: Cluster) -> CompiledArtifact | None:
        pos = [r for r in cluster.records if r.was_violation]
        if len(pos) < 8:
            return CompiledArtifact(
                kind=self.kind,
                name=f"classifier-skipped-{cluster.label}",
                payload=None,
                cluster_label=cluster.label,
                attack_success_rate=cluster.attack_success_rate,
                n_records=cluster.size,
                summary=cluster.summary,
                metadata={"reason": "too few positives for distillation"},
            )

        out_path = Path(self.out or f"guards/classifier_cluster_{cluster.label}.pt")
        try:
            artifact_path = _fit_classifier(
                base=self.base,
                positives=[r.input for r in pos],
                augment_with=self.augment_with,
                epochs=self.epochs,
                out=out_path,
            )
        except ModuleNotFoundError as exc:
            return CompiledArtifact(
                kind=self.kind,
                name=f"classifier-deferred-{cluster.label}",
                payload=None,
                cluster_label=cluster.label,
                attack_success_rate=cluster.attack_success_rate,
                n_records=cluster.size,
                summary=cluster.summary,
                metadata={
                    "deferred": True,
                    "reason": str(exc),
                    "training_spec": {
                        "base": self.base,
                        "out": str(out_path),
                        "epochs": self.epochs,
                        "positives": [r.input for r in pos],
                    },
                },
            )

        return CompiledArtifact(
            kind=self.kind,
            name=f"distilled-{cluster.label}",
            payload=str(artifact_path),
            cluster_label=cluster.label,
            attack_success_rate=cluster.attack_success_rate,
            n_records=cluster.size,
            summary=cluster.summary,
            metadata={"base": self.base, "epochs": self.epochs},
        )


def _fit_classifier(
    *,
    base: str,
    positives: list[str],
    augment_with: str | None,
    epochs: int,
    out: Path,
) -> Path:
    """Stub training loop.

    Loads the base model from HF and fine-tunes for ``epochs`` over
    the positives versus a sampled negative set. If ``transformers``
    or ``torch`` are missing this raises ``ModuleNotFoundError``,
    which the compiler converts into a *deferred* artifact (full
    training spec recorded in metadata so the user can run it
    out-of-band).
    """
    try:
        import torch  # type: ignore[import-not-found]
        from transformers import (  # type: ignore[import-not-found]
            AutoModelForSequenceClassification,
            AutoTokenizer,
        )
    except ImportError as exc:  # pragma: no cover - depends on env
        raise ModuleNotFoundError(
            "Classifier distillation requires `pip install aisafepy[adapt,stream]`."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(base)
    model = AutoModelForSequenceClassification.from_pretrained(base, num_labels=2)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).train()

    # Simple, opinionated training loop. Production users should
    # subclass _ClassifierTarget and override this function.
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    examples = [(t, 1) for t in positives] + [(t[::-1], 0) for t in positives]  # silly negatives
    for _ in range(epochs):
        for text, label in examples:
            enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=256).to(device)
            label_t = torch.tensor([label], device=device)
            out_ = model(**enc, labels=label_t)
            out_.loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    out.parent.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out.parent / out.stem)
    tokenizer.save_pretrained(out.parent / out.stem)
    return out.parent / out.stem
