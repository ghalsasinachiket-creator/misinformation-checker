from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class DetectionResult:
    similarity_score: float
    misinformation_score: float
    is_misleading: bool
    threshold: float
    gradcam_heatmap: Optional[Any]


class CLIPMisinformationDetector:
    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        threshold: float = 0.35,
        model: Optional[Any] = None,
        processor: Optional[Any] = None,
        device: Optional[str] = None,
    ) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be in [0, 1]")

        self.model_name = model_name
        self.threshold = threshold
        self.model = model
        self.processor = processor
        self.device = device

    def predict(self, image: Any, caption: str, with_explainability: bool = True) -> DetectionResult:
        similarity_score = self.compute_similarity(image=image, caption=caption)
        misinformation_score = max(0.0, min(1.0, 1.0 - similarity_score))
        gradcam_heatmap = None
        if with_explainability:
            gradcam_heatmap = self.generate_gradcam(image=image, caption=caption)

        return DetectionResult(
            similarity_score=similarity_score,
            misinformation_score=misinformation_score,
            is_misleading=misinformation_score >= self.threshold,
            threshold=self.threshold,
            gradcam_heatmap=gradcam_heatmap,
        )

    def compute_similarity(self, image: Any, caption: str) -> float:
        model, processor, torch = self._ensure_components()

        inputs = processor(images=image, text=[caption], return_tensors="pt", padding=True)
        if hasattr(inputs, "to") and self.device:
            inputs = inputs.to(self.device)

        with torch.no_grad():
            outputs = model(**inputs)

        similarity = torch.nn.functional.cosine_similarity(outputs.image_embeds, outputs.text_embeds).item()
        return float(max(-1.0, min(1.0, similarity)))

    def generate_gradcam(self, image: Any, caption: str) -> Optional[Any]:
        model, processor, torch = self._ensure_components()

        if not hasattr(model, "vision_model"):
            return None

        inputs = processor(images=image, text=[caption], return_tensors="pt", padding=True)
        if hasattr(inputs, "to") and self.device:
            inputs = inputs.to(self.device)

        outputs = model(**inputs, output_hidden_states=True, return_dict=True)
        vision_output = getattr(outputs, "vision_model_output", None)
        if vision_output is None:
            return None

        patch_activations = vision_output.last_hidden_state[:, 1:, :]
        if hasattr(patch_activations, "retain_grad"):
            patch_activations.retain_grad()

        score = torch.nn.functional.cosine_similarity(outputs.image_embeds, outputs.text_embeds).mean()
        model.zero_grad()
        score.backward()

        grads = getattr(patch_activations, "grad", None)
        if grads is None:
            return None

        weights = grads.mean(dim=1, keepdim=True)
        cam = (weights * patch_activations).sum(dim=-1)
        cam = torch.relu(cam)

        max_value = cam.max()
        if max_value.item() <= 0:
            return cam.squeeze(0).detach().cpu().numpy()

        cam = cam / max_value
        return cam.squeeze(0).detach().cpu().numpy()

    def _ensure_components(self) -> tuple[Any, Any, Any]:
        if self.model is None or self.processor is None:
            try:
                import torch  # type: ignore
                from transformers import CLIPModel, CLIPProcessor  # type: ignore
            except ImportError as exc:
                raise ImportError(
                    "PyTorch and transformers are required. Install dependencies to run the CLIP detector."
                ) from exc

            self.model = CLIPModel.from_pretrained(self.model_name)
            self.processor = CLIPProcessor.from_pretrained(self.model_name)
            if self.device and hasattr(self.model, "to"):
                self.model = self.model.to(self.device)
            elif hasattr(torch, "cuda") and torch.cuda.is_available() and hasattr(self.model, "to"):
                self.device = "cuda"
                self.model = self.model.to(self.device)
            else:
                self.device = "cpu"

        try:
            import torch  # type: ignore
        except ImportError as exc:
            raise ImportError("PyTorch is required to run the CLIP detector.") from exc

        return self.model, self.processor, torch
