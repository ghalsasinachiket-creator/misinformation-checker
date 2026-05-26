# misinformation-checker

A multimodal misinformation detector that flags misleading image-caption pairs using a fine-tuned CLIP model with GradCAM explainability. Built with PyTorch, HuggingFace Transformers, and the NewsCLIPpings dataset.

## Python API

```python
from misinformation_checker import CLIPMisinformationDetector

# model_name can point to your fine-tuned CLIP checkpoint
# e.g. a model fine-tuned on NewsCLIPpings
detector = CLIPMisinformationDetector(model_name="openai/clip-vit-base-patch32", threshold=0.35)
result = detector.predict(image=my_pil_image, caption="A caption to verify")

print(result.is_misleading)
print(result.misinformation_score)
print(result.gradcam_heatmap)  # GradCAM-like patch relevance map
```
