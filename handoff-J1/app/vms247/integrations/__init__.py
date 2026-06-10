"""
Lớp tích hợp model designated (Phase 1b) — chạy trên CLOUD.

- vlm_client.VLMClient      : LocateAnything-3B qua vLLM (OpenAI-compatible API). CODE THẬT.
- wholebody.Wholebody49     : adapter DEIMv2-Wholebody49 (ONNX/TensorRT). ĐIỂM CẮM tích hợp.

Cloud serving do Lead dựng — khi tới Phase 1b sẽ cấp URL endpoint, điền vào configs.
"""
