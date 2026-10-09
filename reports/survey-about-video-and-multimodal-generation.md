# A Comprehensive Survey on Video and Multimodal Generation: Architectures, Conditioning Paradigms, Benchmarks, and Future Horizons

## TL;DR (Executive Summary)
- **Architectural Paradigm Shift:** Video generation has undergone a profound evolution from early 3D U-Net convolutional architectures to scalable Diffusion Transformer (DiT) backbones operating on space-time latent patches [1], enabling models like Sora [2] and Open-Sora [3] to function as emergent physics and world simulators.
- **Multimodal & Controllable Conditioning:** Advanced generative pipelines rely on factorized conditioning (such as text-to-image initialization followed by joint spatial-temporal video diffusion [4]) and specialized control mechanisms (including structural guidance, subject identity preservation, and cross-modal audio-visual synchronization [5][6]).
- **Evaluation & Benchmarking Revolution:** Fréchet Video Distance (FVD) has been augmented by robust benchmarking suites (EvalCrafter [7], VBench [8]) and automated MLLM-based scoring systems (VideoScore [8]) that align closely with human perception across multiple fine-grained dimensions.
- **Efficiency & Scalability Frontiers:** To combat the quadratic spatio-temporal token explosion, novel efficiency techniques—such as Generation-Aware Latent Compression (GRACE) [9] and runtime-adaptive caching—achieve order-of-magnitude latency reductions without sacrificing visual fidelity.

## Background
Video and multimodal generation represents one of the most rapidly advancing frontiers in artificial intelligence, bridging computer vision, natural language processing, audio synthesis, and physical reasoning. Historically, early generative video models extended 2D convolutional generative adversarial networks (GANs) and recurrent neural networks (RNNs), struggling heavily with temporal jitter, short horizons, and low resolutions. The breakthrough of Latent Diffusion Models (LDMs) established a robust foundation for high-fidelity image synthesis, which researchers swiftly extended to video via factorized 3D U-Nets [1]. 

The significance of video generation extends far beyond entertainment and media creation; advanced generative models serve as foundational "world simulators" capable of understanding and predicting complex physical interactions, gravity, object permanence, and human motion dynamics from massive web-scale corpora [1][2]. Foundational milestones—including OpenAI's Sora [2], Stable Video Diffusion, and open-source initiatives like Open-Sora [3]—demonstrate that scaling model parameter sizes alongside high-capacity transformer backbones unlocks powerful emergent capabilities. Concurrently, multimodal conditioning frameworks have expanded video generation from purely text-driven prompts to rich interactive inputs encompassing reference images, skeletal poses, depth maps, audio waveforms, and subject identities [5][6].

## Architectural Foundations: From 3D U-Nets to Diffusion Transformers (DiTs)
The architectural bedrock of video generation has shifted dramatically over recent years. Early video diffusion models (e.g., VDM, Imagen Video, MagicVideo) relied on 3D U-Net architectures, augmenting standard 2D spatial convolutions with factorized temporal convolutions and attention layers [1]. While effective for low-resolution generation, 3D U-Nets exhibit inherent bottlenecks in scaling parameter count and handling long-range dependencies.

In response, modern foundational models have universally adopted Diffusion Transformer (DiT) backbones [1]. Pioneered by architectures like Sora [2], Open-Sora [3], Latte, and W.A.L.T, DiTs replace convolutional encoder-decoder backbones with transformer blocks operating directly on space-time patch tokens extracted from compressed video latents. This design allows video models to leverage the exact scaling laws observed in large language models (LLMs). Furthermore, autoregressive video generation approaches (such as Phenaki and VideoPoet) utilize Causal Video Vision Transformers (C-ViViT) to predict discrete visual tokens autoregressively over compressed latent spaces.

To manage the massive computational overhead of high-resolution video, spatial-temporal compression is critical. Standard video autoencoders compress input frames into compact latent manifolds ($8\times$ spatial and $4\times$ temporal compression), which are then processed via factorized or pseudo-3D attention mechanisms that decouple spatial self-attention from temporal 1D attention [1].

## Multimodal Conditioning, Factorized Pipelines, and Controllable Generation
User intent in real-world creative and simulation workflows rarely relies on text prompts alone. A comprehensive taxonomy of controllable video generation categorizes mechanisms into six major dimensions and seven primary control modalities: Structure Control (poses, depth maps, bounding boxes), ID Control (subject identity preservation), Image Control (reference frames), Temporal Control (camera trajectories and optical flow), Audio Control (voice-to-video synchronization), Other Control (text rendering and style), and Universal Multi-Condition Integration [5].

To resolve the training instability and compounding errors of deep super-resolution cascades, Emu Video introduced a factorized generation paradigm [4]. By explicitly factorizing text-to-video into two steps—generating a pristine keyframe image conditioned on the text prompt, followed by a joint text-and-image conditioned video diffusion process—Emu Video bypassed complex super-resolution cascades, achieving superior alignment and temporal consistency [4]. Similarly, advanced human-centric frameworks like HuMo integrate collaborative multi-modal conditioning to harmonize subject identity preservation and audio-visual synchronization [6].

Controllable generation extensions (such as ControlNet adaptations and auxiliary lightweight encoders like SparseCtrl and CTRL-Adapter) allow practitioners to inject fine-grained spatial and motion guidance into frozen pretrained diffusion backbones without disrupting their underlying distribution priors [5].

## Evaluation Methodologies, Benchmarks, and Automated Quality Scoring
Evaluating video generation models has historically been plagued by the limitations of distribution-level metrics like Fréchet Video Distance (FVD) and Inception Score (IS), which fail to capture motion smoothness, physical realism, or semantic text alignment. 

To overcome this, modern research has established rigorous, multi-faceted benchmarking suites:
- **EvalCrafter (CVPR 2024):** Evaluates large text-to-video models across 700 diverse prompts spanning realistic and stylized domains, utilizing 17 objective metrics and linear regression-based human alignment [7].
- **VBench:** A hierarchical benchmark suite assessing video generative models across 16+ fine-grained dimensions, including subject consistency, background consistency, motion smoothness, dynamic degree, and intrinsic physical faithfulness [8].
- **VideoScore (EMNLP 2024):** Introduces VideoFeedback (37.6K expert-annotated video evaluations across 5 key dimensions). Initialized from Mantis-Idefics2-8B, VideoScore achieves a high Spearman correlation (77.1) on human feedback, outperforming GPT-4o and Gemini 1.5 Pro by over 50 points, thus serving as an automated proxy for RLHF and Best-of-K sampling [8].

## Efficiency Challenges, Latent Compression, and Inference Acceleration
The deployment of video generation models is bottlenecked by severe computational constraints. A 90-minute video sequence yields approximately 54 million tokens, creating massive memory footprints and quadratic attention scaling during iterative denoising.

While standard video autoencoders compress spatial dimensions, increasing compression ratios beyond $8\times$ often degrades reconstruction quality and shifts latent distributions away from pretrained DiT priors. To address this, Generation-Aware Latent Compression (GRACE) introduces a two-stage framework that compresses pretrained video autoencoders up to $16\times$ spatial and $8\times$ temporal by learning residual latents and aligning feature spaces with frozen DiTs, achieving an 8x reduction in token count and an 11.1$\times$ to 15.5$\times$ speedup while preserving VBench quality [9]. Complementary plug-and-play inference acceleration techniques—such as EasyCache, which exploits runtime-adaptive transformation vector caching—further reduce per-step computational overhead without requiring model retraining.

## Trends and Open Problems
Despite monumental progress, several fundamental challenges remain unresolved:
1. **Long-Form Video Consistency & Physics Reasoning:** Maintaining object persistence, semantic identity, and commonsense physical reasoning over extended temporal horizons (beyond a few seconds) remains difficult, as models frequently suffer from drift and physical implausibility.
2. **World Simulator Gap:** Stress-testing frameworks reveal a persistent gap between surface-level visual plausibility and true predictive world-state reasoning [2][9].
3. **Agent-Based and Iterative RLHF:** Moving beyond static test sets toward dynamic evaluation agents and automated scoring models (such as VideoScore [8]) will be essential for continuous alignment and reinforcement learning in video synthesis.

## References
[1] Foundations and Architectural Innovations in Video Diffusion and Autoregressive Models. arxiv. https://arxiv.org/abs/2405.03150 (2024-05-06)
[2] Video generation models as world simulators. web. https://openai.com/index/video-generation-models-as-world-simulators/ (2024-02-15)
[3] Open-Sora: Democratizing Efficient Video Production for All. arxiv. https://arxiv.org/html/2403.05131v3 (2024-03-08)
[4] Emu Video: Factorizing Text-to-Video Generation by Explicit Image Conditioning. hf-search. https://huggingface.co/papers/2311.10709 (2023-11-17)
[5] A Comprehensive Survey on Controllable Video Generation: Mechanisms, Architectures, and Frameworks. web. https://arxiv.org/abs/2507.16869 (2026-01-19)
[6] HuMo: Human-Centric Video Generation via Collaborative Multi-Modal Conditioning. hf-search. https://huggingface.co/papers/2509.08519 (2025-09-10)
[7] EvalCrafter: Benchmarking and Evaluating Large Video Generation Models. arxiv. https://arxiv.org/abs/2310.11440 (2024-04-10)
[8] VBench: Comprehensive Benchmark Suite for Video Generative Models. arxiv. https://arxiv.org/abs/2406.15252 (2024-06-20)
[9] Generation-Aware Latent Compression for Efficient Video Diffusion Transformers. arxiv. https://arxiv.org/abs/2610.10524 (2026-10-10)
