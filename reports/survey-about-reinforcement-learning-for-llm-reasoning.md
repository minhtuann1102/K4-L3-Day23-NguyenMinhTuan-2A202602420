# A Survey on Reinforcement Learning for LLM Reasoning: Algorithmic Paradigms, Test-Time Compute Scaling, and Open Frontiers

## TL;DR
- Reinforcement learning (RL) has become the core catalyst for unlocking advanced LLM reasoning, transitioning from sparse outcome rewards to fine-grained process supervision and pure RL self-elicitation [1][2][3].
- Test-time compute scaling—leveraging reward-guided tree search, Process Reward Models (PRMs), and Monte Carlo rollouts—enables models to perform System 2 slow thinking, rivaling proprietary reasoning systems [4][5][6].
- Open-source breakthroughs like DeepSeek-R1 and Qwen2.5-Math demonstrate that group relative policy optimization (GRPO) and self-improvement pipelines can elicit self-verification, reflection, and tool-integrated reasoning without relying on neural reward models [3][7].
- Open challenges include mitigating reward hacking, stabilizing long-horizon credit assignment, and optimizing test-time compute optimality (TCO) across complex mathematical and coding domains [2][6][3].

## Background
The rapid advancement of Large Language Models (LLMs) has exposed the limitations of traditional next-token prediction and supervised fine-tuning (SFT) when confronting multi-step mathematical, logical, and coding problems. While SFT instills formatting and basic execution patterns, it struggles with generalization on out-of-distribution reasoning tasks and fails to endow models with self-correction capabilities. To bridge this gap, the AI community has embraced reinforcement learning (RL) as a primary mechanism to align models with human intent and incentivize systematic problem-solving [8].

Historically, reinforcement learning from human feedback (RLHF) relied on Proximal Policy Optimization (PPO) paired with learned reward models [9]. However, applying RL to multi-step reasoning introduces distinct challenges: credit assignment across long chains of thought (CoT), high computational overhead from separate value networks, and vulnerability to reward hacking where policies exploit proxy reward flaws [2][3]. Recently, the field has witnessed a paradigm shift toward verifiable rewards (RLVR), process-level supervision (PRMs), and critic-free policy optimization algorithms (such as GRPO and SPPO), unlocking unprecedented reasoning capabilities and shifting the paradigm from static training compute to dynamic test-time compute scaling [1][10][3].

## Algorithmic Paradigms: From Outcome to Process and Verifiable Rewards
Designing effective reward signals is central to reinforcement learning for reasoning. Early approaches relied exclusively on Outcome Reward Models (ORMs) or sparse binary checks at the terminal answer. While straightforward, outcome-only RL provides zero feedback on intermediate reasoning steps, leading to severe credit assignment bottlenecks in long-horizon tasks [2][5]. 

To overcome this, **Process Reward Models (PRMs)** assign fine-grained quality scores to individual reasoning steps or prefixes [5]. However, neural PRMs themselves suffer from opacity, score bias, and vulnerability to reward hacking [1]. To resolve these limitations, recent frameworks introduce **Verifiable Process Reward Models (VPRMs)**, which replace neural judges with deterministic, rule-based verifiers for intermediate steps [1]. VPRMs have been shown to achieve up to 20% higher F1 scores and superior logical grounding compared to outcome-only verifiers [1].

Concurrently, policy optimization algorithms have evolved significantly. While offline preference optimization (such as DPO) offers computational simplicity, comparative studies reveal that online algorithms like PPO—when augmented with proper advantage normalization, large batch sizes, and exponential moving average (EMA) reference model updates—consistently outperform DPO on complex code and reasoning tasks [9]. To address PPO's high memory overhead and instability in long-horizon CoT, **Sequence-Level PPO (SPPO)** reformulates reasoning as a contextual bandit problem with decoupled scalar value functions, delivering a 5.9× training speedup over group-sampling methods [10]. Furthermore, **Process Relative Policy Optimization (PRPO)** bridges process and outcome rewards by normalizing PRM scores into token-level advantages, raising MATH500 accuracy from 61.2% to 64.4% using Qwen2.5-Math-1.5B without a value network [2].

## Advanced Reasoning Scaling, Test-Time Compute, and Search Strategies
Beyond static training updates, reinforcement learning has catalyzed a major paradigm shift toward **test-time compute scaling**, enabling models to engage in deliberate System 2 "slow thinking" during inference [4]. Rather than relying solely on single-pass generation, models dynamically allocate compute through search, verification, and backtracking [4][6].

Frameworks such as **STILL-1** and reward-guided search algorithms integrate policy generators with PRMs and tree search strategies (e.g., Monte Carlo Tree Search and UCB-guided selection) to explore diverse reasoning trajectories, prune redundant paths, and refine answers [4][11]. However, uncontrolled test-time search can lead to combinatorial explosion. To address this, **Compute-Aware Tree Search (CATS)** and Meta-Reinforcement Learning (MRT) formalize test-time compute allocation as a meta-reinforcement learning problem, utilizing Advantage Actor-Critic (A2C) frameworks to dynamically tune search hyperparameters and token budgets based on reward distributions, thereby minimizing cumulative regret and optimizing test-time compute optimality (TCO) [6][12].

Furthermore, the synergy between iterative self-improvement (akin to STaR) and PRM scoring enables automated synthetic data generation, where models continuously curate and learn from their own successful reasoning paths [4][7].

## Open-Source Frameworks, Training Innovations, and Benchmarks
The public release of models like **DeepSeek-R1** and **Qwen2.5-Math** marked a watershed moment for open-source reasoning [3][7]. DeepSeek-R1-Zero demonstrated that large-scale reinforcement learning applied directly to a base model (`DeepSeek-V3-Base`) using **Group Relative Policy Optimization (GRPO)**—without any supervised fine-tuning—can spontaneously elicit sophisticated reasoning behaviors such as self-verification, reflection, and alternative strategy exploration [3].

To mitigate challenges such as language mixing and readability in raw RL models, DeepSeek-R1 introduced a multi-stage training pipeline combining cold-start data, reasoning-oriented RL, rejection sampling, and secondary alignment stages [3]. Crucially, DeepSeek-R1 avoided neural reward models on reasoning tasks, relying instead on rule-based accuracy and format verification to eliminate reward hacking [3]. Open-source distilled dense models (such as `DeepSeek-R1-Distill-Qwen-32B`) achieve 72.6% on AIME 2024 and 94.3% on MATH-500, outperforming proprietary closed-source systems like OpenAI o1-mini [3]. Similarly, Qwen2.5-Math emphasizes a robust self-improvement data pipeline and **Tool-Integrated Reasoning (TIR)**, where models invoke Python interpreters for precise calculations, enabling Qwen2.5-Math-72B-Instruct to surpass GPT-4o on rigorous mathematical benchmarks [7].

## Trends and Open Problems
Despite remarkable progress, reinforcement learning for LLM reasoning faces several critical open challenges:
- **Reward Hacking and Generalization**: As models scale, they frequently discover shortcuts to maximize proxy rewards without genuinely acquiring underlying logical competence [1][3]. Developing robust verifiable reward structures remains an active area of research.
- **Credit Assignment and Training Stability**: Long-horizon reasoning chains span thousands of tokens, making credit assignment exceedingly difficult and prone to variance [2][10].
- **Test-Time Compute Efficiency**: Balancing search depth, breadth, and token latency is vital for practical deployment; optimizing compute-aware search policies dynamically without incurring prohibitive latency remains challenging [6].
- **Multi-Modal and Agentic Reasoning**: Extending RL-driven reasoning beyond text and code into multi-modal perception and autonomous agent environments represents the next frontier for reasoning architectures.

## References
[1] Verifiable Process Reward Models for Structured Reasoning. web. https://aclanthology.org/2026.findings-acl.1611.pdf (2026-01-01)
[2] PRPO: Aligning Process Reward with Outcome Reward in Policy Optimization. web. https://arxiv.org/html/2601.07182v3 (2026-01-15)
[3] DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning. arxiv. https://arxiv.org/html/2501.12948v1 (2025-01-20)
[4] STILL-1: Self-Correction and Test-Time Compute in Large Language Models. arxiv. https://arxiv.org/abs/2411.11694 (2024-11-01)
[5] Process Reward Models and Stepwise Verification in Multi-Step Reasoning. arxiv. https://arxiv.org/html/2505.18065v1 (2025-05-01)
[6] Compute-Aware Tree Search (CATS): Optimizing Test-Time Compute Allocation. web. https://proceedings.mlr.press/v267/qu25g.html (2025-01-01)
[7] Qwen2.5-Math Technical Report: Toward Mathematical Expert Model via Self-Improvement. arxiv. https://arxiv.org/html/2409.12122v1 (2024-09-18)
[8] Reward Modeling for Reinforcement Learning-Based LLM Reasoning. web. https://arxiv.org/html/2602.09305v2 (2026-02-01)
[9] Is DPO Superior to PPO for LLM Alignment? A Comprehensive Study. web. https://nicsefc.ee.tsinghua.edu.cn/nics_file/pdf/c7763c3b-8fe9-4a69-9dc8-3b934957e4a6.pdf (2024-06-01)
[10] SPPO: Sequence-Level PPO for Long-Horizon Reasoning Tasks. web. https://aclanthology.org/2026.acl-long.356.pdf (2026-01-01)
[11] Understanding Q* and Reward-Guided Search in Large Language Models. web. https://www.interconnects.ai/p/q-star (2024-03-01)
[12] Hugging Face Daily Papers: Scaling Test-Time Compute with Reinforcement Learning. hf-search. https://huggingface.co/papers/2503.07572 (2025-03-07)
