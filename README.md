## Co-Boosting++
This repository contains the official implementation for the manuscript:
> Co-Boosting++: Coupled Optimization of Data and Ensemble for One-Shot Federated Learning

One-shot Federated Learning (OFL) has emerged as a promising paradigm, enabling global model training with minimal communication overhead. In OFL, the server model is usually distilled from an ensemble of pre-trained client models, while the ensemble also facilitates synthetic data generation for the knowledge distillation process. Prior works show that the performance of the final model is fundamentally tied to both the quality of the synthetic data and the ensemble. However, existing methods often optimize these two components separately, overlooking their interaction. To address this coupled optimization problem and provide a unified solution to the dual challenges of data and model heterogeneity inherent in OFL, we introduce Co-Boosting++, a novel OFL framework where synthetic data generation and ensemble construction mutually enhance each other in an iterative fashion. First, we fix the ensemble and generate hard samples in an adversarial manner. These samples are crucial for enhancing the robustness of knowledge transfer, as they challenge the model to generalize better, thereby improving quality of the synthetic data and subsequent distillation process. Second, leveraging these hard samples, we enhance the ensemble via a Mixture of Experts (MoE) mechanism. MoE allows dynamic adjustment of ensemble weights based on the generated hard samples, which enables the ensemble to better capture diverse and heterogeneous knowledge from client models. Furthermore, we extend Co-Boosting++ to support the simultaneous generation of multiple heterogeneous target models, enabling efficient adaptation to diverse device constraints. Extensive experiments on benchmark datasets demonstrate that Co-Boosting++ consistently outperforms state-of-the-art methods due to its coupled optimization of data and ensemble quality. Additionally, Co-Boosting++ is highly practical in real-world model market scenarios, requiring no local training modifications, additional transmissions, or restrictions on client model architectures.

## Experiments
The implementations of each method are provided in the folder /datafree/synthesis, while experiments are provided in the main folder.

Use corresponding python file to run the experiments.
> fl_pretrain.py
> 
> dfkd_trans.py
