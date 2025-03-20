# Enhancer Sequence Generation and Strength Prediction
## Overview | 概述  
This repository contains two related models for enhancer sequence tasks:  
本存储库包含两个用于增强子序列任务的相关模型： 


### 1. Enhancer Sequence Generation | 增强子序列生成  
- A **GAN-based model with attention mechanisms**, designed to generate novel enhancer sequences.  
- Combines **Transformer-based self-attention** with **Generative Adversarial Networks (GANs)** to enhance sequence diversity and optimize enhancer-like properties.  
- Hyperparameter tuning is conducted using **Ray Tune** to improve model stability and generation performance.  

- **基于 GAN 和注意力机制** 的增强子序列生成模型，旨在生成全新的增强子序列。  
- 结合 **Transformer 自注意力机制** 和 **生成对抗网络（GAN）** 以提升序列多样性并优化增强子特性。  
- 使用 **Ray Tune** 进行超参数优化，以提升模型稳定性和生成质量。  

### 2. Enhancer Strength Prediction | 增强子强度预测  
- A **DNABERT2-based model**, fine-tuned to predict enhancer activity from DNA sequences.  
- Leverages **pre-trained transformer embeddings** specific to genomic sequences for accurate enhancer strength classification.  
- Training is optimized with **multi-GPU acceleration (DeepSpeed)** and **custom data augmentation strategies**.  

- **基于 DNABERT2** 的增强子强度预测模型，针对 DNA 序列进行增强子活性预测。  
- 利用 **预训练的基因组 Transformer 表示** 进行高精度的增强子强度分类。  
- 采用 **多 GPU 训练加速（DeepSpeed）** 及 **自定义数据增强策略** 进行优化。  

---

## Key Features | 关键特性  
- **Multi-GPU training with DeepSpeed** for efficient large-scale training.  
- **Adaptive attention span mechanisms** for optimal sequence modeling.  
- **Ray Tune for hyperparameter optimization** and model selection.  
- **Custom data preprocessing pipelines** for handling enhancer sequences and experimental datasets.  

- **基于 DeepSpeed 的多 GPU 训练**，高效支持大规模训练。  
- **自适应注意力跨度机制**，优化序列建模能力。  
- **Ray Tune 超参数优化**，提升模型选择和泛化能力。  
- **自定义数据预处理管道**，适用于增强子序列和实验数据集。  

---

## Environment Requirements | 环境要求  
This package has been tested with the following configurations:  
该软件包已在以下环境配置下进行测试：  

- **Python 3.8+**  
- **PyTorch 2.0+**  
- **PyTorch Lightning 2.0+**  
- **DeepSpeed**  
- **Ray Tune**  
- **Pandas**  
- **NumPy**  

Higher versions of Python and PyTorch are also supported.  
更高版本的 Python 和 PyTorch 也兼容此软件包。  

---


## Code Structure | 代码结构  

### Training Scripts | 训练脚本  
- `train_generation.py` – Main script for training the **enhancer sequence generation** model (GAN-based).  
  - 用于训练 **增强子序列生成** 模型（基于 GAN）的主脚本。  
- `train_prediction.py` – Main script for training the **enhancer strength prediction** model (DNABERT2-based).  
  - 用于训练 **增强子强度预测** 模型（基于 DNABERT2）的主脚本。  

### Model Architectures | 模型架构  
- `model_generation.py` – Contains the **GAN + Transformer-based** architecture for enhancer sequence generation.  
  - 包含 **GAN + Transformer 结合** 的增强子序列生成模型架构。  
- `model_prediction.py` – Implements the **DNABERT2-based** model for enhancer strength prediction.  
  - 实现 **基于 DNABERT2** 的增强子强度预测模型。  

### Data Handling | 数据处理  
- `data_generation.py` – Implements the `EnhancerDataset` class for loading and preprocessing enhancer sequence data.  
  - 实现 `EnhancerDataset` 类，用于加载和预处理增强子序列数据。  
- `data_prediction.py` – Implements the `EnhancerPredictDataset` class for enhancer strength prediction tasks.  
  - 实现 `EnhancerPredictDataset` 类，用于增强子强度预测任务的数据加载。  

### Additional Components | 其他组件  
- `callbacks.py` – Includes custom callbacks such as **AdaptiveEarlyStopping** and **MultiMetricStopping**.  
  - 包含 **AdaptiveEarlyStopping** 和 **MultiMetricStopping** 等自定义回调函数。  
- `config.py` – Defines the **hyperparameter search space** for Ray Tune.  
  - 定义 **Ray Tune** 进行超参数搜索的配置文件。
  - 
---

## Usage | 使用方法  
To train the enhancer sequence generation model:  
训练增强子序列生成模型：  
```bash
python train_generation.py




## **Introduction**
This project aims to **generate and predict enhancer sequences** for **154 human tissues** using deep learning models. It includes:
1. **Enhancer Sequence Generation Model**: A **GAN-based generative model** with **attention mechanisms** to synthesize novel enhancer sequences.
2. **Enhancer Strength Prediction Model**: A **DNABERT2-based transformer model** to predict enhancer strength based on sequence information.

Both models leverage **multi-GPU distributed training** and **hyperparameter tuning** to optimize performance.

---

## **Features**
- **Generative model**: Combines **GAN and attention mechanisms** for sequence generation.
- **Prediction model**: Uses **DNABERT2 fine-tuning** for enhancer strength estimation.
- **Multi-GPU support**: Trained on **8 NVIDIA RTX 4090 GPUs** for efficiency.
- **Hyperparameter tuning**: **Optuna** for optimal architecture search.
- **Data scalability**: Supports 154 tissue-specific enhancer datasets.

---

## **Installation**

### **1. Clone the Repository**
```bash
git clone https://github.com/your_username/Enhancer-Gen-Predict.git
cd Enhancer-Gen-Predict

### **2. Set Up a Virtual Environment**

python3 -m venv enhancer_env
source enhancer_env/bin/activate  # For Linux/macOS
# OR
enhancer_env\Scripts\activate  # For Windows
### **3. Install Dependencies**
pip install -r requirements.txt
### **4. Verify GPU Availability**
import torch
print(torch.cuda.is_available())  # Should return True
print(torch.cuda.device_count())  # Check available GPUs
****
