# Enhancer Sequence Generation and Strength Prediction

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
