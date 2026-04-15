Markdown
# RL-Gymnasium-PACSPL602013 Final Project

## 🛠️ 1. 开发者原始配置 (Developer's Reference)
* **操作系统**: Windows 11
* **显卡 (GPU)**: NVIDIA GeForce **RTX 3070** (8GB VRAM)
* **处理器 (CPU)**: Intel Core **i7-12700H**
* **Python 版本**: **3.12.x** (注：建议避开 3.13 以免 CUDA 库不兼容)
* **虚拟环境名**: `final_project_env`

---

## 🚀 2. 环境重建步骤 (Step-by-Step Setup)

### 第一步：创建并激活环境
> **建议安装 Python 3.12 后执行**

```powershell
python -m venv final_project_env
.\final_project_env\Scripts\activate
```

### 第二步：安装 GPU 加速版 PyTorch
> 这是确保 RTX 3070 发挥性能的关键。如果你没有 NVIDIA 显卡，可以跳过此步直接执行第三步（将使用 CPU 运行）。

```PowerShell
# 针对 CUDA 12.1 的安装指令
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)
```

### 第三步：一键安装项目依赖
```PowerShell
pip install -r requirements.txt
```

### 第四步：本地 VLM (Ollama) 配置
> 本项目创新部分依赖本地大模型推理，请确保安装了 Ollama 并准备好模型：

```PowerShell
ollama pull llava:7b
```

## ⚠️ 注意事项 (Important Notes)
Git 忽略: 请勿将 final_project_env/ 文件夹上传至 Git 仓库，该目录已包含在 .gitignore 中。

模型路径: 训练产生的模型和日志将存放在 logs_and_results/ 文件夹下。

Kernel 选择: 在 VS Code 中打开 .ipynb 文件时，请务必在右上角选择 final_project_env (Python 3.12.x) 作为内核。