# 📊 Sentiment Analysis Model - SA Assignment

这是一个基于 PyTorch 的情感分析模型项目，旨在对文本进行情感分类（如正面 / 负面）。该项目用于自然语言处理（NLP）课程作业或相关实验。

---

## 📁 项目结构

```
sa_assignment/
├── data/                # 存放训练和测试数据
├── sa_model.py          # 模型定义
├── train.py             # 训练脚本
├── evaluate.py          # 模型评估脚本
├── utils.py             # 工具函数（如数据处理）
├── model.pt             # 训练好的模型权重
├── requirements.txt     # Python 依赖项
└── README.md            # 项目说明文档
```

---

## 🚀 快速开始

### ✅ 1. 克隆仓库

```bash
git clone https://github.com/wwwwwqs/Wen-Qi-Sheng-LLM-project.git
cd Wen-Qi-Sheng-LLM-project
```

### ✅ 2. 创建虚拟环境（可选）

```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
# 或
venv\Scripts\activate           # Windows
```

### ✅ 3. 安装依赖

```bash
pip install -r requirements.txt
```

---

## 📊 数据说明

请将训练/测试数据集放入 `data/` 文件夹中，默认使用 CSV 文件格式

你可以根据数据集格式，自行修改 `utils.py` 中的数据读取逻辑。

---



## 📄 License

本项目仅用于学习与研究目的，禁止商业使用。
