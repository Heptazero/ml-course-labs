# Elliptic Bitcoin Fraud Detection

按顺序运行三个 notebook：

1. `01_eda.ipynb`：下载 Elliptic 数据集并了解节点、边、特征和类别分布。
2. `02_baseline.ipynb`：使用决策树和 SVM 建立非图模型基线。
3. `03_graph_transformer.ipynb`：使用图结构训练 Graph Transformer。

数据由 PyTorch Geometric 自动下载到运行环境，不依赖本地文件或 Google Drive。
