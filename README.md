# ML Course Labs

机器学习与数据挖掘课程中的 notebook、作业和小型实验。

```text
ml-course-labs/
├── machine-learning/
│   └── optimizer.ipynb                         # Rosenbrock 函数上的梯度下降与牛顿法
└── data-mining/
    └── elliptic-bitcoin-fraud-detection/
        ├── 01_eda.ipynb                        # 数据下载与探索性分析
        ├── 02_baseline.ipynb                   # 决策树、SVM 基线
        └── 03_graph_transformer.ipynb          # 图 Transformer 实验
```

## 直接在 Colab 运行

- [机器学习：优化器](https://colab.research.google.com/github/Heptazero/ml-course-labs/blob/main/machine-learning/optimizer.ipynb)
- [数据挖掘：Elliptic 01 EDA](https://colab.research.google.com/github/Heptazero/ml-course-labs/blob/main/data-mining/elliptic-bitcoin-fraud-detection/01_eda.ipynb)
- [数据挖掘：Elliptic 02 Baseline](https://colab.research.google.com/github/Heptazero/ml-course-labs/blob/main/data-mining/elliptic-bitcoin-fraud-detection/02_baseline.ipynb)
- [数据挖掘：Elliptic 03 Graph Transformer](https://colab.research.google.com/github/Heptazero/ml-course-labs/blob/main/data-mining/elliptic-bitcoin-fraud-detection/03_graph_transformer.ipynb)

Elliptic notebook 会把数据下载到 Colab 的临时空间，不需要挂载 Google Drive。Colab 运行时重启后，临时数据会被清除并在下次运行时重新下载。

真正用于自动验证代码正确性的文件，放在对应项目的 `tests/` 中；一次性的尝试放在对应项目的 `experiments/` 中。
