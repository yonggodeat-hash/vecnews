# VecNews / NewsDisco

财经新闻质量评估与 Reward 建模实验项目。项目围绕“高可信信源、可交叉验证事实、高新闻价值、低修辞污染、价值中立度”五个维度，对财经新闻标注样本进行规则评分、批量评估和 LLM-Judge 校准。

## 项目结构

```text
.
├── news_reward/             # 轻量版财经新闻 Reward 评分系统
├── newsdisco_reward/        # 五维评分引擎与标注 schema
├── newsdisco/               # Phase 0 训练、轨迹采集和验证模块
├── data/
│   ├── annotations/         # 50 条财经新闻金标准标注
│   └── gold_scores.json     # 汇总金标准分数
├── experiments/             # LOO-CV、LLM-Judge 校准实验
├── scripts/                 # 云端部署、TPU 设置、成本监控脚本
├── results/                 # 历史实验输出与报告
└── docs/                    # 项目规划书与标注模板
```

## 快速开始

核心规则评分不依赖外部 API：

```bash
python3 batch_score.py
python3 run_phase0_experiment.py
```

运行更完整的实验：

```bash
python3 experiments/phase0_run.py
```

如需更稳定的置换检验，可调高运行参数：

```bash
PHASE0_PERMUTATIONS=500 PHASE0_EPOCHS=3000 python3 experiments/phase0_run.py
```

LLM-Judge 校准脚本会调用 DeepSeek API。运行前先设置环境变量：

```bash
export DEEPSEEK_API_KEY="your_api_key_here"
python3 run_real_calibration.py
```

## 依赖

核心评分模块主要使用 Python 标准库。实验脚本可安装：

```bash
python3 -m pip install -r requirements.txt
```

如需运行 `newsdisco/` 下的训练与轨迹采集模块，再安装：

```bash
python3 -m pip install -r requirements-ml.txt
```

## 发布说明

- 本仓库已移除本地缓存和硬编码 API key，密钥请通过 `.env` 或环境变量管理。
- `.gitignore` 会排除 `__pycache__`、`.env`、本地 IDE/Claude 配置、训练 checkpoint、日志和轨迹数据。
- 当前工作树内容控制在约 1M，适合直接发布到 GitHub。
