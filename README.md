# VecNews / NewsDisco

财经新闻质量评估与 Reward 建模实验项目。项目围绕“高可信信源、可交叉验证事实、高新闻价值、低修辞污染、价值中立度”五个维度，对财经新闻标注样本进行规则评分、批量评估和 LLM-Judge 校准。

项目最大的亮点，是在财经领域追求“公正可靠的信息”。VecNews / NewsDisco 不只判断一篇文章是否像新闻，而是把财经新闻中最影响读者判断的信源、事实证据链、利益视角、交易诱导和情绪化表达拆成可计算、可复核的质量信号，让每个分数都能回到明确的证据与扣分原因。

## 核心亮点

- **面向财经领域的公正可靠性**：重点识别单一信源、事实未核验、观点事实混淆、交易诱导和标题党等高风险问题，减少市场噪声和主观偏见对内容质量判断的影响。
- **五维 100 分 Reward 体系**：将信源可信度、事实交叉印证、新闻价值、语言克制和客观公正汇总为 0-100 分，并输出评级、优点、缺陷和处理建议。
- **可解释、可复现、可校准**：规则评分可以离线批量运行，LLM-Judge 校准可以接入真实 API，历史实验结果保存在 `results/` 目录中，便于复盘和继续优化。

## 当前评分结果

在 50 条财经新闻金标准标注上，核心规则评分脚本 `python3 batch_score.py` 的当前结果为：

| 指标 | 分数 |
| --- | ---: |
| 平均分 | 63.0 / 100 |
| 中位数 | 62.0 / 100 |
| 最高分 | 80.0 / 100 |
| 最低分 | 35.4 / 100 |
| 评级分布 | B: 14, C: 33, D: 3 |

当前 API 实验中表现最强的配方是 **SFT+Factuality**：平均分 **80.5 / 100**，最低 **75**，最高 **85**，高于 SFT 基线的 **74.0 / 100**。它强悍的地方在于把事实性、可验证性和证据链完整度放在优化核心，对财经新闻里最容易误导读者的未核实主张、单一利益视角、交易诱导和情绪化表达有更强约束，因此更适合作为财经新闻 Reward 建模里的高可靠候选路线。

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

- 特别感谢 David Silver（Google DeepMind）等人在强化学习、reward signal 和自我改进系统方向的论文与思想启发。本项目尝试把类似的奖励建模思路迁移到财经新闻质量评估中，用更可解释的评分体系追求公正、可靠、可复核的信息。
