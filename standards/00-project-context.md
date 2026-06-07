# 00 · 项目上下文 〔本项目活记忆 · AI 维护〕

> **作用**:这是项目的"身份档案"。AI 接管项目时先读这里,了解项目目标、技术栈、目录、部署取值。
> **更新时机**:架构、技术栈、目录结构、端口、部署目录、重要约束变化时更新。
> **填写方式**:把 `<...>` 替换成真实内容;用不到的行删掉。

---

## 1. 项目是什么

- **项目名称**: `bank-marketing-prediction`
- **一句话目标**: 基于银行营销客户数据，构建数据分析可视化界面与客户认购预测系统
- **使用者/受益者**: 银行营销团队、业务分析师
- **核心功能**:
  - **数据分析交互页面**: 可视化展示数据分布、特征相关性、营销效果分析
  - **在线预测系统**: 离线训练预测模型后，提供点选式输入界面，预测客户是否会认购定期存款
- **输入/数据**: 银行营销数据集（来自 `data/` 目录）
  - 训练集: `data/train.csv`（22501 行）
  - 测试集: `data/test.csv`（7501 行）
  - 特征: 21个字段（年龄、职业、婚姻状况、教育程度、财务状况等）
  - 目标变量: `subscribe` (yes/no)
  - 数据状态: **不进 Git**，通过 `.gitignore` 排除

## 2. 技术栈

| 层 | 选型 | 理由 |
|---|---|---|
| 语言/运行时 | Python 3.11 | 稳定版本，数据分析生态成熟 |
| Web/API 框架 | Streamlit | 快速构建数据应用，支持交互式可视化 |
| 数据分析 | pandas, numpy, matplotlib, seaborn, plotly | 数据处理与可视化标准库 |
| 机器学习 | scikit-learn | 经典 ML 库，适合离线训练二分类模型 |
| 测试 | pytest | Python 测试标准框架 |
| 格式/静态检查 | ruff | 快速的 Python linter 和 formatter |
| 打包/运行 | Docker | 容器化部署，环境一致性 |
| CI/CD | GitHub Actions | 通用、可视化、适合教学与团队协作 |

## 3. 目录地图

```text
bank-project/
├── standards/                    # AI 项目记忆与通用规范
│   ├── README.md
│   ├── 00-project-context.md
│   ├── 01-requirements.md
│   ├── 02-architecture.md
│   ├── 03-coding-standards.md
│   ├── 04-testing.md
│   ├── 05-ci-cd.md
│   └── 06-deployment.md
├── src/                          # 源代码目录
│   ├── __init__.py
│   ├── app.py                    # Streamlit 应用入口
│   ├── pages/
│   │   ├── 01_data_analysis.py   # 数据分析页面
│   │   └── 02_prediction.py      # 预测页面
│   ├── models/
│   │   ├── __init__.py
│   │   ├── train.py              # 模型训练脚本
│   │   └── predictor.py          # 模型预测模块
│   └── utils/
│       ├── __init__.py
│       ├── data_loader.py        # 数据加载工具
│       └── preprocessing.py      # 数据预处理
├── tests/                        # 测试目录
│   ├── __init__.py
│   ├── test_data_loader.py
│   ├── test_preprocessing.py
│   ├── test_train.py
│   └── test_predictor.py
├── data/                         # 数据目录（不进 Git）
│   ├── train.csv
│   └── test.csv
├── models/                       # 模型产物目录（不进 Git）
│   └── model.pkl
├── requirements.txt              # 生产运行依赖
├── requirements-dev.txt          # 本地/CI 检查依赖
├── Dockerfile
├── .dockerignore
├── .gitignore
├── .github/workflows/
│   └── ci.yml                    # CI 工作流
├── PROGRESS.md                   # 项目进度追踪
└── README.md
```

> 新增目录前先更新本节,避免项目越做越散。

## 4. 质量门槛

| 类型 | 本项目标准 |
|---|---|
| 格式检查 | `ruff format --check .` |
| 静态检查 | `ruff check .` |
| 单元测试 | `pytest` |
| 覆盖率 | `>=70%` (核心模块)>=70%;整体>=50% |
| 构建 | `docker build 成功` |
| 业务/模型指标 | 模型准确率(AUC-ROC) > 0.75；页面正常加载无报错 |

## 5. 不变约束

- 密钥、密码、私钥、Token **绝不写进代码或文档**,只进 GitHub Secrets / 环境变量。
- `data/`、`models/` 目录**不进 Git**，通过 `.gitignore` 排除。
- `main` 分支受保护,日常开发必须走 feature 分支 + PR。
- CI 红灯不合并。
- 端口固定为 **8004**，不可随意更改。

## 6. 部署/CI 占位符取值

> `guides/` 和 workflow 里的通用占位符,在本项目里的真实值只写这里。

| 占位符 | 本项目取值 | 说明 |
|---|---|---|
| `<APP>` | `bank-marketing` | 应用名/镜像名 |
| `<DEPLOY_DIR>` | `/opt/bank-marketing` | 本地部署目录 |
| `<PORT>` | `8004` | 服务端口 |
| `<PYVER>` | `3.11` | Python 版本 |
| `<HEALTHCHECK>` | `/_stcore/health` | Streamlit 健康检查端点 |
| `<SSH_USER>` | 无（本地部署） | 本地部署无需 SSH |
| `<SSH_HOST>` | 无（本地部署） | 本地部署无需 SSH |