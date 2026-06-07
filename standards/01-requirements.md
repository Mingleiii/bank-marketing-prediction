# 01 · 需求 / 活 PRD 〔本项目活记忆 · AI 维护〕

> **作用**:这是本项目唯一的需求文档。所有新功能、缺陷、技术债都追加到这里,不要另起多个 PRD 文件。
> **更新时机**:每次有新需求、需求变更、验收标准变化时更新。

---

## 1. 需求来源

| 类型 | 来源 | 进入方式 |
|---|---|---|
| 功能需求 Feature | 用户 / 老师 / 产品 / 客户 | 写成用户故事 |
| 缺陷 Bug | 测试 / 线上日志 / 用户反馈 | 写复现步骤和期望结果 |
| 技术债 Tech Debt | 开发 / Review / CI/CD 故障 | 写影响和修复目标 |

---

## 2. Issue 生命周期

| 阶段 | 状态 | 动作 |
|---|---|---|
| 提出 | Open | 写清场景、目标、验收标准 |
| 排期 | Backlog / Todo | 决定优先级和负责人 |
| 开发 | In Progress | 从 main 开 feature 分支 |
| 评审 | In Review | 提 PR,等待 CI 和 Review |
| 合并 | Done | PR 合并 main,自动关闭 Issue |
| 验收 | Verified | 按验收标准确认 |

**追踪规则**:分支名带 Issue 号,PR 描述写 `closes #<编号>`。

---

## 3. 用户故事模板

```text
### US-<编号> <一句话标题> · 状态: Backlog
作为 <角色>,
我想要 <能力>,
以便 <价值>。

验收标准:
- AC1: Given <前提>,When <动作>,Then <可验证结果>。
- AC2: <补充标准>

技术备注:
- <可选:约束、边界、风险>
```

---

## 4. 需求清单

### US-1 初始化项目工程化与 CI · 状态: Backlog

作为 **项目开发者**,
我想要 项目具备基础工程结构、测试、CI 环境,
以便 后续每次开发都能自动检查代码质量。

验收标准:
- AC1: 从 `main` 开 feature 分支完成初始化,不直接 push main。
- AC2: 配置 GitHub Actions CI,包含格式检查、静态检查、单元测试、构建检查。
- AC3: CI 全绿后合并 main。
- AC4: 完成后更新 `PROGRESS.md`。

---

### US-2 构建数据分析交互页面 · 状态: Backlog

作为 **银行营销分析师**,
我想要 通过交互式可视化页面查看营销数据,
以便 理解客户特征分布和营销效果。

验收标准:
- AC1: Given 用户打开数据分析页面,When 页面加载完成,Then 显示数据概览（总样本数、认购率）。
- AC2: Given 用户查看数据分布,When 选择不同特征维度,Then 展示对应的直方图/饼图。
- AC3: Given 用户分析营销效果,When 查看认购情况,Then 展示不同分组下的认购率对比图。
- AC4: Given 用户探索特征关系,When 查看相关性分析,Then 展示特征相关性热力图。
- AC5: Given 数据文件存在,When 读取失败,Then 显示友好的错误提示。

技术备注:
- 使用 pandas 读取 CSV 数据
- 使用 plotly/ seaborn 生成可视化图表
- 支持 Streamlit 侧边栏筛选功能

---

### US-3 离线训练预测模型 · 状态: Backlog

作为 **机器学习工程师**,
我想要 基于历史数据训练一个二分类模型,
以便 预测新客户是否会认购定期存款。

验收标准:
- AC1: Given 训练数据存在,When 执行训练脚本,Then 产出训练好的模型文件。
- AC2: Given 模型训练完成,When 评估模型性能,Then 输出准确率、精确率、召回率、F1 分数、AUC-ROC。
- AC3: Given 模型训练完成,When 性能指标达标（AUC-ROC > 0.75）,Then 模型文件可被预测模块加载。
- AC4: Given 特征中包含分类变量,When 训练时,Then 自动进行编码处理（OneHot/Label Encoding）。
- AC5: Given 需要复现实验,When 训练结束,Then 输出特征重要性排序。

技术备注:
- 使用 scikit-learn 构建分类模型（建议尝试多种算法对比）
- 进行数据预处理：缺失值处理、异常值检测、特征编码
- 使用交叉验证评估模型稳定性
- 模型文件保存为 pickle 格式

---

### US-4 构建在线预测系统 · 状态: Backlog

作为 **银行营销人员**,
我想要 通过点选表单输入客户信息,即时得到认购预测结果,
以便 优化营销策略、提高转化率。

验收标准:
- AC1: Given 用户进入预测页面,When 页面加载完成,Then 显示所有必填字段的点选/输入表单。
- AC2: Given 用户填写完表单,When 点击"预测"按钮,Then 在 3 秒内显示预测结果（会认购/不会认购）及置信度。
- AC3: Given 模型不存在或加载失败,When 进入预测页面,Then 显示友好的错误提示并提供"训练模型"入口。
- AC4: Given 用户输入了无效数据,When 点击预测,Then 显示具体的验证错误信息。
- AC5: Given 查看历史预测记录,When 存在预测历史,Then 展示最近的预测结果列表（可选功能）。

技术备注:
- 表单字段包括：age, job, marital, education, default, housing, loan, contact, month, day_of_week, duration, campaign, pdays, previous, poutcome, emp_var_rate, cons_price_index, cons_conf_index, lending_rate3m, nr_employed
- 部分数值型字段使用 Slider/NumberInput，分类字段使用 Selectbox
- 结果展示包含预测概率和分类标签

---

### US-5 容器化与本地部署 · 状态: Backlog

作为 **运维人员**,
我想要 通过 Docker 容器部署应用,
以便 确保环境一致性、简化部署流程。

验收标准:
- AC1: Given Dockerfile 配置完成,When 执行构建命令,Then 成功生成镜像。
- AC2: Given 镜像构建成功,When 启动容器,Then 服务在端口 8004 正常运行。
- AC3: Given 服务运行中,When 访问 http://localhost:8004,Then 显示 Streamlit 应用首页。
- AC4: Given 容器运行,When 调用健康检查端点,Then 返回 200 状态码。
- AC5: Given 构建过程,When 使用 .dockerignore,Then 排除不必要文件（data/, models/, .git/ 等）。

技术备注:
- 使用 Python 3.11 作为基础镜像
- 安装生产依赖
- 暴露端口 8004
- 设置 Streamlit 配置（headless、运行地址、端口等）

---

## 5. 非功能需求

- **性能**: 页面首屏加载时间 < 5 秒；预测响应时间 < 3 秒
- **安全**: 密钥只进 Secrets,不进 Git；模型文件和数据目录不进 Git
- **可维护**: 一需求一小 PR,避免大爆炸式提交；代码有清晰的注释和文档
- **可测试**: 核心逻辑必须有单元测试；测试覆盖率核心模块 >= 70%
- **可部署**: 部署后必须有健康检查；CI 全绿才合并
- **用户体验**: 错误信息友好、图表交互流畅、响应式设计