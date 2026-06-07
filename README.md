# Bank Marketing Prediction

基于银行营销数据构建的数据分析可视化与客户认购预测系统。

## 功能

- **数据分析交互页面**: 可视化展示数据分布、特征相关性、营销效果分析
- **在线预测系统**: 离线训练预测模型后，提供点选式输入界面，预测客户是否会认购定期存款

## 技术栈

- Python 3.11
- Streamlit
- pandas, numpy
- scikit-learn
- pytest, ruff
- Docker

## 安装

```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安装依赖
pip install -r requirements-dev.txt
```

## 运行

```bash
streamlit run src/app.py --server.port 8004
```

## 测试

```bash
pytest
```

## 部署

```bash
docker build -t bank-marketing .
docker run -p 8004:8004 bank-marketing
```

## 文档

- [项目上下文](standards/00-project-context.md)
- [需求文档](standards/01-requirements.md)
- [项目进度](PROGRESS.md)