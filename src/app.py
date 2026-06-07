"""Main Streamlit application for Bank Marketing Prediction."""

import streamlit as st

# Page config
st.set_page_config(page_title="银行营销预测系统", page_icon="🏦", layout="wide")

# Custom CSS
st.markdown(
    """
    <style>
    .main-header {
        text-align: center;
        padding: 20px 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 30px;
    }
    .stMetric {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Main page content
st.markdown(
    '<div class="main-header"><h1>🏦 银行营销数据分析与预测系统</h1></div>', unsafe_allow_html=True
)

st.markdown("---")

# Introduction
st.markdown("""
### 欢迎使用银行营销预测系统

本系统提供以下功能：

1. **📊 数据分析** - 深入分析银行营销数据，了解客户特征和认购趋势
2. **🔮 智能预测** - 基于机器学习模型预测客户是否会认购定期存款

---

使用左侧导航栏切换不同页面。
""")

# Quick stats section (if model exists)
try:
    from src.models.predictor import ModelPredictor
    from src.utils.data_loader import DataLoader

    try:
        predictor = ModelPredictor()
        model_info = predictor.get_model_info()

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("模型类型", model_info["model_type"].replace("Classifier", " 分类器"))

        with col2:
            loader = DataLoader()
            df = loader.load_train()
            summary = loader.get_data_summary(df)
            st.metric("训练样本数", f"{summary['total_rows']:,}")

        with col3:
            st.metric(
                "数据认购率",
                f"{summary['subscribe_distribution']['yes_rate']:.2%}",
            )

        st.markdown("---")
        st.success("✅ 预测模型已加载，可以开始预测！")

    except FileNotFoundError:
        st.warning("⚠️ 预测模型未找到，请先运行模型训练脚本。")
        st.info("运行 `python -m src.models.train` 来训练模型。")

except ImportError:
    st.info("💡 请安装依赖后使用完整功能：`pip install -r requirements.txt`")

# Feature importance preview (if model supports it)
try:
    predictor = ModelPredictor()
    if model_info.get("supports_feature_importance"):
        st.subheader("📈 模型特征重要性预览")

        importance_df = predictor.get_feature_importance()
        top_features = importance_df.head(10)

        st.dataframe(top_features, use_container_width=True, hide_index=True)

except Exception:
    pass

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
        <p>基于 Python 3.11, Streamlit, scikit-learn 构建</p>
        <p>© 2024 银行营销预测系统</p>
    </div>
    """,
    unsafe_allow_html=True,
)
