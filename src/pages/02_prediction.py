"""Prediction page for bank marketing prediction."""

import streamlit as st

# Page config
st.set_page_config(page_title="在线预测", page_icon="🔮", layout="wide")

# Title
st.title("🔮 客户认购预测")
st.markdown("---")

# Load model
try:
    from src.models.predictor import ModelPredictor

    predictor = ModelPredictor()
    cat_options = predictor.get_categorical_options()
    model_info = predictor.get_model_info()

    # Model info
    st.info(f"✅ 已加载模型: {model_info['model_type']}")

except FileNotFoundError:
    st.error("❌ 预测模型未找到！")
    st.info("请先运行模型训练：`python -m src.models.train`")
    st.stop()

except Exception as e:
    st.error(f"❌ 加载模型失败: {e}")
    st.stop()

# Create form
st.markdown("### 请填写客户信息")

with st.form("prediction_form", clear_on_submit=False):
    # Personal Information Section
    st.subheader("👤 个人信息")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.slider(
            "年龄",
            min_value=18,
            max_value=95,
            value=40,
            step=1,
            help="客户年龄",
        )

        job = st.selectbox(
            "职业",
            options=cat_options.get("job", []),
            index=0,
            help="客户职业类型",
        )

    with col2:
        marital = st.selectbox(
            "婚姻状况",
            options=cat_options.get("marital", []),
            index=0,
            help="客户婚姻状态",
        )

        education = st.selectbox(
            "教育程度",
            options=cat_options.get("education", []),
            index=0,
            help="客户教育背景",
        )

    with col3:
        default = st.selectbox(
            "是否有违约记录",
            options=cat_options.get("default", []),
            index=0,
            help="客户是否有信用违约记录",
        )

    # Financial Information Section
    st.subheader("💰 财务状况")

    col4, col5, col6 = st.columns(3)

    with col4:
        housing = st.selectbox(
            "是否有房贷",
            options=cat_options.get("housing", []),
            index=0,
            help="客户是否有住房贷款",
        )

        loan = st.selectbox(
            "是否有个人贷款",
            options=cat_options.get("loan", []),
            index=0,
            help="客户是否有个人贷款",
        )

    with col5:
        duration = st.slider(
            "上次通话时长（秒）",
            min_value=0,
            max_value=5000,
            value=300,
            step=10,
            help="上一次营销通话的持续时间",
        )

        campaign = st.slider(
            "当前活动联系次数",
            min_value=1,
            max_value=60,
            value=2,
            step=1,
            help="当前营销活动中与客户的联系次数",
        )

    with col6:
        pdays = st.slider(
            "距离上次联系天数",
            min_value=0,
            max_value=1000,
            value=999,
            step=1,
            help="距离上一次营销联系的天数（999表示从未联系过）",
        )

        previous = st.slider(
            "之前活动联系次数",
            min_value=0,
            max_value=50,
            value=0,
            step=1,
            help="之前营销活动中的联系次数",
        )

    # Contact Information Section
    st.subheader("📞 联系信息")

    col7, col8, col9 = st.columns(3)

    with col7:
        contact = st.selectbox(
            "联系方式",
            options=cat_options.get("contact", []),
            index=0,
            help="与客户联系的方式",
        )

        month = st.selectbox(
            "上次联系月份",
            options=cat_options.get("month", []),
            index=0,
            help="上一次联系的月份",
        )

    with col8:
        day_of_week = st.selectbox(
            "上次联系星期",
            options=cat_options.get("day_of_week", []),
            index=0,
            help="上一次联系的星期几",
        )

        poutcome = st.selectbox(
            "上次营销结果",
            options=cat_options.get("poutcome", []),
            index=0,
            help="上一次营销活动的结果",
        )

    with col9:
        emp_var_rate = st.slider(
            "就业变化率",
            min_value=-3.5,
            max_value=1.5,
            value=1.1,
            step=0.1,
            format="%.2f",
            help="就业指标的变化率",
        )

    # Economic Indicators Section
    st.subheader("📊 经济指标")

    col10, col11, col12 = st.columns(3)

    with col10:
        cons_price_index = st.slider(
            "消费者价格指数",
            min_value=92.0,
            max_value=95.0,
            value=93.9,
            step=0.1,
            format="%.2f",
            help="消费者物价指数",
        )

    with col11:
        cons_conf_index = st.slider(
            "消费者信心指数",
            min_value=-50.0,
            max_value=-30.0,
            value=-36.0,
            step=0.5,
            format="%.1f",
            help="消费者信心指数",
        )

    with col12:
        lending_rate3m = st.slider(
            "3个月利率（%）",
            min_value=0.5,
            max_value=5.5,
            value=4.85,
            step=0.05,
            format="%.2f",
            help="银行3个月基准利率",
        )

        nr_employed = st.slider(
            "就业人数（千）",
            min_value=4900.0,
            max_value=5250.0,
            value=5190.0,
            step=5.0,
            format="%.1f",
            help="就业人数（单位：千人）",
        )

    # Submit button
    st.markdown("---")
    submitted = st.form_submit_button("🔮 开始预测", type="primary", use_container_width=True)

    # Make prediction
    if submitted:
        with st.spinner("正在预测..."):
            # Prepare input data
            input_data = {
                "age": age,
                "job": job,
                "marital": marital,
                "education": education,
                "default": default,
                "housing": housing,
                "loan": loan,
                "contact": contact,
                "month": month,
                "day_of_week": day_of_week,
                "duration": duration,
                "campaign": campaign,
                "pdays": pdays,
                "previous": previous,
                "poutcome": poutcome,
                "emp_var_rate": emp_var_rate,
                "cons_price_index": cons_price_index,
                "cons_conf_index": cons_conf_index,
                "lending_rate3m": lending_rate3m,
                "nr_employed": nr_employed,
            }

            try:
                # Get prediction
                result = predictor.predict(input_data)

                # Display results
                st.markdown("---")
                st.subheader("📊 预测结果")

                # Prediction result
                pred_col1, pred_col2, pred_col3 = st.columns(3)

                with pred_col1:
                    if result["prediction"] == "yes":
                        st.metric(
                            "预测结果",
                            "✅ 会认购",
                            delta=f"置信度: {result['confidence']:.2%}",
                        )
                    else:
                        st.metric(
                            "预测结果",
                            "❌ 不会认购",
                            delta=f"置信度: {result['confidence']:.2%}",
                        )

                with pred_col2:
                    st.metric(
                        "认购概率",
                        f"{result['subscribe_probability']:.2%}",
                    )

                with pred_col3:
                    st.metric(
                        "不认购概率",
                        f"{result['not_subscribe_probability']:.2%}",
                    )

                # Probability bar
                st.markdown("---")
                st.subheader("概率分布")

                prob_col1, prob_col2 = st.columns([1, 1])

                with prob_col1:
                    st.progress(
                        result["subscribe_probability"],
                        text=f"认购: {result['subscribe_probability']:.2%}",
                    )

                with prob_col2:
                    st.progress(
                        result["not_subscribe_probability"],
                        text=f"不认购: {result['not_subscribe_probability']:.2%}",
                    )

                # Recommendation
                st.markdown("---")
                st.subheader("💡 营销建议")

                if result["subscribe_probability"] > 0.7:
                    st.success(
                        """
                        ✅ **强烈推荐跟进**

                        该客户有很高的认购意愿，建议：
                        - 安排优先电话联系
                        - 提供更有竞争力的利率
                        - 安排专属客户经理跟进
                        """
                    )
                elif result["subscribe_probability"] > 0.4:
                    st.warning(
                        """
                        ⚠️ **可以考虑跟进**

                        该客户有中等认购意愿，建议：
                        - 评估营销成本后决定是否跟进
                        - 尝试个性化营销策略
                        - 提供差异化产品推荐
                        """
                    )
                else:
                    st.error(
                        """
                        ❌ **不建议优先跟进**

                        该客户认购意愿较低，建议：
                        - 暂不投入过多营销资源
                        - 如需跟进，采用低成本渠道
                        - 等待更合适的时机
                        """
                    )

            except Exception as e:
                st.error(f"❌ 预测失败: {e}")

# Feature importance section
st.markdown("---")
st.markdown("### 📈 模型特征重要性")

try:
    if model_info.get("supports_feature_importance"):
        importance_df = predictor.get_feature_importance()
        top_features = importance_df.head(15)

        # Display as a bar chart
        import plotly.express as px

        fig = px.bar(
            top_features,
            x="importance",
            y="feature",
            orientation="h",
            title="Top 15 最重要的特征",
            labels={"importance": "重要性", "feature": "特征"},
            color="importance",
            color_continuous_scale="Viridis",
        )
        fig.update_layout(height=500, yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("当前模型不支持特征重要性分析")

except Exception as e:
    st.warning(f"无法加载特征重要性: {e}")

# Tips section
st.markdown("---")
st.info(
    """
    💡 **使用提示**

    - 通话时长（duration）是最重要的预测因素之一
    - 之前成功的营销结果会显著提高认购概率
    - 经济指标（如利率、就业率）也会影响预测结果
    - 对于高价值客户，建议进行人工二次确认
    """
)
