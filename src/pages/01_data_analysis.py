"""Data analysis page for bank marketing data."""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.utils.data_loader import DataLoader

# Page config
st.set_page_config(page_title="数据分析", page_icon="📊", layout="wide")

# Title and description
st.title("📊 银行营销数据分析")
st.markdown("---")


# Load data
@st.cache_data
def load_data():
    """Load and cache training data."""
    loader = DataLoader()
    df = loader.load_train()
    return df


try:
    df = load_data()

    # Data summary section
    st.header("数据概览")

    summary = loader = DataLoader()
    summary = loader.get_data_summary(df)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("总样本数", f"{summary['total_rows']:,}")
    with col2:
        st.metric("特征数量", summary["total_columns"])
    with col3:
        st.metric("认购用户", f"{summary['subscribe_distribution']['yes']:,}")
    with col4:
        st.metric(
            "认购率",
            f"{summary['subscribe_distribution']['yes_rate']:.2%}",
        )

    st.markdown("---")

    # Subscribe distribution chart
    st.header("认购分布")

    sub_col1, sub_col2 = st.columns(2)

    with sub_col1:
        fig_pie = go.Figure(
            data=[
                go.Pie(
                    labels=["不认购", "认购"],
                    values=[
                        summary["subscribe_distribution"]["no"],
                        summary["subscribe_distribution"]["yes"],
                    ],
                    marker={"colors": ["#ef553b", "#00cc96"]},
                )
            ]
        )
        fig_pie.update_layout(
            title="认购比例分布",
            height=400,
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with sub_col2:
        # Numerical features statistics
        st.subheader("数值特征统计")
        num_stats_df = pd.DataFrame(summary["numerical_stats"]).T
        num_stats_df = num_stats_df.round(2)
        st.dataframe(num_stats_df, use_container_width=True, height=400)

    st.markdown("---")

    # Age distribution
    st.header("年龄分布分析")

    age_col1, age_col2 = st.columns(2)

    with age_col1:
        fig_age = px.histogram(
            df,
            x="age",
            color="subscribe",
            barmode="overlay",
            nbins=30,
            color_discrete_map={"no": "#ef553b", "yes": "#00cc96"},
            labels={"age": "年龄", "subscribe": "是否认购"},
            title="按认购状态的年龄分布",
        )
        fig_age.update_layout(height=400)
        st.plotly_chart(fig_age, use_container_width=True)

    with age_col2:
        # Subscription rate by age group
        df["age_group"] = pd.cut(
            df["age"],
            bins=[17, 25, 35, 45, 55, 65, 100],
            labels=["18-25", "26-35", "36-45", "46-55", "56-65", "65+"],
        )
        age_rate = (
            df.groupby("age_group", observed=True)["subscribe"]
            .apply(lambda x: (x == "yes").mean() * 100)
            .reset_index()
        )
        age_rate.columns = ["age_group", "subscribe_rate"]

        fig_age_rate = px.bar(
            age_rate,
            x="age_group",
            y="subscribe_rate",
            title="各年龄段认购率",
            labels={"age_group": "年龄段", "subscribe_rate": "认购率 (%)"},
            color="subscribe_rate",
            color_continuous_scale="Viridis",
        )
        fig_age_rate.update_layout(height=400, yaxis_title="认购率 (%)")
        st.plotly_chart(fig_age_rate, use_container_width=True)

    st.markdown("---")

    # Job analysis
    st.header("职业分析")

    job_col1, job_col2 = st.columns(2)

    with job_col1:
        job_count = df["job"].value_counts().reset_index()
        job_count.columns = ["job", "count"]

        fig_job = px.bar(
            job_count,
            x="job",
            y="count",
            title="客户职业分布",
            labels={"job": "职业", "count": "数量"},
            color="count",
            color_continuous_scale="Blues",
        )
        fig_job.update_layout(height=400, xaxis_title=None)
        st.plotly_chart(fig_job, use_container_width=True)

    with job_col2:
        # Subscription rate by job
        job_sub_rate = (
            df.groupby("job")["subscribe"]
            .apply(lambda x: (x == "yes").mean() * 100)
            .sort_values(ascending=False)
            .reset_index()
        )
        job_sub_rate.columns = ["job", "subscribe_rate"]

        fig_job_rate = px.bar(
            job_sub_rate,
            x="job",
            y="subscribe_rate",
            title="各职业认购率",
            labels={"job": "职业", "subscribe_rate": "认购率 (%)"},
            color="subscribe_rate",
            color_continuous_scale="RdYlGn",
        )
        fig_job_rate.update_layout(height=400, xaxis_title=None)
        st.plotly_chart(fig_job_rate, use_container_width=True)

    st.markdown("---")

    # Contact duration analysis
    st.header("通话时长分析")

    duration_col1, duration_col2 = st.columns(2)

    with duration_col1:
        # Distribution
        fig_duration = px.histogram(
            df,
            x="duration",
            color="subscribe",
            barmode="overlay",
            nbins=50,
            color_discrete_map={"no": "#ef553b", "yes": "#00cc96"},
            labels={"duration": "通话时长(秒)", "subscribe": "是否认购"},
            title="通话时长分布（按认购状态）",
        )
        fig_duration.update_layout(height=400)
        st.plotly_chart(fig_duration, use_container_width=True)

    with duration_col2:
        # Box plot
        fig_duration_box = px.box(
            df,
            x="subscribe",
            y="duration",
            color="subscribe",
            color_discrete_map={"no": "#ef553b", "yes": "#00cc96"},
            labels={"subscribe": "是否认购", "duration": "通话时长(秒)"},
            title="通话时长箱线图",
        )
        fig_duration_box.update_layout(height=400)
        st.plotly_chart(fig_duration_box, use_container_width=True)

    st.info("💡 观察：通话时长较长的客户更有可能认购定期存款")

    st.markdown("---")

    # Correlation heatmap
    st.header("数值特征相关性")

    # Get numerical columns only
    numerical_cols = DataLoader.NUMERICAL_COLUMNS
    corr_matrix = df[numerical_cols].corr()

    fig_heatmap = px.imshow(
        corr_matrix,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="RdBu_r",
        title="数值特征相关性矩阵",
    )
    fig_heatmap.update_layout(height=500)
    st.plotly_chart(fig_heatmap, use_container_width=True)

    st.markdown("---")

    # Previous campaign outcome analysis
    st.header("上次营销结果分析")

    outcome_count = df["poutcome"].value_counts().reset_index()
    outcome_count.columns = ["poutcome", "count"]

    fig_outcome = px.bar(
        outcome_count,
        x="poutcome",
        y="count",
        color="poutcome",
        title="上次营销结果分布",
        labels={"poutcome": "上次营销结果", "count": "数量"},
        color_discrete_sequence=["#636EFA", "#EF553B", "#00CC96", "#AB63FA"],
    )
    fig_outcome.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig_outcome, use_container_width=True)

    # Calculate subscription rate by previous outcome
    outcome_rate = (
        df.groupby("poutcome")["subscribe"].apply(lambda x: (x == "yes").mean() * 100).reset_index()
    )
    outcome_rate.columns = ["poutcome", "subscribe_rate"]

    st.subheader("上次营销结果与认购率关系")
    for _, row in outcome_rate.iterrows():
        st.metric(row["poutcome"], f"{row['subscribe_rate']:.2f}%")

    st.markdown("---")

    # Key insights summary
    st.header("关键洞察")

    insight_col1, insight_col2 = st.columns(2)

    with insight_col1:
        st.markdown(
            """
        ### 数据特征
        - 整体认购率约 **{:.1f}%**
        - 平均通话时长 {:.0f} 秒
        - 年龄范围: {} - {} 岁
        """.format(
                summary["subscribe_distribution"]["yes_rate"] * 100,
                summary["numerical_stats"]["duration"]["mean"],
                int(summary["numerical_stats"]["age"]["min"]),
                int(summary["numerical_stats"]["age"]["max"]),
            )
        )

    with insight_col2:
        # Find highest conversion job
        job_sub_rate_sorted = (
            df.groupby("job")["subscribe"]
            .apply(lambda x: (x == "yes").mean() * 100)
            .sort_values(ascending=False)
        )

        st.markdown(
            """
        ### 营销建议
        - **最高转化率职业**: {} ({:.1f}%)
        - 重点关注 **学生** 和 **退休人员** 群体
        - 增加通话时长可能提高转化率
        """.format(
                job_sub_rate_sorted.index[0],
                job_sub_rate_sorted.iloc[0],
            )
        )

except FileNotFoundError:
    st.error("❌ 数据文件未找到！")
    st.info("请确保 `data/train.csv` 文件存在。")

except Exception as e:
    st.error(f"❌ 加载数据时出错: {e}")
    st.info("请检查数据文件格式是否正确。")
