
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from pathlib import Path


# =========================
# Page Settings
# =========================

st.set_page_config(
    page_title="Student Depression EDA",
    layout="wide"
)

sns.set_theme(style="whitegrid", font_scale=1.05)


# =========================
# Custom CSS
# =========================

st.markdown(
    """
    <style>
    .stMetric {
        background-color: #161B22;
        padding: 15px;
        border-radius: 12px;
        border: 1px solid #30363D;
    }

    .stTabs [data-baseweb="tab"] {
        background-color: #161B22;
        border-radius: 10px;
        padding: 10px 16px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# =========================
# Load and Clean Data
# =========================

@st.cache_data
def load_data():

    # Relative path for GitHub / Streamlit Cloud deployment
    data_path = Path(__file__).parent / "student_depression.csv"
    df = pd.read_csv(data_path)

    df_clean = df.copy()

    # Drop id because it is only an identifier
    df_clean = df_clean.drop(columns=["id"])

    # Drop missing Financial Stress because it is important for analysis
    df_clean = df_clean.dropna(subset=["Financial Stress"])

    # Keep only students
    df_clean = df_clean[df_clean["Profession"] == "Student"].copy()

    # Drop Profession because now all records are students
    df_clean = df_clean.drop(columns=["Profession"])

    # Remove unclear categories
    df_clean = df_clean[df_clean["Dietary Habits"] != "Others"].copy()
    df_clean = df_clean[df_clean["Sleep Duration"] != "Others"].copy()

    # Drop columns not useful for students
    df_clean = df_clean.drop(
        columns=["Work Pressure", "Job Satisfaction"],
        errors="ignore"
    )

    # Remove invalid / very rare city values
    city_counts = df_clean["City"].value_counts()
    df_clean = df_clean[df_clean["City"].map(city_counts) >= 10].copy()

    # Group degrees
    bachelors = [
        "B.Ed", "B.Com", "B.Arch", "BCA", "B.Tech", "BHM",
        "BSc", "B.Pharm", "BBA", "LLB", "BE", "BA"
    ]

    masters = [
        "MSc", "MCA", "M.Tech", "M.Ed", "M.Com", "MBBS",
        "M.Pharm", "MBA", "MA", "LLM", "MHM", "ME", "MD"
    ]

    def group_degree(degree):
        degree = str(degree)
        if degree in bachelors:
            return "Bachelors"
        elif degree in masters:
            return "Masters"
        elif degree == "PhD":
            return "PhD"
        elif "Class 12" in degree:
            return "Class 12"
        else:
            return "Other"

    df_clean["Degree Level"] = df_clean["Degree"].apply(group_degree)

    # Remove Other degree level so it does not appear in filters
    df_clean = df_clean[df_clean["Degree Level"] != "Other"].copy()

    return df, df_clean


df, df_clean = load_data()



# =========================
# Dashboard Header
# =========================

st.title("Student Depression — Exploratory Data Analysis")

st.write(
    "This dashboard explores academic and lifestyle factors associated with depression among students."
)

st.markdown("---")


# =========================
# Sidebar Filters
# =========================

st.sidebar.header("Filters")


# Gender filter
gender_options = sorted(df_clean["Gender"].dropna().unique())

selected_gender = st.sidebar.multiselect(
    "Gender",
    gender_options,
    default=gender_options,
    key="gender_filter"
)


# Age filter
min_age = int(df_clean["Age"].min())
max_age = int(df_clean["Age"].max())

selected_age = st.sidebar.slider(
    "Age range",
    min_value=min_age,
    max_value=max_age,
    value=(min_age, max_age),
    key="age_filter"
)


# Degree Level filter
degree_options = sorted(df_clean["Degree Level"].dropna().unique())

selected_degree = st.sidebar.multiselect(
    "Degree Level",
    degree_options,
    default=degree_options,
    key="degree_filter"
)


# Sleep Duration filter
sleep_options = [
    "Less than 5 hours",
    "5-6 hours",
    "7-8 hours",
    "More than 8 hours"
]

selected_sleep = st.sidebar.multiselect(
    "Sleep Duration",
    sleep_options,
    default=sleep_options,
    key="sleep_filter"
)


# Dietary Habits filter
diet_options = [
    "Healthy",
    "Moderate",
    "Unhealthy"
]

selected_diet = st.sidebar.multiselect(
    "Dietary Habits",
    diet_options,
    default=diet_options,
    key="diet_filter"
)


# =========================
# Apply Filters
# =========================

filtered_df = df_clean[
    (df_clean["Gender"].isin(selected_gender)) &
    (df_clean["Age"].between(selected_age[0], selected_age[1])) &
    (df_clean["Degree Level"].isin(selected_degree)) &
    (df_clean["Sleep Duration"].isin(selected_sleep)) &
    (df_clean["Dietary Habits"].isin(selected_diet))
].copy()


# =========================
# Key Metrics
# =========================

st.subheader("Key Metrics")

if filtered_df.empty:
    st.warning("No data available with the current filter selection.")

else:
    total_students = len(filtered_df)
    depressed_students = int(filtered_df["Depression"].sum())
    depression_rate = filtered_df["Depression"].mean() * 100
    avg_academic_pressure = filtered_df["Academic Pressure"].mean()
    avg_financial_stress = filtered_df["Financial Stress"].mean()

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric("Students", f"{total_students:,}")
    col2.metric("Depressed Students", f"{depressed_students:,}")
    col3.metric("Depression Rate", f"{depression_rate:.1f}%")
    col4.metric("Avg Academic Pressure", f"{avg_academic_pressure:.2f} / 5")
    col5.metric("Avg Financial Stress", f"{avg_financial_stress:.2f} / 5")

st.markdown("---")



# =========================
# Function: Depression Rate Bar Chart
# =========================

def show_depression_rate_chart(data, feature, palette="viridis", order=None):

    summary = data.groupby(feature)["Depression"].agg(
        Total_Students="count",
        Depressed_Students="sum",
        Depression_Rate="mean"
    ).reset_index()

    summary["Depression Rate (%)"] = (summary["Depression_Rate"] * 100).round(1)
    summary = summary.drop(columns=["Depression_Rate"])

    summary = summary[summary["Total_Students"] >= 20]

    if summary.empty:
        st.warning(f"Not enough data to show {feature}.")
        return

    if order is not None:
        summary[feature] = pd.Categorical(summary[feature], categories=order, ordered=True)
        summary = summary.sort_values(feature)

    elif pd.api.types.is_numeric_dtype(summary[feature]):
        summary = summary.sort_values(feature)

    else:
        summary = summary.sort_values("Depression Rate (%)", ascending=False)

    summary["Feature Label"] = summary[feature].astype(str)

    st.markdown(f"#### Depression Rate by {feature}")

    fig, ax = plt.subplots(figsize=(8, 4.5))

    sns.barplot(
        data=summary,
        x="Feature Label",
        y="Depression Rate (%)",
        hue="Feature Label",
        palette=palette,
        legend=False,
        ax=ax
    )

    ax.set_title(f"Depression Rate by {feature}", fontsize=14, fontweight="bold")
    ax.set_xlabel(feature)
    ax.set_ylabel("Depression Rate (%)")
    ax.set_ylim(0, 100)
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.3)

    for container in ax.containers:
        ax.bar_label(container, fmt="%.1f%%", fontsize=9)

    st.pyplot(fig)
    plt.close(fig)


# =========================
# Function: Count Chart
# =========================

def show_count_chart(data, feature, order=None):

    temp_df = data.copy()

    temp_df["Depression Label"] = temp_df["Depression"].map({
        0: "Not Depressed",
        1: "Depressed"
    })

    st.markdown(f"#### Depression Count by {feature}")

    fig, ax = plt.subplots(figsize=(9, 4.5))

    sns.countplot(
        data=temp_df,
        x=feature,
        hue="Depression Label",
        order=order,
        palette={
            "Not Depressed": "#4C78A8",
            "Depressed": "#F58518"
        },
        ax=ax
    )

    ax.set_title(f"Depression Count by {feature}", fontsize=14, fontweight="bold")
    ax.set_xlabel(feature)
    ax.set_ylabel("Number of Students")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.3)

    for container in ax.containers:
        ax.bar_label(container, fontsize=9)

    st.pyplot(fig)
    plt.close(fig)


# =========================
# Function: Trend Chart
# =========================

def show_rate_trend_chart(data, feature):

    summary = data.groupby(feature)["Depression"].agg(
        Total_Students="count",
        Depressed_Students="sum",
        Depression_Rate="mean"
    ).reset_index()

    summary["Depression Rate (%)"] = (summary["Depression_Rate"] * 100).round(1)
    summary = summary.drop(columns=["Depression_Rate"])
    summary = summary[summary["Total_Students"] >= 20]
    summary = summary.sort_values(feature)

    st.markdown(f"#### Depression Rate Trend by {feature}")

    fig, ax = plt.subplots(figsize=(9, 4.8))

    sns.lineplot(
        data=summary,
        x=feature,
        y="Depression Rate (%)",
        marker="o",
        linewidth=2.8,
        color="#00A6A6",
        ax=ax
    )

    sns.scatterplot(
        data=summary,
        x=feature,
        y="Depression Rate (%)",
        size="Total_Students",
        sizes=(90, 500),
        color="#F58518",
        legend=False,
        ax=ax
    )

    for _, row in summary.iterrows():
        ax.text(
            row[feature],
            row["Depression Rate (%)"] + 1.5,
            f'{row["Depression Rate (%)"]:.1f}%',
            ha="center",
            fontsize=9
        )

    ax.set_title(f"Depression Rate Trend by {feature}", fontsize=14, fontweight="bold")
    ax.set_xlabel(feature)
    ax.set_ylabel("Depression Rate (%)")
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)

    st.pyplot(fig)
    plt.close(fig)


# =========================
# Function: Boxplot by Depression
# =========================

def show_boxplot_by_depression(data, feature):

    temp_df = data.copy()

    temp_df["Depression Status"] = temp_df["Depression"].map({
        0: "Not Depressed",
        1: "Depressed"
    })

    st.markdown(f"#### {feature} Distribution by Depression Status")

    fig, ax = plt.subplots(figsize=(8, 4.8))

    sns.boxplot(
        data=temp_df,
        x="Depression Status",
        y=feature,
        hue="Depression Status",
        palette={
            "Not Depressed": "#4C78A8",
            "Depressed": "#F58518"
        },
        legend=False,
        ax=ax
    )

    sns.stripplot(
        data=temp_df.sample(min(1200, len(temp_df)), random_state=0),
        x="Depression Status",
        y=feature,
        color="black",
        alpha=0.15,
        jitter=0.25,
        ax=ax
    )

    ax.set_title(f"{feature} by Depression Status", fontsize=14, fontweight="bold")
    ax.set_xlabel("Depression Status")
    ax.set_ylabel(feature)
    ax.grid(axis="y", alpha=0.3)

    st.pyplot(fig)
    plt.close(fig)


# =========================
# Function: Overview Pie
# =========================

def show_overview_pie(data):

    depression_counts = data["Depression"].value_counts().reindex([1, 0], fill_value=0)

    pie_df = pd.DataFrame({
        "Depression Status": ["Depression", "No Depression"],
        "Count": depression_counts.values
    })

    fig = px.pie(
        pie_df,
        names="Depression Status",
        values="Count",
        title="Overall Depression Split",
        color="Depression Status",
        color_discrete_map={
            "Depression": "#D65A5A",
            "No Depression": "#5B84C4"
        }
    )

    fig.update_traces(
        textinfo="percent",
        textfont_size=14,
        marker=dict(line=dict(color="#111111", width=2))
    )

    fig.update_layout(
        template="plotly_dark",
        showlegend=True,
        height=420
    )

    st.plotly_chart(fig, use_container_width=True)


# =========================
# Function: Age Histogram
# =========================

def show_age_histogram_by_depression(data):

    temp_df = data.copy()

    temp_df["Depression Status"] = temp_df["Depression"].map({
        1: "Depression",
        0: "No Depression"
    })

    fig = px.histogram(
        temp_df,
        x="Age",
        color="Depression Status",
        nbins=20,
        barmode="overlay",
        title="Age Distribution by Depression Status",
        color_discrete_map={
            "Depression": "#D65A5A",
            "No Depression": "#5B84C4"
        }
    )

    fig.update_traces(opacity=0.65)

    fig.update_layout(
        template="plotly_dark",
        xaxis_title="Age",
        yaxis_title="Count",
        height=420
    )

    st.plotly_chart(fig, use_container_width=True)


# =========================
# Function: Suicidal Thoughts Rate
# =========================

def show_suicidal_thoughts_rate(data):

    feature = "Have you ever had suicidal thoughts ?"

    summary = data.groupby(feature)["Depression"].agg(
        Total_Students="count",
        Depressed_Students="sum",
        Depression_Rate="mean"
    ).reset_index()

    summary["Depression Rate (%)"] = (summary["Depression_Rate"] * 100).round(1)

    order = ["No", "Yes"]
    summary[feature] = pd.Categorical(summary[feature], categories=order, ordered=True)
    summary = summary.sort_values(feature)

    fig = px.bar(
        summary,
        x=feature,
        y="Depression Rate (%)",
        color=feature,
        text="Depression Rate (%)",
        title="Depression Rate by History of Suicidal Thoughts",
        color_discrete_map={
            "No": "#5B84C4",
            "Yes": "#D65A5A"
        }
    )

    fig.update_traces(
        texttemplate="%{text:.1f}%",
        textposition="inside"
    )

    fig.update_layout(
        template="plotly_dark",
        xaxis_title="Have you ever had suicidal thoughts?",
        yaxis_title="Depression rate (%)",
        yaxis_range=[0, 90],
        height=430,
        showlegend=True
    )

    st.plotly_chart(fig, use_container_width=True)


# =========================
# Function: Correlation Matrix
# =========================

def show_correlation_matrix(data):

    numeric_data = data.select_dtypes(include=["int64", "float64"]).copy()

    numeric_data = numeric_data.drop(
        columns=["Job Satisfaction", "Work Pressure"],
        errors="ignore"
    )

    corr_matrix = numeric_data.corr()

    fig, ax = plt.subplots(figsize=(5, 3.5))

    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        linewidths=0.4,
        annot_kws={"size": 8},
        cbar_kws={"shrink": 0.75},
        ax=ax
    )

    ax.set_title(
        "Correlation Matrix",
        fontsize=12,
        fontweight="bold"
    )

    ax.tick_params(axis="x", labelsize=8, rotation=45)
    ax.tick_params(axis="y", labelsize=8, rotation=0)

    st.pyplot(fig)
    plt.close(fig)

# =========================
# Function: Work/Study Hours Boxplot
# =========================

def show_work_study_hours_by_depression(data):

    temp_df = data.copy()

    temp_df["Depression Label"] = temp_df["Depression"].map({
        0: "Not Depressed",
        1: "Depressed"
    })

    st.markdown("#### Work/Study Hours by Depression")

    fig, ax = plt.subplots(figsize=(8, 4.8))

    sns.boxplot(
        data=temp_df,
        x="Depression Label",
        y="Work/Study Hours",
        hue="Depression Label",
        order=["Depressed", "Not Depressed"],
        palette={
            "Depressed": "#E67E22",
            "Not Depressed": "#2ECC71"
        },
        legend=False,
        ax=ax
    )

    sns.stripplot(
        data=temp_df.sample(min(1200, len(temp_df)), random_state=0),
        x="Depression Label",
        y="Work/Study Hours",
        order=["Depressed", "Not Depressed"],
        color="black",
        alpha=0.12,
        jitter=0.25,
        ax=ax
    )

    ax.set_title("Work/Study Hours Distribution by Depression", fontsize=14, fontweight="bold")
    ax.set_xlabel("Depression Status")
    ax.set_ylabel("Work/Study Hours")
    ax.grid(axis="y", alpha=0.3)

    st.pyplot(fig)
    plt.close(fig)


# =========================
# Function: Sleep + Diet Heatmap
# =========================

def show_sleep_diet_heatmap(data):

    st.markdown("#### Sleep + Diet Depression Rate")

    heatmap_data = (
        data.pivot_table(
            values="Depression",
            index="Sleep Duration",
            columns="Dietary Habits",
            aggfunc="mean"
        ) * 100
    )

    sleep_order = [
        "Less than 5 hours",
        "5-6 hours",
        "7-8 hours",
        "More than 8 hours"
    ]

    diet_order = [
        "Healthy",
        "Moderate",
        "Unhealthy"
    ]

    heatmap_data = heatmap_data.reindex(
        index=sleep_order,
        columns=diet_order
    )

    fig, ax = plt.subplots(figsize=(8, 4.8))

    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt=".1f",
        cmap="Reds",
        linewidths=0.5,
        linecolor="white",
        ax=ax
    )

    ax.set_title("Depression Rate by Sleep and Diet", fontsize=14, fontweight="bold")
    ax.set_xlabel("Dietary Habits")
    ax.set_ylabel("Sleep Duration")

    st.pyplot(fig)
    plt.close(fig)



tab1, tab2, tab3, tab4 = st.tabs([
    "Overview",
    "Academic Factors",
    "Lifestyle Factors",
    "Correlation & Dataset"
])


# =========================
# Tab 1: Overview
# =========================

with tab1:

    st.subheader("Overview")

    if filtered_df.empty:
        st.warning("No data available with the current filter selection.")

    else:
        col1, col2 = st.columns(2)

        with col1:
            show_overview_pie(filtered_df)

        with col2:
            show_age_histogram_by_depression(filtered_df)

        st.markdown("### Depression Rate by Suicidal Thoughts History")

        st.write(
            "This is the single strongest signal in the dataset — included here for context, not as a cause of depression, "
            "since the two are closely linked conditions rather than one causing the other."
        )

        show_suicidal_thoughts_rate(filtered_df)


# =========================
# Tab 2: Academic Factors
# =========================

with tab2:

    st.subheader("Academic Factors")

    if filtered_df.empty:
        st.warning("No data available with the current filter selection.")

    else:
        col1, col2 = st.columns(2)

        with col1:
            show_depression_rate_chart(
                filtered_df,
                "Academic Pressure",
                palette="rocket"
            )

        with col2:
            show_depression_rate_chart(
                filtered_df,
                "Study Satisfaction",
                palette="mako"
            )

        col3, col4 = st.columns(2)

        with col3:
            show_boxplot_by_depression(filtered_df, "Financial Stress")

        with col4:
            show_rate_trend_chart(filtered_df, "Work/Study Hours")


# =========================
# Tab 3: Lifestyle Factors
# =========================

with tab3:

    st.subheader("Lifestyle Factors")

    if filtered_df.empty:
        st.warning("No data available with the current filter selection.")

    else:
        col1, col2 = st.columns(2)

        with col1:
            show_depression_rate_chart(
                filtered_df,
                "Sleep Duration",
                palette="rocket",
                order=[
                    "Less than 5 hours",
                    "5-6 hours",
                    "7-8 hours",
                    "More than 8 hours"
                ]
            )

        with col2:
            show_depression_rate_chart(
                filtered_df,
                "Dietary Habits",
                palette=["#2ECC71", "#F1C40F", "#E74C3C"],
                order=[
                    "Healthy",
                    "Moderate",
                    "Unhealthy"
                ]
            )

        col3, col4 = st.columns(2)

        with col3:
            show_work_study_hours_by_depression(filtered_df)

        with col4:
            show_sleep_diet_heatmap(filtered_df)


# =========================
# Tab 4: Correlation & Dataset
# =========================

with tab4:

    st.subheader("Correlation Matrix")

    if filtered_df.empty:
        st.warning("No data available with the current filter selection.")

    else:
        show_correlation_matrix(filtered_df)

        st.write("Filtered dataset preview:")
        st.dataframe(filtered_df.head(50))

        st.markdown("---")

        st.subheader("Key Factors Associated with Student Depression")

        st.write(
            "Academic pressure is strongly related to depression. When academic pressure is low, "
            "the depression rate is around 19.4%, while when academic pressure is high, "
            "the depression rate is around 86.1%."
        )

        st.write(
            "Financial stress is higher among depressed students. The boxplot shows that students "
            "classified as depressed have higher financial stress levels than students who are not depressed."
        )

        st.write(
            "Students with low study satisfaction have higher depression rates. When study satisfaction is low, "
            "the depression rate is around 70.7%, while students with high study satisfaction have a depression rate around 47.2%."
        )

        st.write(
            "Poor lifestyle habits are associated with depression. Students with healthy habits have a depression rate around 45.3%, "
            "while students with unhealthy habits have a depression rate around 70.7%. Also, students sleeping less than 5 hours show more depression cases."
        )

        st.write(
            "Students who spend more hours studying or working per day tend to have higher levels of depression. "
            "This may be because long daily hours can increase stress and leave less time for rest."
        )
