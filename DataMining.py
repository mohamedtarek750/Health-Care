import io
import textwrap
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


mode = st.selectbox(
    "Choose Mode",
    ["Clustering", "Classification"]
)

st.title("Healthcare Patient Clustering Dashboard")
st.caption("Interactive dashboard for patient clustering, visualizations, and business insights")

TARGET_COL = "readmitted"
DEFAULT_CAT_COLS = ["gender", "primary_diagnosis", "discharge_to"]
DEFAULT_NUM_COLS = ["age", "num_procedures", "days_in_hospital", "comorbidity_score"]

def preprocess_and_classify(df, selected_num_cols, selected_cat_cols):
    work_df = df.copy()

    if TARGET_COL not in work_df.columns:
        st.error("Target column not found")
        st.stop()

    X = work_df[selected_num_cols + selected_cat_cols].copy()
    y = work_df[TARGET_COL]

    # cleaning
    for col in selected_num_cols:
        X[col] = pd.to_numeric(X[col], errors="coerce")
        X[col] = X[col].fillna(X[col].median())

    for col in selected_cat_cols:
        X[col] = X[col].astype(str)

    # encoding
    X = pd.get_dummies(X)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)

    acc = accuracy_score(y_val, y_pred)
    report = classification_report(y_val, y_pred, output_dict=True)
    cm = confusion_matrix(y_val, y_pred)

    return acc, report, cm

def safe_mode(series: pd.Series):
    mode = series.mode(dropna=True)
    return mode.iloc[0] if not mode.empty else "N/A"


def build_insights(df: pd.DataFrame, cluster_col: str = "cluster"):
    insights = []

    if cluster_col not in df.columns:
        return ["No clusters available yet."]

    cluster_sizes = df[cluster_col].value_counts().sort_index()
    largest_cluster = int(cluster_sizes.idxmax())
    smallest_cluster = int(cluster_sizes.idxmin())
    insights.append(
        f"Largest patient segment is Cluster {largest_cluster} with {cluster_sizes.max()} patients, while Cluster {smallest_cluster} is the smallest with {cluster_sizes.min()} patients."
    )

    if TARGET_COL in df.columns:
        readm = df.groupby(cluster_col)[TARGET_COL].mean().sort_values(ascending=False)
        high_risk = int(readm.index[0])
        low_risk = int(readm.index[-1])
        insights.append(
            f"Cluster {high_risk} has the highest readmission rate ({readm.iloc[0]:.1%}), compared with Cluster {low_risk} ({readm.iloc[-1]:.1%})."
        )

    numeric_cols = [c for c in DEFAULT_NUM_COLS if c in df.columns]
    if numeric_cols:
        means = df.groupby(cluster_col)[numeric_cols].mean()
        top_age_cluster = int(means["age"].idxmax()) if "age" in means.columns else None
        top_stay_cluster = int(means["days_in_hospital"].idxmax()) if "days_in_hospital" in means.columns else None
        if top_age_cluster is not None:
            insights.append(
                f"Cluster {top_age_cluster} is the oldest segment on average ({means.loc[top_age_cluster, 'age']:.1f} years)."
            )
        elif top_stay_cluster is not None:
            insights.append(
                f"Cluster {top_stay_cluster} has the longest average hospital stay ({means.loc[top_stay_cluster, 'days_in_hospital']:.1f} days)."
            )

    return insights[:3]


@st.cache_data
def load_data(uploaded_file) -> pd.DataFrame:
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    if name.endswith(".xlsx"):
        return pd.read_excel(uploaded_file)
    raise ValueError("Please upload a CSV or XLSX file.")

@st.cache_data
def preprocess_and_cluster(
    df: pd.DataFrame,
    selected_num_cols,
    selected_cat_cols,
    n_clusters: int,
    random_state: int,
):
    work_df = df.copy()

    feature_cols = list(selected_num_cols) + list(selected_cat_cols)
    work_df = work_df[feature_cols + ([TARGET_COL] if TARGET_COL in df.columns else [])].copy()

    for col in selected_num_cols:
        work_df[col] = pd.to_numeric(work_df[col], errors="coerce")
        work_df[col] = work_df[col].fillna(work_df[col].median())

    for col in selected_cat_cols:
        work_df[col] = work_df[col].astype(str).fillna("Unknown")

    encoded = pd.get_dummies(work_df[feature_cols], columns=list(selected_cat_cols), drop_first=False)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(encoded)

    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = model.fit_predict(X_scaled)

    result_df = df.copy()
    result_df["cluster"] = labels

    pca = PCA(n_components=2, random_state=random_state)
    pca_components = pca.fit_transform(X_scaled)
    pca_df = pd.DataFrame(
        {
            "PC1": pca_components[:, 0],
            "PC2": pca_components[:, 1],
            "cluster": labels.astype(str),
        }
    )

    silhouette = None
    if n_clusters > 1 and len(set(labels)) > 1:
        silhouette = silhouette_score(X_scaled, labels)

    numeric_summary = (
        result_df.groupby("cluster")[[c for c in selected_num_cols if c in result_df.columns] + ([TARGET_COL] if TARGET_COL in result_df.columns else [])]
        .mean(numeric_only=True)
        .round(3)
    )

    categorical_summary = {}
    for col in selected_cat_cols:
        if col in result_df.columns:
            categorical_summary[col] = result_df.groupby("cluster")[col].agg(safe_mode)

    return result_df, encoded, pca_df, silhouette, numeric_summary, categorical_summary


with st.sidebar:
    st.header("Controls")
    uploaded_file = st.file_uploader("Upload patient dataset", type=["csv", "xlsx"])
    n_clusters = st.slider("Number of clusters", min_value=2, max_value=8, value=3, step=1)
    random_state = st.number_input("Random state", min_value=0, value=42, step=1)

st.markdown(
    """
This dashboard helps you:
- cluster patients interactively
- compare cluster characteristics
- visualize diagnosis and readmission patterns
- generate business insights for your project discussion
"""
)

if uploaded_file is None:
    st.info("Upload your CSV/XLSX file to start. Your file should include columns like age, gender, primary_diagnosis, num_procedures, days_in_hospital, comorbidity_score, discharge_to, and readmitted.")
    st.stop()

try:
    df = load_data(uploaded_file)
except Exception as e:
    st.error(f"Could not read the file: {e}")
    st.stop()

st.subheader("1) Dataset Preview")
col1, col2, col3 = st.columns(3)
col1.metric("Rows", f"{len(df):,}")
col2.metric("Columns", df.shape[1])
col3.metric("Missing cells", int(df.isna().sum().sum()))

with st.expander("Show raw data"):
    st.dataframe(df.head(20), use_container_width=True)

available_num_cols = [c for c in DEFAULT_NUM_COLS if c in df.columns]
available_cat_cols = [c for c in DEFAULT_CAT_COLS if c in df.columns]

if not available_num_cols:
    st.error("No expected numeric columns were found. Add numeric columns like age, num_procedures, days_in_hospital, and comorbidity_score.")
    st.stop()

selected_num_cols = st.multiselect(
    "Select numeric features for clustering",
    options=[c for c in df.columns if df[c].dtype != "object" or c in DEFAULT_NUM_COLS],
    default=available_num_cols,
)

selected_cat_cols = st.multiselect(
    "Select categorical features for clustering",
    options=[c for c in df.columns if df[c].dtype == "object" or c in DEFAULT_CAT_COLS],
    default=available_cat_cols,
)

if not selected_num_cols and not selected_cat_cols:
    st.warning("Choose at least one feature for clustering.")
    st.stop()

numeric_summary = pd.DataFrame()
categorical_summary = {}
result_df = pd.DataFrame()
encoded_df = pd.DataFrame()
pca_df = pd.DataFrame()
silhouette = None

if mode == "Clustering":
    result_df, encoded_df, pca_df, silhouette, numeric_summary, categorical_summary = preprocess_and_cluster(
        df,
        selected_num_cols,
        selected_cat_cols,
        n_clusters,
        random_state,
    )

elif mode == "Classification":
    acc, report, cm = preprocess_and_classify(
        df,
        selected_num_cols,
        selected_cat_cols,
    )

    st.subheader("2) Classification Overview")
    st.metric("Accuracy", f"{acc:.3f}")

    st.subheader("Classification Report")
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df)

    st.subheader("Confusion Matrix")
    fig_cm = px.imshow(cm, text_auto=True, title="Confusion Matrix")
    st.plotly_chart(fig_cm, use_container_width=True)

    st.subheader("Feature Importance")

    X = pd.get_dummies(df[selected_num_cols + selected_cat_cols])
    y = df[TARGET_COL]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_scaled, y)

    importances = model.feature_importances_
    feat_imp = pd.Series(importances, index=X.columns).sort_values(ascending=False)

    fig_imp = px.bar(feat_imp.head(10), title="Top 10 Important Features")
    st.plotly_chart(fig_imp, use_container_width=True)

    st.stop()

if mode == "Clustering":

    st.subheader("2) Clustering Overview")
    metric_cols = st.columns(3)
    metric_cols[0].metric("Clusters", n_clusters)
    metric_cols[1].metric("Encoded features", encoded_df.shape[1])
    metric_cols[2].metric("Silhouette score", f"{silhouette:.3f}" if silhouette is not None else "N/A")

    cluster_counts = result_df["cluster"].value_counts().sort_index().reset_index()
    cluster_counts.columns = ["cluster", "count"]

    fig_counts = px.bar(
        cluster_counts,
        x="cluster",
        y="count",
        title="Cluster Sizes",
        text="count",
    )
    fig_counts.update_layout(xaxis_title="Cluster", yaxis_title="Patients")

    fig_pca = px.scatter(
        pca_df,
        x="PC1",
        y="PC2",
        color="cluster",
        title="Patient Clusters in 2D PCA Space",
        opacity=0.75,
    )

    left, right = st.columns(2)
    left.plotly_chart(fig_counts, use_container_width=True)
    right.plotly_chart(fig_pca, use_container_width=True)

st.subheader("3) Cluster Profiling")
profile_tab1, profile_tab2, profile_tab3 = st.tabs(["Numeric Summary", "Categorical Summary", "Cluster Data"])

with profile_tab1:
    if numeric_summary is not None and not numeric_summary.empty:
        st.dataframe(numeric_summary, use_container_width=True)

        heatmap_df = numeric_summary.reset_index()

        heatmap_df = heatmap_df.melt(
            id_vars=heatmap_df.columns[0],
            var_name="feature",
            value_name="value"
        )

        fig_heatmap = px.density_heatmap(
            heatmap_df,
            x="feature",
            y="cluster",
            z="value",
            histfunc="avg",
            text_auto=True,
            title="Average Numeric Profile per Cluster",
        )

        st.plotly_chart(fig_heatmap, use_container_width=True)

    else:
        st.info("No numeric cluster summary available (you are likely in Classification mode).")

with profile_tab2:
    if categorical_summary:
        cat_frames = []
        for col, series in categorical_summary.items():
            tmp = series.reset_index()
            tmp.columns = ["cluster", "most_common_value"]
            tmp["feature"] = col
            cat_frames.append(tmp)
        cat_summary_df = pd.concat(cat_frames, ignore_index=True)
        st.dataframe(cat_summary_df, use_container_width=True)
    else:
        st.info("No categorical columns selected.")

with profile_tab3:
    cluster_filter = st.selectbox("Inspect cluster", sorted(result_df["cluster"].unique()))
    filtered = result_df[result_df["cluster"] == cluster_filter]
    st.write(f"Patients in Cluster {cluster_filter}: {len(filtered):,}")
    st.dataframe(filtered.head(100), use_container_width=True)

st.subheader("4) Visual Analytics")
chart_tab1, chart_tab2, chart_tab3 = st.tabs(["Diagnosis Mix", "Readmission", "Distribution Explorer"])

with chart_tab1:
    if "primary_diagnosis" in result_df.columns:
        diag_dist = (
            result_df.groupby(["cluster", "primary_diagnosis"]).size().reset_index(name="count")
        )
        fig_diag = px.bar(
            diag_dist,
            x="cluster",
            y="count",
            color="primary_diagnosis",
            title="Diagnosis Distribution by Cluster",
            barmode="stack",
        )
        st.plotly_chart(fig_diag, use_container_width=True)

        diag_pct = pd.crosstab(result_df["cluster"], result_df["primary_diagnosis"], normalize="index")
        st.dataframe((diag_pct * 100).round(2), use_container_width=True)
    else:
        st.info("Column 'primary_diagnosis' not found.")

with chart_tab2:
    if TARGET_COL in result_df.columns:
        readm_df = result_df.groupby("cluster")[TARGET_COL].mean().reset_index()
        fig_readm = px.bar(
            readm_df,
            x="cluster",
            y=TARGET_COL,
            title="Readmission Rate by Cluster",
            text=readm_df[TARGET_COL].map(lambda x: f"{x:.1%}"),
        )
        fig_readm.update_layout(yaxis_tickformat=".0%")
        st.plotly_chart(fig_readm, use_container_width=True)
    else:
        st.info("Column 'readmitted' not found.")

with chart_tab3:
    numeric_for_plot = [c for c in selected_num_cols if c in result_df.columns]
    if numeric_for_plot:
        feature_to_plot = st.selectbox("Choose a numeric feature", numeric_for_plot)
        fig_box = px.box(
            result_df,
            x="cluster",
            y=feature_to_plot,
            color="cluster",
            title=f"{feature_to_plot} Distribution by Cluster",
        )
        st.plotly_chart(fig_box, use_container_width=True)
    else:
        st.info("No numeric columns available for plotting.")

if mode == "Clustering":

    st.subheader("5) Business Insights Panel")
    insights = build_insights(result_df, cluster_col="cluster")
    for i, insight in enumerate(insights, start=1):
        st.markdown(f"**Insight {i}.** {insight}")

    st.subheader("6) Interpretation Notes for Your Report")
    with st.expander("Suggested discussion text"):
        discussion = f"""
        - The dataset was clustered into {n_clusters} patient segments using K-Means after encoding categorical features and scaling numeric variables.
        - The silhouette score is {silhouette:.3f}.
        - Cluster profiling shows how patient groups differ in age, procedures, hospital stay, comorbidity burden, diagnosis mix, and readmission behavior.
        - If cluster differences appear weak, this is still a valid finding: it may indicate limited feature separability or a relatively homogeneous patient population.
        - Recommended next steps include testing DBSCAN or hierarchical clustering, adding richer clinical features, and comparing clustering quality across methods.
        """
        st.code(textwrap.dedent(discussion).strip(), language="markdown")

    csv_bytes = result_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download clustered dataset",
        data=csv_bytes,
        file_name="clustered_patients.csv",
        mime="text/csv",
    )


