import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from io import BytesIO

def normalize_column(series):
    """Normalize a numerical column to range [0,1]."""
    return (series - series.min()) / (series.max() - series.min())

def get_image_download_link(fig, filename):
    """Generate a download button for the given plot."""
    img_buffer = BytesIO()
    fig.savefig(img_buffer, format="png", dpi=300, bbox_inches="tight")
    img_buffer.seek(0)
    
    st.download_button(
        label=f"📥 Download {filename}",
        data=img_buffer,
        file_name=filename,
        mime="image/png"
    )

import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator  # For automatic tick spacing

import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator  # For automatic tick spacing

def plot_modelwise_ehi_vs_entity_f1(data):
    # Ensure unique ID-Model pairs
    data = data.drop_duplicates(subset=['id', 'Model'])

    # Create a mapping of long IDs to shorter numeric labels (0, 1, 2, ...)
    unique_ids = sorted(data['id'].unique())  # Get unique IDs in sorted order
    id_mapping = {orig_id: new_id for new_id, orig_id in enumerate(unique_ids)}

    # Replace actual IDs with new numeric labels
    data['id_numeric'] = data['id'].map(id_mapping)

    # Figure 1: Combined Model-wise EHI vs. Entity F1 Score
    fig, ax = plt.subplots(figsize=(12, 7))
    
    models = data['Model'].unique()  # Get unique models
    
    for model in models:
        model_data = data[data['Model'] == model].set_index('id_numeric').sort_index()
        ax.plot(model_data.index, model_data['EHI'], label=f'EHI - {model}', marker='o', linewidth=2)
    
    ax.set_xlabel('ID (Renamed Sequentially)')
    ax.set_ylabel('EHI', color='blue')
    ax.tick_params(axis='y', labelcolor='blue')
    ax.xaxis.set_major_locator(MaxNLocator(nbins=15))  # Automatically adjust X-axis ticks

    ax2 = ax.twinx()
    
    for model in models:
        model_data = data[data['Model'] == model].set_index('id_numeric').sort_index()
        ax2.plot(model_data.index, model_data['Entity F1 Score'], label=f'Entity F1 - {model}', linestyle='dashed', marker='x', linewidth=2)
    
    ax2.set_ylabel('Entity F1 Score', color='green')
    ax2.tick_params(axis='y', labelcolor='green')
    
    ax.legend(loc='upper left', fontsize=8)
    ax2.legend(loc='upper right', fontsize=8)
    
    plt.xticks(rotation=45)  # Rotate X-axis labels for better readability
    st.pyplot(fig)
    get_image_download_link(fig, "modelwise_ehi_vs_entity_f1.png")

    # Figure 2: Model-wise EHI
    fig_ehi, ax_ehi = plt.subplots(figsize=(12, 7))
    
    for model in models:
        model_data = data[data['Model'] == model].set_index('id_numeric').sort_index()
        ax_ehi.plot(model_data.index, model_data['EHI'], label=f'EHI - {model}', marker='o', linewidth=2)
    
    ax_ehi.set_xlabel('ID (Renamed Sequentially)')
    ax_ehi.set_ylabel('EHI')
    ax_ehi.xaxis.set_major_locator(MaxNLocator(nbins=15))  # Adjust X-axis ticks
    ax_ehi.legend(title="Model")
    
    plt.xticks(rotation=45)
    st.pyplot(fig_ehi)
    get_image_download_link(fig_ehi, "modelwise_ehi.png")

    # Figure 3: Model-wise Entity F1 Score
    fig_f1, ax_f1 = plt.subplots(figsize=(12, 7))
    
    for model in models:
        model_data = data[data['Model'] == model].set_index('id_numeric').sort_index()
        ax_f1.plot(model_data.index, model_data['Entity F1 Score'], label=f'Entity F1 - {model}', marker='x', linewidth=2)
    
    ax_f1.set_xlabel('ID (Renamed Sequentially)')
    ax_f1.set_ylabel('Entity F1 Score')
    ax_f1.xaxis.set_major_locator(MaxNLocator(nbins=15))  # Adjust X-axis ticks
    ax_f1.legend(title="Model")
    
    plt.xticks(rotation=45)
    st.pyplot(fig_f1)
    get_image_download_link(fig_f1, "modelwise_entity_f1.png")

import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

def plot_cdf_ehi_vs_entity_f1(data):
    # Figure 1: Combined CDF plot for all models
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.ecdfplot(data['EHI'], label='EHI', color='blue', ax=ax)
    sns.ecdfplot(data['Entity F1 Score'], label='Entity F1 Score', color='green', ax=ax)
    
    ax.set_title('CDF of EHI vs Entity F1 Score (All Models)')
    ax.legend()
    
    st.pyplot(fig)
    get_image_download_link(fig, "cdf_ehi_vs_entity_f1.png")

    # Figure 2: Separate CDF plots for EHI per model
    fig_ehi, ax_ehi = plt.subplots(figsize=(8, 6))
    for model, group in data.groupby("Model"):
        sns.ecdfplot(group['EHI'], label=f'{model}', ax=ax_ehi)
    
    ax_ehi.set_title('CDF of EHI (Model-wise)')
    ax_ehi.legend(title="Model")
    
    st.pyplot(fig_ehi)
    get_image_download_link(fig_ehi, "cdf_ehi_modelwise.png")

    # Figure 3: Separate CDF plots for Entity F1 Score per model
    fig_f1, ax_f1 = plt.subplots(figsize=(8, 6))
    for model, group in data.groupby("Model"):
        sns.ecdfplot(group['Entity F1 Score'], label=f'{model}', ax=ax_f1)
    
    ax_f1.set_title('CDF of Entity F1 Score (Model-wise)')
    ax_f1.legend(title="Model")
    
    st.pyplot(fig_f1)
    get_image_download_link(fig_f1, "cdf_entity_f1_modelwise.png")

def generate_metrics_table(df):
    metric_columns = [col for col in df.columns if col not in ['id', 'Model'] and df[col].dtype in ['float64', 'int64']]
    avg_metrics = df.groupby("Model")[metric_columns].mean().reset_index()
    st.dataframe(avg_metrics)
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_mean_variance_ehi_entity_f1(data):
    # Compute mean and variance
    stats_df = data.groupby("Model").agg(
        EHI_Mean=("EHI", "mean"),
        EHI_Variance=("EHI", "var"),
        F1_Mean=("Entity F1 Score", "mean"),
        F1_Variance=("Entity F1 Score", "var")
    ).reset_index()

    # Figure 1: Mean Bar Chart
    fig_mean, ax_mean = plt.subplots(figsize=(8, 6))
    stats_df.plot(x="Model", y=["EHI_Mean", "F1_Mean"], kind="bar", ax=ax_mean)
    ax_mean.set_title("Mean of EHI & Entity F1 Score per Model")
    ax_mean.set_ylabel("Mean Value")
    ax_mean.legend(["EHI Mean", "Entity F1 Score Mean"])
    
    st.pyplot(fig_mean)
    get_image_download_link(fig_mean, "mean_ehi_f1_modelwise.png")

    # Figure 2: Variance Bar Chart
    fig_var, ax_var = plt.subplots(figsize=(8, 6))
    stats_df.plot(x="Model", y=["EHI_Variance", "F1_Variance"], kind="bar", ax=ax_var)
    ax_var.set_title("Variance of EHI & Entity F1 Score per Model")
    ax_var.set_ylabel("Variance Value")
    ax_var.legend(["EHI Variance", "Entity F1 Score Variance"])
    
    st.pyplot(fig_var)
    get_image_download_link(fig_var, "variance_ehi_f1_modelwise.png")

def main():
    st.title('EHI and Entity F1 Score Analysis')
    uploaded_file = st.file_uploader("Upload your CSV file", type="csv")
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write(data.head())
        if st.button("Plot EHI vs Entity F1"):
            plot_modelwise_ehi_vs_entity_f1(data)
        if st.button("Plot CDF of EHI vs Entity F1"):
            plot_cdf_ehi_vs_entity_f1(data)
        if st.button("Generate Model-wise Metrics Table"):
            generate_metrics_table(data)
        if st.button("Plot Mean and Variance of EHI and Entity F1"):
            plot_mean_variance_ehi_entity_f1(data)
if __name__ == "__main__":
    main()
