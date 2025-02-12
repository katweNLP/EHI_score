import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from io import BytesIO

def plot_modelwise(df):
    models = df['Model'].unique()
    for model in models:
        model_df = df[df['Model'] == model]
        plt.figure(figsize=(10, 5))
        sns.lineplot(x=model_df['Sentence ID'], y=model_df['EHI'], label='EHI')
        sns.lineplot(x=model_df['Sentence ID'], y=model_df['Entity F1 Score'], label='Entity F1 Score')
        plt.xlabel("Sentence ID")
        plt.ylabel("Metric Value")
        plt.title(f"EHI vs Entity F1 Score for {model}")
        plt.legend()
        st.pyplot(plt)

def plot_combined(df):
    plt.figure(figsize=(10, 5))
    sns.lineplot(x=df['Sentence ID'], y=df['EHI'], hue=df['Model'], style=df['Model'])
    sns.lineplot(x=df['Sentence ID'], y=df['Entity F1 Score'], hue=df['Model'], style=df['Model'], linestyle="dashed")
    plt.xlabel("Sentence ID")
    plt.ylabel("Metric Value")
    plt.title("EHI vs Entity F1 Score (Combined)")
    plt.legend()
    st.pyplot(plt)

def plot_cdf(df):
    plt.figure(figsize=(10, 5))
    for model in df['Model'].unique():
        model_df = df[df['Model'] == model]
        sns.ecdfplot(model_df['EHI'], label=f"EHI - {model}")
        sns.ecdfplot(model_df['Entity F1 Score'], label=f"Entity F1 - {model}")
    plt.xlabel("Metric Value")
    plt.ylabel("CDF")
    plt.title("CDF of EHI vs Entity F1 Score")
    plt.legend()
    st.pyplot(plt)
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

def plot_cdf_normalised(df):
    """
    Computes Normalized EHI and plots the CDF of EHI, Normalized EHI, and Entity F1 Score for each model.
    """
    # Avoid division by zero
    if df["EHI"].max() == df["EHI"].min():
        st.error("EHI values are constant, normalization is not possible.")
        return
    
    # Compute Normalized EHI = (EHI - min) / (max - min)
    df["EHI Normalized"] = (df["EHI"] - df["EHI"].min()) / (df["EHI"].max() - df["EHI"].min())

    plt.figure(figsize=(10, 5))
    
    for model in df['Model'].unique():
        model_df = df[df['Model'] == model]
        
        sns.ecdfplot(model_df['EHI'], label=f"EHI - {model}", linestyle="dashed", color="blue")
        sns.ecdfplot(model_df['EHI Normalized'], label=f"EHI Normalized - {model}", linestyle="solid", color="red")
        sns.ecdfplot(model_df['Entity F1 Score'], label=f"Entity F1 - {model}", linestyle="dotted", color="green")
    
    plt.xlabel("Metric Value")
    plt.ylabel("CDF")
    plt.title("CDF of EHI, Normalized EHI, and Entity F1 Score")
    plt.legend()
    st.pyplot(plt)

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO

def plot_mean_variance(df):
    # Select only EHI and Entity F1 Score
    selected_metrics = ["EHI", "Entity F1 Score"]
    df_filtered = df[selected_metrics]

    # Compute mean and variance
    mean_vals = df_filtered.mean()
    var_vals = df_filtered.var()
    st.write("Mean Values:",mean_vals)
    st.write("Variance Values:",var_vals)
    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    width = 0.4
    x = np.arange(len(selected_metrics))
    ax.bar(x - width/2, mean_vals, width, label='Mean', color='b', alpha=0.7)
    ax.bar(x + width/2, var_vals, width, label='Variance', color='r', alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels(selected_metrics, rotation=45)
    ax.set_ylabel("Value")
    ax.set_title("Mean and Variance of EHI & Entity F1 Score")
    ax.legend()
    st.pyplot(fig)

    # Save the figure to a buffer
    img_buf = BytesIO()
    fig.savefig(img_buf, format="png")
    img_buf.seek(0)

    # Download button
    st.download_button(
        label="Download Plot",
        data=img_buf,
        file_name="mean_variance_plot.png",
        mime="image/png"
    )
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from io import BytesIO

def plot_mean_variance_normalised(df):
    """
    Computes Normalized EHI, calculates mean & variance for EHI, EHI Normalized, and Entity F1 Score,
    and plots a grouped bar chart of mean & variance (without overlap).
    """
    # Compute Normalized EHI using Min-Max Scaling
    if df["EHI"].max() == df["EHI"].min():
        st.error("EHI values are constant, normalization is not possible.")
        return
    
    df["EHI Normalized"] = (df["EHI"] - df["EHI"].min()) / (df["EHI"].max() - df["EHI"].min())

    # Select only required metrics
    selected_metrics = ["EHI", "EHI Normalized", "Entity F1 Score"]
    df_filtered = df[selected_metrics]

    # Compute mean and variance
    mean_vals = df_filtered.mean()
    var_vals = df_filtered.var()

    st.write("📊 **Mean Values:**", mean_vals)
    st.write("📈 **Variance Values:**", var_vals)

    # Plot with no overlap
    fig, ax = plt.subplots(figsize=(10, 6))
    width = 0.3  # Reduce width to prevent overlap
    x = np.arange(len(selected_metrics))  # X-axis positions

    ax.bar(x, mean_vals, width, label='Mean', color='royalblue', alpha=0.8)
    ax.bar(x + width, var_vals, width, label='Variance', color='orangered', alpha=0.8)

    ax.set_xticks(x + width / 2)  # Center labels between bars
    ax.set_xticklabels(selected_metrics, rotation=30, ha="right", fontsize=12)
    ax.set_ylabel("Value", fontsize=12)
    ax.set_title("Mean & Variance of EHI, Normalized EHI & Entity F1 Score", fontsize=14, fontweight='bold')
    ax.legend()
    
    # Display the plot in Streamlit
    st.pyplot(fig)

    # Save the figure to a buffer
    img_buf = BytesIO()
    fig.savefig(img_buf, format="png", bbox_inches="tight")
    img_buf.seek(0)

    # Download button
    st.download_button(
        label="📥 Download Plot",
        data=img_buf,
        file_name="mean_variance_plot.png",
        mime="image/png"
    )

def plot_frequency_distribution(df):
    numerical_metrics = df.select_dtypes(include=[np.number]).columns.tolist()
    models = df['Model'].unique()
    
    for model in models:
        model_df = df[df['Model'] == model]
        fig, axes = plt.subplots(len(numerical_metrics), 1, figsize=(10, 5 * len(numerical_metrics)))
        
        if len(numerical_metrics) == 1:
            axes = [axes]
        
        for ax, metric in zip(axes, numerical_metrics):
            sns.histplot(model_df[metric], bins=20, kde=True, ax=ax)
            ax.set_title(f"Frequency Distribution of {metric} for {model}")
            ax.set_xlabel(metric)
            ax.set_ylabel("Frequency")
        
        st.pyplot(fig)
def plot_frequency_distribution_table(df):
    numerical_metrics = df.select_dtypes(include=[np.number]).columns.tolist()
    models = df['Model'].unique()
    
    for model in models:
        model_df = df[df['Model'] == model]
        st.subheader(f"Frequency Distribution Table for {model}")

        # Create frequency tables for each metric
        for metric in numerical_metrics:
            st.write(f"**{metric}**")
            freq_table = pd.cut(model_df[metric], bins=10).value_counts().sort_index()
            freq_df = pd.DataFrame({"Range": freq_table.index.astype(str), "Frequency": freq_table.values})
            st.dataframe(freq_df)

        # Generate histograms
        fig, axes = plt.subplots(len(numerical_metrics), 1, figsize=(10, 5 * len(numerical_metrics)))
        
        if len(numerical_metrics) == 1:
            axes = [axes]
        
        for ax, metric in zip(axes, numerical_metrics):
            sns.histplot(model_df[metric], bins=10, kde=True, ax=ax)
            ax.set_title(f"Frequency Distribution of {metric} for {model}")
            ax.set_xlabel(metric)
            ax.set_ylabel("Frequency")
        
        st.pyplot(fig)

def generate_download_link(fig, filename):
    buf = BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    st.download_button(label="Download Plot", data=buf, file_name=filename, mime="image/png")
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Function to plot metrics model-wise
def plot_metrics_by_model(df):
    st.header("📊 मॉडल-वार मीट्रिक प्लॉट (Model-wise Metric Plots)")

    numeric_cols = ['EHI', 'Extractiveness Factor (EF)', 'Positive Hallucination (PH)', 
                    'Over Focus (OF)', 'Negative Hallucination (NH)', 'Lost Focus (LF)', 
                    'Entity F1 Score']
    
    models = df['Model'].unique()

    for metric in numeric_cols:
        fig, ax = plt.subplots(figsize=(8, 5))
        for model in models:
            subset = df[df['Model'] == model]
            ax.plot(subset["Sentence ID"], subset[metric], marker='o', label=model)
        
        ax.set_title(f"{metric} across Models")
        ax.set_xlabel("Sentence ID")
        ax.set_ylabel(metric)
        ax.legend()
        ax.grid(True)
        
        st.pyplot(fig)
        
        # Provide download button for the plot
        st.download_button(
            label=f"Download {metric} Plot",
            data=fig_to_bytes(fig),
            file_name=f"{metric}_plot.png",
            mime="image/png"
        )

# Helper function to convert Matplotlib figure to bytes
import io
def fig_to_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    return buf.read()

# Function to generate and display average metrics table
def generate_avg_metrics_table(df):
    st.header("📊 औसत मीट्रिक्स सारणी (Average Metrics Table)")

    numeric_cols = ['EHI', 'Extractiveness Factor (EF)', 'Positive Hallucination (PH)', 
                    'Over Focus (OF)', 'Negative Hallucination (NH)', 'Lost Focus (LF)', 
                    'Entity F1 Score']

    avg_metrics_df = df.groupby("Model")[numeric_cols].mean()

    # Format table for better readability
    styled_df = avg_metrics_df.style.format("{:.4f}")

    st.dataframe(styled_df)

    # Provide download button for CSV
    csv = avg_metrics_df.to_csv().encode("utf-8")
    st.download_button("डाउनलोड करें (Download Average Metrics CSV)", csv, "average_metrics.csv", "text/csv")
    
import streamlit as st

def normalize_column(series):
    """Normalize a numerical column to range [0,1]."""
    return (series - series.min()) / (series.max() - series.min())

# Function to generate and display average metrics table
def generate_avg_metrics_table_normalised(df):
    st.header("📊 औसत मीट्रिक्स सारणी (Average Metrics Table)")

    numeric_cols = ['EHI', 'Extractiveness Factor (EF)', 'Positive Hallucination (PH)', 
                    'Over Focus (OF)', 'Negative Hallucination (NH)', 'Lost Focus (LF)', 
                    'Entity F1 Score']

    # Compute Normalized EHI if 'EHI' exists
    if "EHI" in df.columns:
        df["EHI Normalized"] = normalize_column(df["EHI"])
        numeric_cols.append("EHI Normalized")

    # Compute the average metrics per model
    avg_metrics_df = df.groupby("Model")[numeric_cols].mean()

    # Format table for better readability
    styled_df = avg_metrics_df.style.format("{:.4f}")

    st.dataframe(styled_df)

    # Provide download button for CSV
    csv = avg_metrics_df.to_csv().encode("utf-8")
    st.download_button("डाउनलोड करें (Download Average Metrics norma CSV)", csv, "average_metrics.csv", "text/csv")

import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from io import BytesIO
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from io import BytesIO

def normalize_column(column):
    return (column - column.min()) / (column.max() - column.min())

def get_image_download_link(fig, filename):
    img_buf = BytesIO()
    fig.savefig(img_buf, format="png", dpi=300, bbox_inches='tight')
    img_buf.seek(0)
    st.download_button(label="Download Plot", data=img_buf, file_name=filename, mime="image/png")

def plot_normalized_combined(df):
    # Normalize EHI and Entity F1 Score
    df["EHI_normalized"] = normalize_column(df["EHI"])
    df["Entity_F1_normalized"] = normalize_column(df["Entity F1 Score"])

    # Set figure size and style
    plt.figure(figsize=(12, 6))
    sns.set_style("whitegrid")

    # Define color palette
    colors = sns.color_palette("tab10", n_colors=len(df["Model"].unique()))

    # Plot EHI and Entity F1 Score with improved styling
    sns.lineplot(x=df["Sentence ID"], y=df["EHI_normalized"], hue=df["Model"], style=df["Model"], 
                 alpha=0.8, linewidth=2, palette=colors, marker="o")
    sns.lineplot(x=df["Sentence ID"], y=df["Entity_F1_normalized"], hue=df["Model"], style=df["Model"], 
                 linestyle="dashed", alpha=0.7, linewidth=2, palette=colors)

    # Improve axis labels
    plt.xlabel("Sentence ID", fontsize=14, fontweight='bold')
    plt.ylabel("Normalized Metric Value", fontsize=14, fontweight='bold')
    plt.title("Normalized EHI vs Normalized Entity F1 Score (Combined)", fontsize=16, fontweight='bold')

    # Adjust x-ticks dynamically
    xticks = list(range(0, 1600, 200)) + list(range(2000, 9000, 2000))
    plt.xticks(xticks, rotation=45, fontsize=12)

    # Improve legend placement
    plt.legend(title="Model", loc="upper right", fontsize=12)

    # Save the plot for download
    fig = plt.gcf()
    get_image_download_link(fig, "improved_normalized_plot.png")

    # Display the plot in Streamlit
    st.pyplot(fig)
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from io import BytesIO

def normalize_column(column):
    return (column - column.min()) / (column.max() - column.min())

def get_image_download_link(fig, filename):
    img_buf = BytesIO()
    fig.savefig(img_buf, format="png", dpi=300, bbox_inches='tight')
    img_buf.seek(0)
    st.download_button(label="Download Plot", data=img_buf, file_name=filename, mime="image/png")

def reduce_ticks(ax, x_values, step_small=200, step_large=3000, threshold=2000):
    """ Reduce the number of X-axis ticks dynamically """
    xticks = [x for x in x_values if x < threshold and x % step_small == 0] + \
             [x for x in x_values if x >= threshold and x % step_large == 0]
    ax.set_xticks(xticks)

def plot_normalized_metrics(df):
    df["EHI_normalized"] = normalize_column(df["EHI"])
    df["Entity_F1_normalized"] = normalize_column(df["Entity F1 Score"])

    sns.set_style("whitegrid")
    colors = sns.color_palette("tab10", n_colors=len(df["Model"].unique()))
    model_colors = {model: color for model, color in zip(df["Model"].unique(), colors)}
    model_colors["RefSum"] = "#C0C0C0"  # Light gray for RefSum

    fig1, ax1 = plt.subplots(figsize=(10, 5))
    sns.lineplot(x=df["Sentence ID"], y=df["EHI_normalized"], hue=df["Model"], style=df["Model"], 
                 alpha=0.7, linewidth=1.5, palette=model_colors, marker="o", markersize=4, ax=ax1)

    ax1.set_xlabel("Sentence ID", fontsize=12, fontweight='bold')
    ax1.set_ylabel("Normalized EHI", fontsize=12, fontweight='bold')
    ax1.set_title("Normalized EHI (Model-wise)", fontsize=14, fontweight='bold')
    ax1.legend(title="Model", loc="upper right", fontsize=10)
    reduce_ticks(ax1, df["Sentence ID"])
    ax1.yaxis.set_major_locator(plt.MaxNLocator(nbins=5))  # Reduce Y-axis labels
    plt.xticks(rotation=30, fontsize=10)
    plt.tight_layout()

    get_image_download_link(fig1, "normalized_EHI_plot.png")
    st.pyplot(fig1)

    fig2, ax2 = plt.subplots(figsize=(10, 5))
    sns.lineplot(x=df["Sentence ID"], y=df["Entity_F1_normalized"], hue=df["Model"], style=df["Model"], 
                 linestyle="dashed", alpha=0.6, linewidth=1.5, palette=model_colors, marker="s", markersize=4, ax=ax2)

    ax2.set_xlabel("Sentence ID", fontsize=12, fontweight='bold')
    ax2.set_ylabel("Normalized Entity F1 Score", fontsize=12, fontweight='bold')
    ax2.set_title("Normalized Entity F1 Score (Model-wise)", fontsize=14, fontweight='bold')
    ax2.legend(title="Model", loc="upper right", fontsize=10)
    reduce_ticks(ax2, df["Sentence ID"])
    ax2.yaxis.set_major_locator(plt.MaxNLocator(nbins=5))
    plt.xticks(rotation=30, fontsize=10)
    plt.tight_layout()

    get_image_download_link(fig2, "normalized_Entity_F1_plot.png")
    st.pyplot(fig2)
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from io import BytesIO

def plot_mean_variance_normalised(df):
    """
    Computes Normalized EHI, calculates mean & variance for EHI, EHI Normalized, and Entity F1 Score,
    and plots a grouped bar chart of mean & variance (without overlap).
    """
    # Compute Normalized EHI using Min-Max Scaling
    if df["EHI"].max() == df["EHI"].min():
        st.error("EHI values are constant, normalization is not possible.")
        return
    
    df["EHI Normalized"] = (df["EHI"] - df["EHI"].min()) / (df["EHI"].max() - df["EHI"].min())

    # Select only required metrics
    selected_metrics = ["EHI", "EHI Normalized", "Entity F1 Score"]
    df_filtered = df[selected_metrics]

    # Compute mean and variance
    mean_vals = df_filtered.mean()
    var_vals = df_filtered.var()

    st.write("📊 **Mean Values:**", mean_vals)
    st.write("📈 **Variance Values:**", var_vals)

    # Plot with no overlap
    fig, ax = plt.subplots(figsize=(10, 6))
    width = 0.3  # Reduce width to prevent overlap
    x = np.arange(len(selected_metrics))  # X-axis positions

    ax.bar(x, mean_vals, width, label='Mean', color='royalblue', alpha=0.8)
    ax.bar(x + width, var_vals, width, label='Variance', color='orangered', alpha=0.8)

    ax.set_xticks(x + width / 2)  # Center labels between bars
    ax.set_xticklabels(selected_metrics, rotation=30, ha="right", fontsize=12)
    ax.set_ylabel("Value", fontsize=12)
    ax.set_title("Mean & Variance of EHI, Normalized EHI & Entity F1 Score", fontsize=14, fontweight='bold')
    ax.legend()
    
    # Display the plot in Streamlit
    st.pyplot(fig)

    # Save the figure to a buffer
    img_buf = BytesIO()
    fig.savefig(img_buf, format="png", bbox_inches="tight")
    img_buf.seek(0)

    # Download button
    st.download_button(
        label="📥 Download Plot",
        data=img_buf,
        file_name="mean_variance_plot.png",
        mime="image/png"
    )

def main():
    st.title("EHI vs Entity F1 Score Analysis")
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        df['Sentence ID'] = df['Sentence ID'].astype(int)
        
        st.subheader("Model-wise Line Plots")
        plot_modelwise(df)
        
        st.subheader("Combined Model Line Plot")
        plot_combined(df)
        
        st.subheader("CDF Plot")
        plot_cdf(df)
        
        st.subheader("Mean and Variance Bar Chart")
        plot_mean_variance(df)
        
        st.subheader("Frequency Distribution of Metrics")
        plot_frequency_distribution(df)
        
        st.subheader("Frequency Distribution Table")
        plot_frequency_distribution_table(df)
        st.subheader("Average Metrics Table")
        avg_table = df.groupby('Model').mean().reset_index()
        st.dataframe(avg_table)
        
        # Generate and display the average metrics table
        generate_avg_metrics_table(df)
        
        # Generate and display model-wise metric plots
        plot_metrics_by_model(df)
        
        # Generate and display normalized EHI vs Entity F1 Score plot
        st.write("Normalized EHI vs Normalized Entity F1 Score (Combined)")
        plot_normalized_combined(df)
        plot_normalized_metrics(df)
        generate_avg_metrics_table_normalised(df)
        plot_cdf_normalised(df)
        plot_mean_variance_normalised(df)
        
if __name__ == "__main__":
    main()