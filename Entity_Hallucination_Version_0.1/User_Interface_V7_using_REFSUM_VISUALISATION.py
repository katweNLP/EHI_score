import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import io  # For handling image downloads

def load_data(uploaded_file):
    """Load the CSV data into a Pandas DataFrame."""
    return pd.read_csv(uploaded_file)

def plot_ehi_entity_f1(df):
    """Plot EHI scores for each model and Entity F1 Score across article indices and return the figure."""
    fig, ax = plt.subplots(figsize=(16, 6))
    
    # Unique IDs and their corresponding index mapping
    unique_ids = df['Id'].unique()
    
    # Prepare data structures
    ehi_values = {model: [] for model in df['Model'].unique()}
    entity_f1_values = []
    
    # Populate values for each index
    for id_val in unique_ids:
        subset = df[df['Id'] == id_val]
        entity_f1_values.append(subset['Entity F1 Score'].mean())  # Mean Entity F1 Score per ID
        for model in subset['Model'].unique():
            ehi_values[model].append(subset[subset['Model'] == model]['EHI'].values[0])
    
    # Define x-axis indexes (0, 1, 2, ... up to total articles)
    x_indexes = list(range(len(unique_ids)))
    
    # Plot each model's EHI scores
    for model, values in ehi_values.items():
        ax.plot(x_indexes, values, label=f"{model} EHI", marker='o', linestyle='-')
    
    # Plot Entity F1 Score as a separate line
    ax.plot(x_indexes, entity_f1_values, label="Entity F1 Score", marker='s', linestyle='--', color='black')

    # Adjust x-axis labels every 40 articles
    num_articles = len(unique_ids)
    x_ticks = list(range(0, num_articles, 40))
    x_labels = [f"{i}-{i+39}" for i in x_ticks]
    
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels, rotation=45)  # Setting custom x-ticks for readability
    ax.set_xlabel("Article Index (Grouped by 40)")
    ax.set_ylabel("Metric Value")
    ax.set_title("EHI and Entity F1 Score Across Models")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.6)
    
    st.pyplot(fig)  # Display the plot in Streamlit
    
    return fig  # Return the figure for downloading

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

def plot_cdf_ehi_vs_entity_f1(df):
    """Plot the CDF of EHI scores across models vs Entity F1 Score with improved readability."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Get unique models
    models = df['Model'].unique()
    
    # Plot CDF for each model's EHI values
    for model in models:
        model_data = df[df['Model'] == model]['EHI'].dropna().values
        sorted_data = np.sort(model_data)
        cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        ax.plot(
            sorted_data, cdf, marker='.', linestyle='-', 
            linewidth=1.2, alpha=0.7, label=f"{model} EHI CDF"
        )

    # Plot CDF for Entity F1 Score with thinner line & smaller markers
    entity_f1_data = df['Entity F1 Score'].dropna().values
    sorted_entity_f1 = np.sort(entity_f1_data)
    cdf_entity_f1 = np.arange(1, len(sorted_entity_f1) + 1) / len(sorted_entity_f1)
    ax.plot(
        sorted_entity_f1, cdf_entity_f1, marker='.', markersize=3, linestyle='-', 
        linewidth=1, alpha=0.5, color='black', label="Entity F1 Score CDF"
    )
    
    ax.set_xlabel("Metric Value")
    ax.set_ylabel("Cumulative Probability")
    ax.set_title("CDF of EHI Across Models vs Entity F1 Score")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.6)
    
    st.pyplot(fig)  # Display the CDF plot in Streamlit
    
    return fig  # Return the figure for downloading
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

def plot_cdf_ehi(df):
    """Plot the CDF of EHI scores across models with improved readability."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Get unique models
    models = df['Model'].unique()
    
    # Plot CDF for each model's EHI values
    for model in models:
        model_data = df[df['Model'] == model]['EHI'].dropna().values
        sorted_data = np.sort(model_data)
        cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        ax.plot(
            sorted_data, cdf, marker='.', linestyle='-', 
            linewidth=1.2, alpha=0.7, label=f"{model} EHI CDF"
        )
    
    ax.set_xlabel("EHI Value")
    ax.set_ylabel("Cumulative Probability")
    ax.set_title("CDF of EHI Across Models")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.6)
    
    st.pyplot(fig)  # Display the CDF plot in Streamlit
    
    return fig  # Return the figure for downloading

def plot_modelwise_cdf(df):
    """Plot the CDF of EHI and Entity F1 Score for each model separately."""
    models = df['Model'].unique()
    
    for model in models:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # EHI CDF for the current model
        model_data = df[df['Model'] == model]['EHI'].dropna().values
        sorted_data = np.sort(model_data)
        cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        ax.plot(
            sorted_data, cdf, marker='.', linestyle='-', 
            linewidth=1.2, alpha=0.7, label=f"{model} EHI CDF"
        )

        # Entity F1 CDF for the current model
        entity_f1_data = df[df['Model'] == model]['Entity F1 Score'].dropna().values
        sorted_entity_f1 = np.sort(entity_f1_data)
        cdf_entity_f1 = np.arange(1, len(sorted_entity_f1) + 1) / len(sorted_entity_f1)
        ax.plot(
            sorted_entity_f1, cdf_entity_f1, marker='.', markersize=3, linestyle='-', 
            linewidth=1, alpha=0.5, color='black', label=f"{model} Entity F1 CDF"
        )

        ax.set_xlabel("Metric Value")
        ax.set_ylabel("Cumulative Probability")
        ax.set_title(f"CDF of EHI and Entity F1 Score for {model}")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.6)

        st.pyplot(fig)  # Display the plot in Streamlit
        
        # Save the plot for downloading
        img_buffer = io.BytesIO()
        fig.savefig(img_buffer, format="png", dpi=300, bbox_inches="tight")
        img_buffer.seek(0)

        st.download_button(
            label=f"📥 Download CDF for {model}",
            data=img_buffer,
            file_name=f"{model}_cdf_plot.png",
            mime="image/png"
        )
def calculate_mean_variance(df):
    """Calculate the mean and variance for EHI and Entity F1 Score for each model."""
    # Calculate mean and variance for each model and metric
    summary = []
    models = df['Model'].unique()

    for model in models:
        model_data = df[df['Model'] == model]
        
        # EHI mean and variance
        ehi_mean = model_data['EHI'].mean()
        ehi_variance = model_data['EHI'].var()
        
        # Entity F1 Score mean and variance
        entity_f1_mean = model_data['Entity F1 Score'].mean()
        entity_f1_variance = model_data['Entity F1 Score'].var()
        
        # Append the summary for the model
        summary.append([model, ehi_mean, ehi_variance, entity_f1_mean, entity_f1_variance])
    
    # Create a DataFrame for the summary
    summary_df = pd.DataFrame(summary, columns=['Model', 'EHI Mean', 'EHI Variance', 'Entity F1 Score Mean', 'Entity F1 Score Variance'])
    
    # Display the table in Streamlit
    st.subheader("Mean and Variance of EHI and Entity F1 Score Across Models")
    st.write(summary_df)
    
    return summary_df
def plot_comparison_graphs(df):
    """Plot comparison graphs of mean and variance for EHI and Entity F1 Score."""
    # Calculate mean and variance for EHI and Entity F1 Score
    summary_df = calculate_mean_variance(df)
    
    # Plot Mean comparison for EHI and Entity F1 Score
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.bar(summary_df['Model'], summary_df['EHI Mean'], width=0.4, label='EHI Mean', align='center', color='skyblue')
    ax1.bar(summary_df['Model'], summary_df['Entity F1 Score Mean'], width=0.4, label='Entity F1 Score Mean', align='edge', color='lightgreen')
    ax1.set_xlabel('Model')
    ax1.set_ylabel('Mean Value')
    ax1.set_title('Comparison of Mean: EHI vs Entity F1 Score')
    ax1.legend()
    st.pyplot(fig1)
    
    # Plot Variance comparison for EHI and Entity F1 Score
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.bar(summary_df['Model'], summary_df['EHI Variance'], width=0.4, label='EHI Variance', align='center', color='lightcoral')
    ax2.bar(summary_df['Model'], summary_df['Entity F1 Score Variance'], width=0.4, label='Entity F1 Score Variance', align='edge', color='gold')
    ax2.set_xlabel('Model')
    ax2.set_ylabel('Variance Value')
    ax2.set_title('Comparison of Variance: EHI vs Entity F1 Score')
    ax2.legend()
    st.pyplot(fig2)
import io
import pandas as pd
import streamlit as st

import io
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import streamlit as st
import matplotlib.pyplot as plt
import io
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import io

def normalize_column(series):
    """Normalize a numerical column to range [0,1]."""
    return (series - series.min()) / (series.max() - series.min())

def generate_metrics_table(df):
    """
    Generates a model-wise table displaying all metric values across the dataset,
    computes the average for each metric grouped by Model, and provides a download option as an image.
    """
    
    # Identifying metric columns (excluding non-metric columns like Id and Model)
    metric_columns = [col for col in df.columns if col not in ['Id', 'Model'] and df[col].dtype in ['float64', 'int64']]

    # Check if "EHI" exists and compute "EHI Normalized"
    if "EHI" in df.columns:
        df["EHI Normalized"] = normalize_column(df["EHI"])
        if "EHI Normalized" not in metric_columns:
            metric_columns.append("EHI Normalized")
    
    if not metric_columns:
        st.error("No numerical metric columns found in the dataset.")
        return
    
    # Compute the average of each metric grouped by Model
    avg_metrics = df.groupby("Model")[metric_columns].mean().reset_index()
    
    # Display the table in Streamlit
    st.subheader("📊 Model-wise Average Metrics Table")
    st.dataframe(avg_metrics)
    
    # Convert DataFrame to an image
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    ax.axis('tight')
    ax.axis('off')
    ax.table(cellText=avg_metrics.values, colLabels=avg_metrics.columns, cellLoc='center', loc='center')
    
    # Save table as an image
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format="png", dpi=300, bbox_inches="tight")
    img_buffer.seek(0)
    
    st.download_button(
        label="📥 Download Model-wise Metrics Table as PNG",
        data=img_buffer,
        file_name="modelwise_average_metrics.png",
        mime="image/png"
    )
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import io

def normalize_column(series):
    """Normalizes a Pandas Series using min-max scaling."""
    return (series - series.min()) / (series.max() - series.min()) if not series.isnull().all() else series

def generate_normalized_metrics_table(df):
    """
    Generates a model-wise table displaying all metric values across the dataset,
    computes the average for each metric grouped by Model, adds Normalized EHI,
    and provides a download option as an image.
    """

    # **Step 1: Check for duplicate column names**
    duplicated_cols = df.columns[df.columns.duplicated()].tolist()
    if duplicated_cols:
        st.error(f"Duplicate column names found: {duplicated_cols}")
        df = df.loc[:, ~df.columns.duplicated()].copy()
        st.warning("Duplicate columns removed.")

    # **Step 2: Identify metric columns (excluding non-metric columns like Id and Model)**
    metric_columns = [col for col in df.columns if col not in ['Id', 'Model'] and df[col].dtype in ['float64', 'int64']]

    if not metric_columns:
        st.error("No numerical metric columns found in the dataset.")
        return

    # **Step 3: Remove existing 'EHI_Normalized' column if it exists**
    if "EHI_Normalized" in df.columns:
        df.drop(columns=["EHI_Normalized"], inplace=True)

    # **Step 4: Normalize EHI before computing averages**
    if "EHI" in df.columns:
        df["EHI_Normalized"] = normalize_column(df["EHI"])
        metric_columns.append("EHI_Normalized")

    # **Step 5: Compute the average of each metric grouped by Model**
    avg_metrics = df.groupby("Model")[metric_columns].mean().reset_index()

    # **Step 6: Display the table in Streamlit**
    st.subheader("📊 Model-wise Average Metrics Table (with Normalized EHI)")
    st.dataframe(avg_metrics)

    # **Step 7: Convert DataFrame to an image**
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    ax.axis('tight')
    ax.axis('off')
    ax.table(cellText=avg_metrics.values, colLabels=avg_metrics.columns, cellLoc='center', loc='center')

    # **Step 8: Save table as an image**
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format="png", dpi=300, bbox_inches="tight")
    img_buffer.seek(0)

    # **Step 9: Provide download button**
    st.download_button(
        label="📥 Download Model-wise Normalized Metrics Table as PNG",
        data=img_buffer,
        file_name="modelwise_average_metrics.png",
        mime="image/png"
    )


import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

def plot_normalized_ehi_vs_entity_f1(df):
    """Plot the normalized scores of EHI vs Entity F1 Score across models."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Normalize EHI and Entity F1 Score
    df['Normalized EHI'] = (df['EHI'] - df['EHI'].min()) / (df['EHI'].max() - df['EHI'].min())
    df['Normalized Entity F1'] = (df['Entity F1 Score'] - df['Entity F1 Score'].min()) / (df['Entity F1 Score'].max() - df['Entity F1 Score'].min())

    # Get unique models
    models = df['Model'].unique()

    # Plot normalized EHI vs normalized Entity F1 Score for each model
    for model in models:
        model_data = df[df['Model'] == model]
        ax.scatter(
            model_data['Normalized EHI'], model_data['Normalized Entity F1'],
            alpha=0.7, label=f"{model}", s=50
        )

    ax.set_xlabel("Normalized EHI")
    ax.set_ylabel("Normalized Entity F1 Score")
    ax.set_title("Normalized EHI vs Entity F1 Score Across Models")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.6)
    
    st.pyplot(fig)  # Display the plot in Streamlit
    
    return fig  # Return the figure for downloading
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

def plot_normalized_metrics(df):
    """Plot a line plot of normalized EHI and Entity F1 Score with article index on the X-axis."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Normalize EHI and Entity F1 Score
    df['Normalized EHI'] = (df['EHI'] - df['EHI'].min()) / (df['EHI'].max() - df['EHI'].min())
    df['Normalized Entity F1'] = (df['Entity F1 Score'] - df['Entity F1 Score'].min()) / (df['Entity F1 Score'].max() - df['Entity F1 Score'].min())

    # Sort data by article index
    df = df.sort_values(by='Unnamed: 0')

    # Plot normalized EHI
    ax.plot(
        df['Unnamed: 0'], df['Normalized EHI'],
        marker='o', linestyle='-', linewidth=1.5, alpha=0.8, label="Normalized EHI"
    )

    # Plot normalized Entity F1 Score
    ax.plot(
        df['Unnamed: 0'], df['Normalized Entity F1'],
        marker='s', linestyle='-', linewidth=1.5, alpha=0.8, label="Normalized Entity F1 Score"
    )

    ax.set_xlabel("Article Index")
    ax.set_ylabel("Normalized Metric Value")
    ax.set_title("Line Plot of Normalized EHI and Entity F1 Score Across Articles")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.6)
    
    st.pyplot(fig)  # Display the plot in Streamlit
    
    return fig  # Return the figure for downloading
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

def plot_normalized_ehi_vs_entity_f1(df):
    """Plot a line plot of normalized EHI vs. Entity F1 Score with distinct colors and markers."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Normalize EHI and Entity F1 Score
    df['Normalized EHI'] = (df['EHI'] - df['EHI'].min()) / (df['EHI'].max() - df['EHI'].min())
    df['Normalized Entity F1'] = (df['Entity F1 Score'] - df['Entity F1 Score'].min()) / (df['Entity F1 Score'].max() - df['Entity F1 Score'].min())

    # Set color palette
    sns.set_palette("husl") 

    # Sort data by article index
    df = df.sort_values(by='Unnamed: 0')

    # Plot normalized EHI
    ax.plot(
        df['Unnamed: 0'], df['Normalized EHI'],
        marker='o', linestyle='-', linewidth=2, markersize=5, 
        alpha=0.8, color='blue', label="Normalized EHI"
    )

    # Plot normalized Entity F1 Score
    ax.plot(
        df['Unnamed: 0'], df['Normalized Entity F1'],
        marker='s', linestyle='--', linewidth=2, markersize=5, 
        alpha=0.8, color='red', label="Normalized Entity F1 Score"
    )

    # Labels and title
    ax.set_xlabel("Article Index")
    ax.set_ylabel("Normalized Metric Value")
    ax.set_title("Line Plot of Normalized EHI vs Entity F1 Score")
    
    # Grid, legend, and style tweaks
    ax.legend(fontsize=12, loc="best", fancybox=True, framealpha=0.6)
    ax.grid(True, linestyle="--", alpha=0.6)
    
    # Display the plot in Streamlit
    st.pyplot(fig)  
    
    return fig  # Return the figure for downloading
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from io import BytesIO

def normalized_ehi_entity_f1_plot_1(df):
    """Plot Normalized EHI scores for each model and Entity F1 Score across article indices, with a download option."""
    fig, ax = plt.subplots(figsize=(16, 6))
    
    # Unique article IDs and their corresponding index mapping
    unique_ids = df['Id'].unique()
    
    # Normalize EHI
    df['Normalized EHI'] = (df['EHI'] - df['EHI'].min()) / (df['EHI'].max() - df['EHI'].min())

    # Prepare data structures
    ehi_values = {model: [] for model in df['Model'].unique()}
    entity_f1_values = []

    # Populate values for each index
    for id_val in unique_ids:
        subset = df[df['Id'] == id_val]
        entity_f1_values.append(subset['Entity F1 Score'].mean())  # Mean Entity F1 Score per ID
        for model in subset['Model'].unique():
            ehi_values[model].append(subset[subset['Model'] == model]['Normalized EHI'].values[0])

    # Define x-axis indexes (0, 1, 2, ... up to total articles)
    x_indexes = list(range(len(unique_ids)))
    
    # Plot each model's Normalized EHI scores
    for model, values in ehi_values.items():
        ax.plot(x_indexes, values, label=f"{model} Normalized EHI", marker='o', linestyle='-')

    # Plot Entity F1 Score as a separate line
    ax.plot(x_indexes, entity_f1_values, label="Entity F1 Score", marker='s', linestyle='--', color='black')

    # Adjust x-axis labels every 40 articles
    num_articles = len(unique_ids)
    x_ticks = list(range(0, num_articles, 40))
    x_labels = [f"{i}-{i+39}" for i in x_ticks]
    
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels, rotation=45)  # Setting custom x-ticks for readability
    ax.set_xlabel("Article Index (Grouped by 40)")
    ax.set_ylabel("Normalized Metric Value")
    ax.set_title("Normalized EHI and Entity F1 Score Across Models")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.6)
    
    st.pyplot(fig)  # Display the plot in Streamlit

    # Download Button
    buf = BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    st.download_button(label="Download Plot as PNG", data=buf, file_name="normalized_ehi_vs_entity_f1.png", mime="image/png")

    return fig


import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import io

def normalize_column(column):
    """Min-Max Normalize a given column."""
    return (column - column.min()) / (column.max() - column.min())

import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import io

def normalize_column(series):
    """Normalize a numerical column to range [0,1]."""
    return (series - series.min()) / (series.max() - series.min())

def plot_modelwise_cdf_normalised(df):
    """Plot the CDF of Normalized EHI and Entity F1 Score for each model separately, along with a combined CDF plot."""
    
    st.subheader("📊 Model-wise CDF Plot: Normalized EHI vs Entity F1 Score")

    # Normalize EHI if not already done
    if "EHI_Normalized" not in df.columns:
        df["EHI_Normalized"] = normalize_column(df["EHI"])

    models = df['Model'].unique()

    # 📌 **Plot Combined CDF for All Models**
    fig_combined, ax_combined = plt.subplots(figsize=(10, 6))
    
    for model in models:
        # Extract model-specific data
        model_data = df[df['Model'] == model]['EHI_Normalized'].dropna().values
        sorted_data = np.sort(model_data)
        cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        
        entity_f1_data = df[df['Model'] == model]['Entity F1 Score'].dropna().values
        sorted_entity_f1 = np.sort(entity_f1_data)
        cdf_entity_f1 = np.arange(1, len(sorted_entity_f1) + 1) / len(sorted_entity_f1)

        # 📌 **Add to Combined Plot**
        ax_combined.plot(
            sorted_data, cdf, linestyle='-', linewidth=1.2, alpha=0.7, label=f"{model} Normalized EHI"
        )
        ax_combined.plot(
            sorted_entity_f1, cdf_entity_f1, linestyle='--', linewidth=1, alpha=0.5, label=f"{model} Entity F1"
        )

        # 📌 **Plot Model-wise CDF**
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(sorted_data, cdf, marker='.', linestyle='-', linewidth=1.2, alpha=0.7, label=f"{model} Normalized EHI CDF")
        ax.plot(sorted_entity_f1, cdf_entity_f1, marker='.', markersize=3, linestyle='-', linewidth=1, alpha=0.5, color='black', label=f"{model} Entity F1 CDF")

        ax.set_xlabel("Metric Value")
        ax.set_ylabel("Cumulative Probability")
        ax.set_title(f"CDF of Normalized EHI and Entity F1 Score for {model}")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.6)

        st.pyplot(fig)  # Display the plot in Streamlit

        # Save model-wise plot for downloading
        img_buffer = io.BytesIO()
        fig.savefig(img_buffer, format="png", dpi=300, bbox_inches="tight")
        img_buffer.seek(0)

        st.download_button(
            label=f"📥 Download CDF 1 for {model}",
            data=img_buffer,
            file_name=f"{model}_cdf_plot.png",
            mime="image/png"
        )

    # 📌 **Finalize Combined CDF Plot**
    ax_combined.set_xlabel("Metric Value")
    ax_combined.set_ylabel("Cumulative Probability")
    ax_combined.set_title("Combined CDF of Normalized EHI and Entity F1 Score Across Models")
    ax_combined.legend()
    ax_combined.grid(True, linestyle="--", alpha=0.6)

    st.subheader("🌎 Combined CDF Plot for All Models")
    st.pyplot(fig_combined)

    # Save combined plot for downloading
    img_combined_buffer = io.BytesIO()
    fig_combined.savefig(img_combined_buffer, format="png", dpi=300, bbox_inches="tight")
    img_combined_buffer.seek(0)

    st.download_button(
        label="📥 Download Combined CDF Plot",
        data=img_combined_buffer,
        file_name="combined_cdf_plot.png",
        mime="image/png"
    )
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import io

import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import io

def plot_modelwise_cdf_EHI(df):
    """Plot the CDF of EHI and Entity F1 Score for each model separately, along with a combined CDF plot."""
    
    st.subheader("📊 Model-wise CDF Plot: EHI vs Entity F1 Score")
    
    models = df['Model'].unique()
    colors = sns.color_palette("tab10", len(models))  # Unique colors for each model
    
    # 📌 **Plot Combined CDF for All Models**
    fig_combined, ax_combined = plt.subplots(figsize=(10, 6))
    
    for i, model in enumerate(models):
        # Extract model-specific data
        model_data = df[df['Model'] == model]['EHI'].dropna().values
        sorted_data = np.sort(model_data)
        cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        
        entity_f1_data = df[df['Model'] == model]['Entity F1 Score'].dropna().values
        sorted_entity_f1 = np.sort(entity_f1_data)
        cdf_entity_f1 = np.arange(1, len(sorted_entity_f1) + 1) / len(sorted_entity_f1)

        # 📌 **Add to Combined Plot**
        ax_combined.plot(
            sorted_data, cdf, linestyle='-', linewidth=1.5, alpha=0.8, color=colors[i], label=f"{model} EHI"
        )
        ax_combined.plot(
            sorted_entity_f1, cdf_entity_f1, linestyle='--', linewidth=1.2, alpha=0.6, color=colors[i], label=f"{model} Entity F1"
        )

        # 📌 **Plot Model-wise CDF**
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(sorted_data, cdf, linestyle='-', linewidth=1.5, alpha=0.8, color=colors[i], label=f"{model} EHI CDF")
        ax.plot(sorted_entity_f1, cdf_entity_f1, linestyle='--', linewidth=1.2, alpha=0.6, color='black', label=f"{model} Entity F1 CDF")

        ax.set_xlabel("Metric Value")
        ax.set_ylabel("Cumulative Probability")
        ax.set_title(f"CDF of EHI and Entity F1 Score for {model}")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.6)

        st.pyplot(fig)  # Display the plot in Streamlit

        # Save model-wise plot for downloading
        img_buffer = io.BytesIO()
        fig.savefig(img_buffer, format="png", dpi=300, bbox_inches="tight")
        img_buffer.seek(0)

        st.download_button(
            label=f"📥 Download CDF norm_EHI_EF1 for {model}",
            data=img_buffer,
            file_name=f"{model}_cdf_plot.png",
            mime="image/png"
        )

    # 📌 **Finalize Combined CDF Plot**
    ax_combined.set_xlabel("Metric Value")
    ax_combined.set_ylabel("Cumulative Probability")
    ax_combined.set_title("Combined CDF of EHI and Entity F1 Score Across Models")
    ax_combined.legend(ncol=2, fontsize=8)  # Adjust legend to prevent clutter
    ax_combined.grid(True, linestyle="--", alpha=0.6)

    st.subheader("🌎 Combined CDF Plot for All Models")
    st.pyplot(fig_combined)

    # Save combined plot for downloading
    img_combined_buffer = io.BytesIO()
    fig_combined.savefig(img_combined_buffer, format="png", dpi=300, bbox_inches="tight")
    img_combined_buffer.seek(0)

    st.download_button(
        label="📥 Download Combined norm EHI CDF Plot",
        data=img_combined_buffer,
        file_name="combined_cdf_plot.png",
        mime="image/png"
    )

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
def normalize_column(column):
    """Min-Max Normalize a given column."""
    return (column - column.min()) / (column.max() - column.min())


def calculate_mean_variance_with_normalized_EHI(df):
    """Calculate mean and variance for EHI, Normalized EHI, and Entity F1 Score."""
    summary_df = df.groupby("Model").agg(
        {
            "EHI": ["mean", "var"],
            "Entity F1 Score": ["mean", "var"]
        }
    ).reset_index()
    
    summary_df.columns = ["Model", "EHI Mean", "EHI Variance", "Entity F1 Score Mean", "Entity F1 Score Variance"]

    # Add Normalized EHI calculations
    df["EHI_Normalized"] = normalize_column(df["EHI"])
    normalized_summary = df.groupby("Model")["EHI_Normalized"].agg(["mean", "var"]).reset_index()
    normalized_summary.columns = ["Model", "EHI Normalized Mean", "EHI Normalized Variance"]
    
    # Merge the summary DataFrames
    summary_df = summary_df.merge(normalized_summary, on="Model")
    
    # Calculate statistics separately for Pegasus model in two parts
    if "google/pegasus-xsum" in df["Model"].unique():
        pegasus_df = df[df["Model"] == "google/pegasus-xsum"]
        pegasus_part1 = pegasus_df.iloc[0:200]  # Indices 0-199
        pegasus_part2 = pegasus_df.iloc[200:400]  # Indices 200-399
        
        pegasus_stats = pd.DataFrame({
            "Model": ["google/pegasus-xsum (0-199)", "google/pegasus-xsum (200-399)"],
            "EHI Mean": [pegasus_part1["EHI"].mean(), pegasus_part2["EHI"].mean()],
            "EHI Variance": [pegasus_part1["EHI"].var(), pegasus_part2["EHI"].var()],
        })
        
        summary_df = pd.concat([summary_df, pegasus_stats], ignore_index=True)
    
    return summary_df

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

def plot_comparison_graphs_with_normalized_EHI(df):
    """Plot comparison graphs of mean and variance for EHI, Normalized EHI, and Entity F1 Score."""
    
    # Compute Mean & Variance including Normalized EHI
    summary_df = calculate_mean_variance_with_normalized_EHI(df)

    # Generate x-axis positions
    x = np.arange(len(summary_df["Model"]))  # Position for each model
    width = 0.2  # Bar width to prevent overlapping

    # Plot Mean comparison
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.bar(x - width, summary_df["EHI Mean"], width=width, label="EHI Mean", color="skyblue")
    ax1.bar(x, summary_df["EHI Normalized Mean"], width=width, label="EHI Normalized Mean", color="blue")
    ax1.bar(x + width, summary_df["Entity F1 Score Mean"], width=width, label="Entity F1 Score Mean", color="lightgreen")
    
    ax1.set_xlabel("Model")
    ax1.set_ylabel("Mean Value")
    ax1.set_title("Comparison of Mean: EHI vs Normalized EHI vs Entity F1 Score")
    ax1.set_xticks(x)
    ax1.set_xticklabels(summary_df["Model"], rotation=45)
    ax1.legend()
    st.pyplot(fig1)

    # Plot Variance comparison
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.bar(x - width, summary_df["EHI Variance"], width=width, label="EHI Variance", color="lightcoral")
    ax2.bar(x, summary_df["EHI Normalized Variance"], width=width, label="EHI Normalized Variance", color="red")
    ax2.bar(x + width, summary_df["Entity F1 Score Variance"], width=width, label="Entity F1 Score Variance", color="gold")
    
    ax2.set_xlabel("Model")
    ax2.set_ylabel("Variance Value")
    ax2.set_title("Comparison of Variance: EHI vs Normalized EHI vs Entity F1 Score")
    ax2.set_xticks(x)
    ax2.set_xticklabels(summary_df["Model"], rotation=45)
    ax2.legend()
    st.pyplot(fig2)

# Streamlit UI
st.title("EHI and Entity F1 Score Visualization")

uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is not None:
    df = load_data(uploaded_file)
    st.write("Data Preview:")
    st.write(df.head())
    
    st.subheader("Graph of EHI & Entity F1 Score")
    fig1 = plot_ehi_entity_f1(df)  # Get the figure object

    # Add Download Button for the First Figure
    img_buffer1 = io.BytesIO()
    fig1.savefig(img_buffer1, format="png", dpi=300, bbox_inches="tight")
    img_buffer1.seek(0)

    st.download_button(
        label="📥 Download EHI vs Entity F1 Graph",
        data=img_buffer1,
        file_name="ehi_entity_f1_plot.png",
        mime="image/png"
    )

    st.subheader("CDF of EHI Across Models vs Entity F1 Score")
    fig2 = plot_cdf_ehi_vs_entity_f1(df)  # Get the CDF figure object

    # Add Download Button for the CDF Figure
    img_buffer2 = io.BytesIO()
    fig2.savefig(img_buffer2, format="png", dpi=300, bbox_inches="tight")
    img_buffer2.seek(0)

    st.download_button(
        label="📥 Download CDF Graph",
        data=img_buffer2,
        file_name="cdf_ehi_entity_f1_plot.png",
        mime="image/png"
    )
    
      # Plot Model-wise CDF for each model
    st.subheader("CDF of EHI and Entity F1 Score for Each Model")
    plot_modelwise_cdf(df)  # Call the new function to plot CDF for each model
    plot_cdf_ehi(df)
    
    # Plot comparison of mean and variance for EHI and Entity F1 Score
    plot_comparison_graphs(df)
    
    generate_metrics_table(df)
    plot_normalized_ehi_vs_entity_f1(df)
    normalized_ehi_entity_f1_plot_1(df)
    plot_modelwise_cdf_normalised(df)
    
    st.write("Normalized Metrics for variance and mean Plot")
    plot_comparison_graphs_with_normalized_EHI(df)
    plot_modelwise_cdf_EHI(df)
    # generate_normalized_metrics_table(df)
