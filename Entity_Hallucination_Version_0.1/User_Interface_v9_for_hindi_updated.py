import streamlit as st
import pandas as pd
import json

# Function to read Hindi text from a file
def read_text_file(uploaded_file):
    try:
        content = uploaded_file.read().decode("utf-8").strip()
        if not content:
            st.error("फ़ाइल खाली है! (The file is empty!)")
            return None
        return content  # Return full text
    except Exception as e:
        st.error(f"Error reading text file: {e}")
        return None
import streamlit as st
import pandas as pd
import numpy as np
import json
from typing import List, Tuple, Union
import stanza
from typing import Tuple
import spacy
import stanza

import streamlit as st
import pandas as pd
import spacy
import json
import spacy
import streamlit as st
from transformers import pipeline

# Load SpaCy multilingual NER model
nlp_hi = spacy.load("xx_ent_wiki_sm")

# Load AI4Bharat's IndicNER model
try:
    ner_pipeline = pipeline("ner", model="ai4bharat/indicner-hindi")
    ai4bharat_available = True
except:
    ai4bharat_available = False
    st.warning("AI4Bharat IndicNER model could not be loaded. Using only SpaCy.")

def extract_entities(text: str) -> set:
    """
    Extract named entities from a given Hindi text using both SpaCy and AI4Bharat's IndicNER.
    """
    entities_spacy = {ent.text for ent in nlp_hi(text).ents}  # Extract entities from SpaCy
    st.write(f"🔵 **SpaCy Entities:** {entities_spacy}")

    entities_ai4bharat = set()
    if ai4bharat_available:
        entities_ai4bharat = {entity["word"] for entity in ner_pipeline(text)}
        st.write(f"🟢 **AI4Bharat IndicNER Entities:** {entities_ai4bharat}")

    # Combine results from both models
    combined_entities = entities_spacy.union(entities_ai4bharat)

    # st.write(f"✅ **Final Extracted Entities (Merged):** {combined_entities}")

    return combined_entities


def read_text_file(uploaded_file):
    """Reads a Hindi text file and returns a list of lines."""
    try:
        content = uploaded_file.read().decode("utf-8").strip()
        if not content:
            st.error("फ़ाइल खाली है! (The file is empty!)")
            return None
        return content.split("\n")  # Return list of lines
    except Exception as e:
        st.error(f"Error reading text file: {e}")
        return None

import unicodedata
import re
from indicnlp.tokenize import indic_tokenize
from indicnlp.normalize import indic_normalize

GLOBAL_MATCHED_ENTITIES = []
GLOBAL_UNMATCHED_ENTITIES = []
def update_global_lists(matched, unmatched):
    """
    Update global lists for matched and unmatched entities.
    """
    GLOBAL_MATCHED_ENTITIES.extend(matched)
    GLOBAL_UNMATCHED_ENTITIES.extend(unmatched)

def normalize_text(text):
    """Normalize Hindi text for better matching."""
    text = unicodedata.normalize('NFC', text)  # Normalize Unicode representation
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

def preprocess_hindi_entities(entity_list):
    """Preprocess entity list: normalize and tokenize."""
    return [normalize_text(ent) for ent in entity_list]

def calculate_exact_entity_matches_for_two(list1, list2):
    """Calculate exact entity matches with improved Hindi handling."""
    list1 = preprocess_hindi_entities(list1)
    list2 = preprocess_hindi_entities(list2)

    matched_entities = []
    unmatched_entities = list1.copy()

    for entity in list1:
        if entity in list2 and entity not in matched_entities:
            matched_entities.append(entity)
            if entity in unmatched_entities:
                unmatched_entities.remove(entity)

    update_global_lists(matched_entities, unmatched_entities)
    return len(matched_entities)

def calculate_entity_matches_for_three_match(list1, list2, list3):
    """Calculate exact entity matches across three lists with Hindi normalization."""
    list1 = preprocess_hindi_entities(list1)
    list2 = preprocess_hindi_entities(list2)
    list3 = preprocess_hindi_entities(list3)

    matched_all_three = []
    unmatched_entities = list1.copy()

    for entity in list1:
        if entity in list2 and entity in list3 and entity not in matched_all_three:
            matched_all_three.append(entity)
            if entity in unmatched_entities:
                unmatched_entities.remove(entity)

    update_global_lists(matched_all_three, unmatched_entities)
    return len(matched_all_three)

def softmax_ehi(PH, EF, NH, OF, LF):
    """
    Computes EHI using the softmax-based normalization method.
    Ensures values stay between 0 and 1.
    """
    values = np.array([PH, EF, NH, OF, LF])
    exp_values = np.exp(values)  # Apply softmax transformation
    numerator = exp_values[0] + exp_values[1]  # PH + EF
    denominator = np.sum(exp_values)  # Sum of all exponentiated values
    return numerator / denominator


from typing import Tuple

def Compute_EHI(input_text: str, reference_text: str, generated_text: str) -> Tuple[float, float, float, float, float, float, float]:
    """Computes the Entity Hallucination Index (EHI) and related metrics."""
    
    # Extract entities from the three text sources
    input_entities = extract_entities(input_text)
    reference_entities = extract_entities(reference_text)
    generated_entities = extract_entities(generated_text)

    # Convert sets for easier calculations
    I, R, G = input_entities, reference_entities, generated_entities

    # Compute entity matches
    I_R_G = calculate_entity_matches_for_three_match(I, R, G)
    I_R = calculate_exact_entity_matches_for_two(I, R)
    R_G = calculate_exact_entity_matches_for_two(R, G)
    I_G = calculate_exact_entity_matches_for_two(I, G)
     
    # Compute metrics with safe division handling
    total_entities = len(I) + len(R) + len(G)
    EF = 3 * I_R_G / total_entities if total_entities > 0 else 0

    RG_total = len(R) + len(G)
    PH = 2 * R_G / RG_total if RG_total > 0 else 0

    IG_total = len(I) + len(G)
    OF = (2 * I_G - I_R_G) / IG_total if IG_total > 0 else 0

    NH = (len(G) - (R_G + I_G - I_R_G)) / len(G) if len(G) > 0 else 0
    LF = (I_R - I_R_G) / len(G) if len(G) > 0 else 0

    # Compute EHI
    # EHI = 1 + (EF * PH) - (OF + NH + LF)

    # EHI = (EF+PH)/(EF+PH+OF+NH+LF) if (EF+PH+OF+NH+LF) and (EF+PH)> 0 else 0 
    EHI = softmax_ehi(PH, EF, NH, OF, LF)
    
    # st.write("EHI Counts:")
    # st.write(f"ℹ️ **Entity Matches (I_R_G):** {I_R_G}")
    # st.write(f"ℹ️ **Entity Matches (I_R):** {I_R}")
    # st.write(f"ℹ️ **Entity Matches (R_G):** {R_G}")
    # st.write(f"ℹ️ **Entity Matches (I_G):** {I_G}") 
    # st.write("EHI",EHI)
    # Compute Entity F1 Score
    Entity_Precision = len(R.intersection(G)) / len(G) if len(G) > 0 else 0
    
    Entity_Recall = len(R.intersection(G)) / len(R) if len(R) > 0 else 0
    Entity_F1 = (2 * Entity_Precision * Entity_Recall) / (Entity_Precision + Entity_Recall) if (Entity_Precision + Entity_Recall) > 0 else 0
    # st.write("EF 1 Counts:")
    # st.write("R_G",len(R.intersection(G)))
    # st.write("R",len(R))
    # st.write("G",len(G))
    
    # st.write(f"ℹ️ **Entity Precision:** {Entity_Precision}")
    # st.write(f"ℹ️ **Entity Recall:** {Entity_Recall}")
    # st.write(f"ℹ️ **Entity F1 Score:** {Entity_F1}")
    EF1 = Entity_F1
    

    return EHI, EF, PH, OF, NH, LF, EF1
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io

# Function to plot EHI vs Entity F1 Score comparison for each model
def plot_ehi_vs_entity_f1(df):
    models = df["Model"].unique()
    
    st.subheader("📊 Entity F1 Score vs EHI Comparison (Model-wise)")
    
    for model in models:
        model_df = df[df["Model"] == model]
        
        plt.figure(figsize=(10, 6))
        sns.lineplot(x="Text Index", y="Entity F1 Score", data=model_df, label="Entity F1 Score", color="blue")
        sns.lineplot(x="Text Index", y="EHI", data=model_df, label="EHI", color="red")
        plt.title(f"{model} - Entity F1 vs EHI")
        plt.xlabel("Text Index")
        plt.ylabel("Value")
        plt.legend()
        
        st.pyplot(plt)  # Display the plot in Streamlit
        
        # Save plot for download
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        plt.close()
        
        st.download_button(f"डाउनलोड करें {model} का ग्राफ (Download {model} Graph)", buf, f"{model}_entity_f1_vs_ehi.png", "image/png")
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io

# Function to plot EHI vs Entity F1 Score for all models in a single plot
def plot_ehi_vs_entity_f1_all_models(df):
    st.subheader("📊 EHI vs Entity F1 Score Comparison (All Models in One Plot)")

    plt.figure(figsize=(12, 7))

    # Plot lines for each model
    models = df["Model"].unique()
    for model in models:
        model_df = df[df["Model"] == model]
        sns.lineplot(x="Text Index", y="Entity F1 Score", data=model_df, label=f"{model} - Entity F1", linestyle="dashed")
        sns.lineplot(x="Text Index", y="EHI", data=model_df, label=f"{model} - EHI")

    plt.xlabel("Text Index")
    plt.ylabel("Score")
    plt.title("EHI vs Entity F1 Score (All Models)")
    plt.legend()
    
    st.pyplot(plt)  # Display the plot in Streamlit

    # Save plot for download
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plt.close()

    st.download_button("डाउनलोड करें संयुक्त ग्राफ (Download Combined Graph)", buf, "EHI_vs_EntityF1_AllModels.png", "image/png")

# Function to compute and plot mean & variance of EHI and Entity F1 Score model-wise
def plot_mean_variance_modelwise(df):
    st.subheader("📊 Mean and Variance of EHI & Entity F1 Score (Model-wise)")
    
    metrics = ["EHI", "Entity F1 Score"]
    models = df["Model"].unique()
    
    mean_df = df.groupby("Model")[metrics].mean().reset_index()
    var_df = df.groupby("Model")[metrics].var().reset_index()
    
    # Convert to long format for seaborn barplot
    mean_df = mean_df.melt(id_vars=["Model"], var_name="Metric", value_name="Mean")
    var_df = var_df.melt(id_vars=["Model"], var_name="Metric", value_name="Variance")
    
    # Plot Mean Values
    plt.figure(figsize=(10, 6))
    sns.barplot(x="Model", y="Mean", hue="Metric", data=mean_df)
    plt.title("Mean of EHI & Entity F1 Score per Model")
    plt.xlabel("Model")
    plt.ylabel("Mean Value")
    plt.xticks(rotation=45)
    
    st.pyplot(plt)  # Display the plot
    
    # Save plot for download
    mean_buf = io.BytesIO()
    plt.savefig(mean_buf, format="png")
    mean_buf.seek(0)
    plt.close()
    
    st.download_button("डाउनलोड करें Mean ग्राफ (Download Mean Graph)", mean_buf, "mean_plot.png", "image/png")
    
    # Plot Variance Values
    plt.figure(figsize=(10, 6))
    sns.barplot(x="Model", y="Variance", hue="Metric", data=var_df)
    plt.title("Variance of EHI & Entity F1 Score per Model")
    plt.xlabel("Model")
    plt.ylabel("Variance Value")
    plt.xticks(rotation=45)
    
    st.pyplot(plt)  # Display the plot
    
    # Save plot for download
    var_buf = io.BytesIO()
    plt.savefig(var_buf, format="png")
    var_buf.seek(0)
    plt.close()
    
    st.download_button("डाउनलोड करें Variance ग्राफ (Download Variance Graph)", var_buf, "variance_plot.png", "image/png")
    
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import io

def normalize_column(column):
    return (column - column.min()) / (column.max() - column.min())

# Function to compute and plot mean & variance of Normalized EHI and Entity F1 Score model-wise
def plot_mean_variance_modelwise_normalised(df):
    st.subheader("📊 Mean and Variance of Normalized EHI & Entity F1 Score (Model-wise)")

    # Normalize EHI
    df["EHI_Normalized"] = normalize_column(df["EHI"])

    metrics = ["EHI", "EHI_Normalized", "Entity F1 Score"]
    models = df["Model"].unique()

    mean_df = df.groupby("Model")[metrics].mean().reset_index()
    var_df = df.groupby("Model")[metrics].var().reset_index()

    # Convert to long format for seaborn barplot
    mean_df = mean_df.melt(id_vars=["Model"], var_name="Metric", value_name="Mean")
    var_df = var_df.melt(id_vars=["Model"], var_name="Metric", value_name="Variance")

    # Plot Mean Values
    plt.figure(figsize=(10, 6))
    sns.barplot(x="Model", y="Mean", hue="Metric", data=mean_df)
    plt.title("Mean of Normalized EHI & Entity F1 Score per Model")
    plt.xlabel("Model")
    plt.ylabel("Mean Value")
    plt.xticks(rotation=45)
    
    st.pyplot(plt)  # Display the plot
    
    # Save plot for download
    mean_buf = io.BytesIO()
    plt.savefig(mean_buf, format="png")
    mean_buf.seek(0)
    plt.close()
    
    st.download_button("📥 डाउनलोड करें Mean ग्राफ (Download Mean Graph)", mean_buf, "mean_plot.png", "image/png")

    # Plot Variance Values
    plt.figure(figsize=(10, 6))
    sns.barplot(x="Model", y="Variance", hue="Metric", data=var_df)
    plt.title("Variance of Normalized EHI & Entity F1 Score per Model")
    plt.xlabel("Model")
    plt.ylabel("Variance Value")
    plt.xticks(rotation=45)
    
    st.pyplot(plt)  # Display the plot
    
    # Save plot for download
    var_buf = io.BytesIO()
    plt.savefig(var_buf, format="png")
    var_buf.seek(0)
    plt.close()
    
    st.download_button("📥 डाउनलोड करें Variance ग्राफ (Download Variance Graph)", var_buf, "variance_plot.png", "image/png")

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io

# Function to plot EHI and Entity F1 Score for all models separately
def plot_ehi_and_entity_f1_all_models_single(df):
    st.subheader("📊 EHI and Entity F1 Score Comparison (All Models)")

    models = df["Model"].unique()

    # Plot EHI for all models
    plt.figure(figsize=(12, 6))
    for model in models:
        model_df = df[df["Model"] == model]
        sns.lineplot(x="Text Index", y="EHI", data=model_df, label=model)
    plt.xlabel("Text Index")
    plt.ylabel("EHI Score")
    plt.title("EHI Score Comparison (All Models)")
    plt.legend()
    st.pyplot(plt)  # Display the plot in Streamlit

    # Save EHI plot for download
    buf_ehi = io.BytesIO()
    plt.savefig(buf_ehi, format="png")
    buf_ehi.seek(0)
    plt.close()

    st.download_button("डाउनलोड करें EHI ग्राफ (Download EHI Graph)", buf_ehi, "EHI_AllModels.png", "image/png")

    # Plot Entity F1 Score for all models
    plt.figure(figsize=(12, 6))
    for model in models:
        model_df = df[df["Model"] == model]
        sns.lineplot(x="Text Index", y="Entity F1 Score", data=model_df, label=model)
    plt.xlabel("Text Index")
    plt.ylabel("Entity F1 Score")
    plt.title("Entity F1 Score Comparison (All Models)")
    plt.legend()
    st.pyplot(plt)  # Display the plot in Streamlit

    # Save Entity F1 Score plot for download
    buf_entity_f1 = io.BytesIO()
    plt.savefig(buf_entity_f1, format="png")
    buf_entity_f1.seek(0)
    plt.close()

    st.download_button("डाउनलोड करें Entity F1 ग्राफ (Download Entity F1 Graph)", buf_entity_f1, "Entity_F1_AllModels.png", "image/png")
import numpy as np
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import io

# Function to plot CDF of EHI and Entity F1 Score for all models
def plot_cdf_ehi_entity_f1(df):
    st.subheader("📊 CDF Plot for EHI and Entity F1 Score (All Models)")

    models = df["Model"].unique()

    # Plot CDF for EHI across all models
    plt.figure(figsize=(12, 6))
    for model in models:
        model_df = df[df["Model"] == model]
        sorted_values = np.sort(model_df["EHI"])
        cdf = np.arange(1, len(sorted_values) + 1) / len(sorted_values)
        plt.plot(sorted_values, cdf, label=model)
    plt.xlabel("EHI Score")
    plt.ylabel("Cumulative Probability")
    plt.title("Cumulative Distribution Function (CDF) - EHI")
    plt.legend()
    st.pyplot(plt)  # Display the plot in Streamlit

    # Save CDF EHI plot for download
    buf_ehi_cdf = io.BytesIO()
    plt.savefig(buf_ehi_cdf, format="png")
    buf_ehi_cdf.seek(0)
    plt.close()
    
    st.download_button("डाउनलोड करें EHI CDF ग्राफ (Download EHI CDF Graph)", buf_ehi_cdf, "EHI_CDF_AllModels.png", "image/png")

    # Plot CDF for Entity F1 Score across all models
    plt.figure(figsize=(12, 6))
    for model in models:
        model_df = df[df["Model"] == model]
        sorted_values = np.sort(model_df["Entity F1 Score"])
        cdf = np.arange(1, len(sorted_values) + 1) / len(sorted_values)
        plt.plot(sorted_values, cdf, label=model)
    plt.xlabel("Entity F1 Score")
    plt.ylabel("Cumulative Probability")
    plt.title("Cumulative Distribution Function (CDF) - Entity F1 Score")
    plt.legend()
    st.pyplot(plt)  # Display the plot in Streamlit

    # Save CDF Entity F1 Score plot for download
    buf_entity_f1_cdf = io.BytesIO()
    plt.savefig(buf_entity_f1_cdf, format="png")
    buf_entity_f1_cdf.seek(0)
    plt.close()

    st.download_button("डाउनलोड करें Entity F1 CDF ग्राफ (Download Entity F1 CDF Graph)", buf_entity_f1_cdf, "Entity_F1_CDF_AllModels.png", "image/png")
import numpy as np
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import io

# Function to plot combined CDF of EHI and Entity F1 Score for all models
def plot_combined_cdf(df):
    st.subheader("📊 CDF Plot for EHI and Entity F1 Score (All Models in One Graph)")

    models = df["Model"].unique()
    
    plt.figure(figsize=(12, 6))
    
    # Plot CDF for EHI
    for model in models:
        model_df = df[df["Model"] == model]
        sorted_values = np.sort(model_df["EHI"])
        cdf = np.arange(1, len(sorted_values) + 1) / len(sorted_values)
        plt.plot(sorted_values, cdf, linestyle="--", label=f"EHI - {model}")
    
    # Plot CDF for Entity F1 Score
    for model in models:
        model_df = df[df["Model"] == model]
        sorted_values = np.sort(model_df["Entity F1 Score"])
        cdf = np.arange(1, len(sorted_values) + 1) / len(sorted_values)
        plt.plot(sorted_values, cdf, linestyle="-", label=f"Entity F1 - {model}")
    
    plt.xlabel("Score Value")
    plt.ylabel("Cumulative Probability")
    plt.title("Cumulative Distribution Function (CDF) - EHI & Entity F1 Score")
    plt.legend()
    plt.grid()
    
    st.pyplot(plt)  # Display the plot in Streamlit

    # Save plot for download
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plt.close()

    st.download_button("📥 डाउनलोड करें CDF ग्राफ (Download CDF Graph)", buf, "EHI_EntityF1_CDF_AllModels.png", "image/png")
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

def plot_entity_f1_vs_ph(df):
    """
    Plot Entity F1 Score vs Positive Hallucination (PH) for each model separately.
    """
    models = df["Model"].unique()

    for model in models:
        model_df = df[df["Model"] == model]

        plt.figure(figsize=(8, 5))

        # Plot PH vs Index in BLUE
        sns.lineplot(x=model_df.index, y=model_df["Positive Hallucination (PH)"], marker="o", linestyle="-", label="Positive Hallucination (PH)", color="blue")

        # Plot Entity F1 Score vs Index in RED
        sns.lineplot(x=model_df.index, y=model_df["Entity F1 Score"], marker="s", linestyle="--", label="Entity F1 Score", color="red")

        plt.xlabel("Text Index")
        plt.ylabel("Score")
        plt.title(f"Entity F1 vs PH for {model}")
        plt.legend()
        plt.grid(True)

        st.pyplot(plt)  # Display plot in Streamlit
import pandas as pd
import streamlit as st
from io import BytesIO
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import io

def generate_avg_metrics_table(result_df):
    st.header("औसत मेट्रिक्स सारणी (Average Metrics Table)")

    # Selecting only numerical columns for averaging
    numeric_cols = ['EHI', 'Extractiveness Factor (EF)', 'Positive Hallucination (PH)', 
                    'Over Focus (OF)', 'Negative Hallucination (NH)', 'Lost Focus (LF)', 
                    'Entity F1 Score']
    
    avg_metrics_df = result_df.groupby("Model")[numeric_cols].mean()  # Compute average

    # Display in Streamlit
    st.dataframe(avg_metrics_df.style.format("{:.4f}"))

    # Save as image
    img_bytes = save_table_as_image(avg_metrics_df)

    # Provide download buttons
    csv = avg_metrics_df.to_csv(index=True).encode("utf-8")
    st.download_button("डाउनलोड करें (Download Average Metrics CSV)", csv, "average_metrics.csv", "text/csv")

    st.download_button("डाउनलोड करें (Download as Image)", img_bytes, "average_metrics.png", "image/png")

def save_table_as_image(df):
    """Convert DataFrame to an image and return as bytes."""
    fig, ax = plt.subplots(figsize=(10, 4))  # Adjust size
    ax.axis("tight")
    ax.axis("off")
    sns.heatmap(df, annot=True, fmt=".4f", cmap="Blues", linewidths=0.5, ax=ax)

    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format="png", bbox_inches="tight", dpi=300)  # High-quality image
    plt.close()
    img_buffer.seek(0)
    return img_buffer
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import io

def normalize_column(column):
    return (column - column.min()) / (column.max() - column.min())

def get_image_download_link(fig, filename):
    img_buf = io.BytesIO()
    fig.savefig(img_buf, format="png", dpi=300, bbox_inches='tight')
    img_buf.seek(0)
    st.download_button(label=f"Download {filename}", data=img_buf, file_name=filename, mime="image/png")

# Function to plot Normalized EHI and Raw Entity F1 Score
def plot_normalized_ehi_and_entity_f1(df):
    st.subheader("📊 EHI & Entity F1 Score Analysis (All Models)")

    # Normalize EHI only
    df["EHI_normalized"] = normalize_column(df["EHI"])

    models = df["Model"].unique()

    # Color Palette
    colors = sns.color_palette("tab10", len(models))
    model_colors = {model: color for model, color in zip(models, colors)}

    # ---- Plot 1: Normalized EHI ----
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    sns.lineplot(x="Text Index", y="EHI_normalized", hue="Model", style="Model", 
                 alpha=0.8, linewidth=2, palette=model_colors, marker="o", data=df, ax=ax1)

    ax1.set_xlabel("Text Index", fontsize=14, fontweight='bold')
    ax1.set_ylabel("Normalized EHI", fontsize=14, fontweight='bold')
    ax1.set_title("Normalized EHI (Model-wise)", fontsize=16, fontweight='bold')
    ax1.legend(title="Model", loc="upper right", fontsize=12)

    get_image_download_link(fig1, "Normalized_EHI_Plot.png")
    st.pyplot(fig1)

    # ---- Plot 2: Raw Entity F1 Score ----
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    sns.lineplot(x="Text Index", y="Entity F1 Score", hue="Model", style="Model", 
                 linestyle="dashed", alpha=0.7, linewidth=2, palette=model_colors, marker="s", data=df, ax=ax2)

    ax2.set_xlabel("Text Index", fontsize=14, fontweight='bold')
    ax2.set_ylabel("Entity F1 Score", fontsize=14, fontweight='bold')
    ax2.set_title("Entity F1 Score (Model-wise)", fontsize=16, fontweight='bold')
    ax2.legend(title="Model", loc="upper right", fontsize=12)

    get_image_download_link(fig2, "Entity_F1_Score_Plot.png")
    st.pyplot(fig2)

    # ---- Plot 3: Normalized EHI vs. Raw Entity F1 Score ----
    fig3, ax3 = plt.subplots(figsize=(12, 6))
    sns.scatterplot(x="EHI_normalized", y="Entity F1 Score", hue="Model", style="Model", 
                    palette=model_colors, alpha=0.7, data=df, ax=ax3)

    ax3.set_xlabel("Normalized EHI", fontsize=14, fontweight='bold')
    ax3.set_ylabel("Entity F1 Score", fontsize=14, fontweight='bold')
    ax3.set_title("Comparison: Normalized EHI vs. Entity F1 Score", fontsize=16, fontweight='bold')
    ax3.legend(title="Model", loc="upper right", fontsize=12)

    get_image_download_link(fig3, "Normalized_EHI_vs_Entity_F1_Score.png")
    st.pyplot(fig3)
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import io

def normalize_column(column):
    return (column - column.min()) / (column.max() - column.min())

def generate_avg_metrics_table_with_normalized_EHI(result_df):
    st.header("औसत मेट्रिक्स सारणी (Average Metrics Table with Normalized EHI)")

    # Selecting numerical columns
    numeric_cols = ['EHI', 'Extractiveness Factor (EF)', 'Positive Hallucination (PH)', 
                    'Over Focus (OF)', 'Negative Hallucination (NH)', 'Lost Focus (LF)', 
                    'Entity F1 Score']

    # Normalize only EHI
    result_df["EHI_Normalized"] = normalize_column(result_df["EHI"])

    # Compute the average of all metrics, keeping normalized EHI
    avg_metrics_df = result_df.groupby("Model")[numeric_cols].mean()
    avg_metrics_df["EHI_Normalized"] = result_df.groupby("Model")["EHI_Normalized"].mean()

    # Display in Streamlit
    st.dataframe(avg_metrics_df.style.format("{:.4f}"))

    # Convert to CSV for download
    csv = avg_metrics_df.to_csv(index=True).encode("utf-8")
    st.download_button("डाउनलोड करें (Download Average Metrics CSV)", csv, "average_metrics_with_normalized_EHI.csv", "text/csv")

    # Convert table to image for download
    img_bytes = save_table_as_image(avg_metrics_df)
    st.download_button("डाउनलोड करें (Download as Image)", img_bytes, "average_metrics_with_normalized_EHI.png", "image/png")
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import io

def normalize_column(column):
    return (column - column.min()) / (column.max() - column.min())

# Function to plot combined CDF of Normalized EHI and Entity F1 Score for all models
def plot_combined_cdf_normalized_ehi(df):
    st.subheader("📊 CDF Plot for Normalized EHI and Entity F1 Score (All Models in One Graph)")

    models = df["Model"].unique()

    # Normalize EHI
    df["EHI_Normalized"] = normalize_column(df["EHI"])

    plt.figure(figsize=(12, 6))

    # Plot CDF for Normalized EHI
    for model in models:
        model_df = df[df["Model"] == model]
        sorted_values = np.sort(model_df["EHI_Normalized"])
        cdf = np.arange(1, len(sorted_values) + 1) / len(sorted_values)
        plt.plot(sorted_values, cdf, linestyle="--", label=f"Normalized EHI - {model}")

    # Plot CDF for Entity F1 Score
    for model in models:
        model_df = df[df["Model"] == model]
        sorted_values = np.sort(model_df["Entity F1 Score"])
        cdf = np.arange(1, len(sorted_values) + 1) / len(sorted_values)
        plt.plot(sorted_values, cdf, linestyle="-", label=f"Entity F1 - {model}")

    plt.xlabel("Score Value")
    plt.ylabel("Cumulative Probability")
    plt.title("Cumulative Distribution Function (CDF) - Normalized EHI & Entity F1 Score")
    plt.legend()
    plt.grid()

    st.pyplot(plt)  # Display the plot in Streamlit

    # Save plot for download
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plt.close()

    st.download_button("📥 डाउनलोड करें CDF ग्राफ (Download CDF Graph)", buf, "Normalized_EHI_EntityF1_CDF_AllModels.png", "image/png")


def main():
    st.title("हिंदी पाठ विश्लेषक (Hindi Text Analysis)")

    # **File Upload Section**
    st.header("फ़ाइलें अपलोड करें (Upload Files)")
    uploaded_files = st.file_uploader("7 टेक्स्ट फ़ाइलें अपलोड करें (.txt)", type=["txt"], accept_multiple_files=True)
    
    if uploaded_files and len(uploaded_files) == 7:
        file_mapping = {
            "facebookbart-large-cnn": None,
            "googlepegasus-xsum": None,
            "gpt-3.5-turbo": None,
            "t5-large": None,
            "RefSum": None,
            "ReferenceSummary": None,
            "input_eng_out_hindi": None
        }
        
        for file in uploaded_files:
            if "facebookbart-large-cnn" in file.name:
                file_mapping["facebookbart-large-cnn"] = read_text_file(file)
            elif "googlepegasus-xsum" in file.name:
                file_mapping["googlepegasus-xsum"] = read_text_file(file)
            elif "gpt-3.5-turbo" in file.name:
                file_mapping["gpt-3.5-turbo"] = read_text_file(file)
            elif "t5-large" in file.name:
                file_mapping["t5-large"] = read_text_file(file)
            elif "RefSum" in file.name:
                file_mapping["RefSum"] = read_text_file(file)
            elif "ReferenceSummary" in file.name:
                file_mapping["ReferenceSummary"] = read_text_file(file)
            elif "input_eng_out_hindi" in file.name:
                file_mapping["input_eng_out_hindi"] = read_text_file(file)

        if None in file_mapping.values():
            st.error("कृपया सुनिश्चित करें कि सभी आवश्यक फ़ाइलें अपलोड की गई हैं।")
            return
        
        # **Batch Processing**
        results = []
        for idx in range(len(file_mapping["input_eng_out_hindi"])):
            input_text = file_mapping["input_eng_out_hindi"][idx]
            ref_summary = file_mapping["ReferenceSummary"][idx] if idx < len(file_mapping["ReferenceSummary"]) else ""
            
            for model in ["facebookbart-large-cnn", "googlepegasus-xsum", "gpt-3.5-turbo", "t5-large", "RefSum"]:
                if idx < len(file_mapping[model]):
                    generated_summary = file_mapping[model][idx]
                    EHI, EF, PH, OF, NH, LF, EF1 = Compute_EHI(input_text, ref_summary, generated_summary)
                    results.append({
                        'Model': model,
                        'Text Index': idx,
                        'Input Text': input_text,
                        'Reference Summary': ref_summary,
                        'Generated Summary': generated_summary,
                        'EHI': EHI,
                        'Extractiveness Factor (EF)': EF,
                        'Positive Hallucination (PH)': PH,
                        'Over Focus (OF)': OF,
                        'Negative Hallucination (NH)': NH,
                        'Lost Focus (LF)': LF,
                        'Entity F1 Score': EF1
                    })
        
        if results:
            result_df = pd.DataFrame(results)
            st.write(result_df)
            
            st.download_button("डाउनलोड करें (Download CSV)", result_df.to_csv(index=False).encode("utf-8"), "EHI_results.csv", "text/csv")
            st.download_button("डाउनलोड करें (Download JSON)", result_df.to_json(orient="records", indent=4).encode("utf-8"), "EHI_results.json", "application/json")
            
            plot_ehi_and_entity_f1_all_models_single(result_df)
            st.write("Avg Metrics Table with Normalized EHI")
            generate_avg_metrics_table_with_normalized_EHI(result_df)
            st.write("Normalized EHI mean and Variance")
            plot_mean_variance_modelwise_normalised(result_df)
            st.write("Combined CDF of Normalised EHI and EF_1:")
            plot_combined_cdf_normalized_ehi(result_df)
            plot_ehi_vs_entity_f1(result_df)
            plot_combined_cdf(result_df)
            plot_cdf_ehi_entity_f1(result_df)
            plot_entity_f1_vs_ph(result_df)
            plot_ehi_vs_entity_f1_all_models(result_df)
            plot_mean_variance_modelwise(result_df)
            generate_avg_metrics_table(result_df)
            st.write("Normalized EHI and Entity F1 Score Analysis")
            plot_normalized_ehi_and_entity_f1(result_df)
if __name__ == "__main__":
    main()