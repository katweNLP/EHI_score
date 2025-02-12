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

    st.write(f"✅ **Final Extracted Entities (Merged):** {combined_entities}")

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

def extract_entities_from_ner_file(file_data):
    """Extract entities from NER-tagged text grouped by sentence ID."""
    entities_by_sentence = {}
    current_sentence_id = None
    current_entities = set()
    
    for line in file_data.split('\n'):
        line = line.strip()
        
        if line.startswith("<sentence_id"):
            if current_sentence_id is not None:
                entities_by_sentence[current_sentence_id] = current_entities
                current_entities = set()
            current_sentence_id = int(line.split('"')[1])
        elif line and "B-LOC" in line or "B-PER" in line or "B-ORG" in line:
            entity = line.split('\t')[0]
            current_entities.add(entity)
    
    if current_sentence_id is not None:
        entities_by_sentence[current_sentence_id] = current_entities
    
    return entities_by_sentence
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure correct column names
def preprocess_dataframe(df):
    column_mapping = {
        "EHI": "Entity Hallucination Index",
        "Entity_F1": "Entity F1 Score",
        "PH": "Positive Hallucination"
    }
    df = df.rename(columns=column_mapping)
    
    required_columns = ["Entity Hallucination Index", "Entity F1 Score", "Model"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        print(f"⚠️ Warning: Missing columns in DataFrame → {missing_columns}")
        return None  # Prevents function from running if essential columns are missing
    
    return df.dropna()  # Remove NaN values


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

def Compute_EHI(input_entities, reference_entities, generated_entities) -> Tuple[float, float, float, float, float, float, float]:
    """Computes the Entity Hallucination Index (EHI) based on entity sets."""
    I, R, G = input_entities, reference_entities, generated_entities

    I_R_G = len(I & R & G)
    I_R = len(I & R)
    R_G = len(R & G)
    I_G = len(I & G)
    
    total_entities = len(I) + len(R) + len(G)
    EF = 3 * I_R_G / total_entities if total_entities > 0 else 0
    
    RG_total = len(R) + len(G)
    PH = 2 * R_G / RG_total if RG_total > 0 else 0
    
    IG_total = len(I) + len(G)
    OF = (2 * I_G - I_R_G) / IG_total if IG_total > 0 else 0
    
    NH = (len(G) - (R_G + I_G - I_R_G)) / len(G) if len(G) > 0 else 0
    LF = (I_R - I_R_G) / len(G) if len(G) > 0 else 0
    
    # EHI = 1 + (EF * PH) - (OF + NH + LF)
    # EHI = (EF+PH)/(EF+PH+OF+NH+LF) if (EF+PH+OF+NH+LF) and (EF+PH)> 0 else 0 
    EHI = softmax_ehi(PH, EF, NH, OF, LF)
    
    Entity_Precision = len(R & G) / len(G) if len(G) > 0 else 0
    Entity_Recall = len(R & G) / len(R) if len(R) > 0 else 0
    Entity_F1 = (2 * Entity_Precision * Entity_Recall) / (Entity_Precision + Entity_Recall) if (Entity_Precision + Entity_Recall) > 0 else 0
    
    return EHI, EF, PH, OF, NH, LF, Entity_F1

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def download_plot(fig, filename):
    """Utility function to allow downloading of plots."""
    import io
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    st.download_button(
        label="Download Plot",
        data=buf.getvalue(),
        file_name=filename,
        mime="image/png"
    )
# 1. **Plot EHI vs Entity F1 (Model-wise & Combined) using Line Plots**
def plot_ehi_vs_entity_f1(result_df, sentence_id=None):
    filtered_df = result_df if sentence_id is None else result_df[result_df["Sentence ID"] == sentence_id]
    
    # Sort by Sentence ID for line plot consistency
    filtered_df = filtered_df.sort_values(by=["Sentence ID"])

    # **Combined Model Plot**
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(data=filtered_df, x="Sentence ID", y="EHI", hue="Model", marker="o", linestyle="-", ax=ax)
    sns.lineplot(data=filtered_df, x="Sentence ID", y="Entity F1 Score", hue="Model", marker="s", linestyle="--", ax=ax)
    
    plt.title("EHI vs Entity F1 Score (All Models)")
    plt.xlabel("Sentence ID")
    plt.ylabel("Score")
    plt.grid(True)
    plt.legend(title="Model", bbox_to_anchor=(1, 1))
    st.pyplot(fig)
    download_plot(fig, "EHI_vs_EntityF1_AllModels_LinePlot.png")

    # **Model-wise Plots**
    models = result_df["Model"].unique()
    for model in models:
        fig, ax = plt.subplots(figsize=(12, 6))
        model_df = filtered_df[filtered_df["Model"] == model]
        
        sns.lineplot(data=model_df, x="Sentence ID", y="EHI", marker="o", linestyle="-", label="EHI", ax=ax)
        sns.lineplot(data=model_df, x="Sentence ID", y="Entity F1 Score", marker="s", linestyle="--", label="Entity F1 Score", ax=ax)

        plt.title(f"EHI vs Entity F1 Score - {model}")
        plt.xlabel("Sentence ID")
        plt.ylabel("Score")
        plt.grid(True)
        plt.legend()
        st.pyplot(fig)
        download_plot(fig, f"EHI_vs_EntityF1_{model}_LinePlot.png")

# 2. **Plot Mean & Variance Bar Graph & Table**
def plot_mean_variance(result_df):
    summary_df = result_df.groupby("Model")[["EHI", "Entity F1 Score"]].agg(["mean", "var"])
    
    # Convert MultiIndex columns to single-level
    summary_df.columns = ["_".join(col) for col in summary_df.columns]
    summary_df.reset_index(inplace=True)

    st.write("### Mean & Variance Table")
    st.write(summary_df)

    # Bar plot
    fig, ax = plt.subplots(figsize=(10, 6))
    summary_df.set_index("Model").plot(kind="bar", ax=ax)
    plt.title("Mean & Variance of EHI & Entity F1 Score")
    plt.xlabel("Model")
    plt.ylabel("Score")
    plt.grid(True)
    plt.xticks(rotation=45)
    st.pyplot(fig)
    download_plot(fig, "Mean_Variance_EHI_EntityF1.png")

# 3. **CDF of EHI & Entity F1 Score**
def plot_cdf_ehi_entity_f1(result_df):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.ecdfplot(data=result_df, x="EHI", label="EHI", linewidth=2, ax=ax)
    sns.ecdfplot(data=result_df, x="Entity F1 Score", label="Entity F1", linewidth=2, ax=ax)
    plt.title("CDF of EHI & Entity F1 Score")
    plt.xlabel("Score")
    plt.ylabel("Cumulative Probability")
    plt.legend()
    plt.grid(True)
    st.pyplot(fig)
    download_plot(fig, "CDF_EHI_EntityF1.png")

# 4. **Plot of All Average Metrics**
def plot_all_avg_metrics(result_df):
    avg_metrics = result_df.groupby("Model").mean().reset_index()

    fig, ax = plt.subplots(figsize=(12, 6))
    avg_metrics.set_index("Model").plot(kind="bar", ax=ax)
    plt.title("Average Metrics Across Models")
    plt.xlabel("Model")
    plt.ylabel("Average Score")
    plt.xticks(rotation=45)
    plt.grid(True)
    st.pyplot(fig)
    download_plot(fig, "All_Avg_Metrics.png")

# Streamlit App
def main():
    st.title("हिंदी पाठ विश्लेषक (Hindi Text Analysis)")
    st.header("फ़ाइलें अपलोड करें (Upload Files)")

    uploaded_files = st.file_uploader("7 टेक्स्ट फ़ाइलें अपलोड करें (.txt)", type=["txt"], accept_multiple_files=True)

    if uploaded_files and len(uploaded_files) == 7:
        file_mapping = {}

        for file in uploaded_files:
            content = file.read().decode("utf-8")
            if "facebookbart-large-cnn" in file.name:
                file_mapping["facebookbart-large-cnn"] = extract_entities_from_ner_file(content)
            elif "googlepegasus-xsum" in file.name:
                file_mapping["googlepegasus-xsum"] = extract_entities_from_ner_file(content)
            elif "gpt-3.5-turbo" in file.name:
                file_mapping["gpt-3.5-turbo"] = extract_entities_from_ner_file(content)
            elif "t5-large" in file.name:
                file_mapping["t5-large"] = extract_entities_from_ner_file(content)
            elif "RefSum" in file.name:
                file_mapping["RefSum"] = extract_entities_from_ner_file(content)
            elif "ReferenceSummary" in file.name:
                file_mapping["ReferenceSummary"] = extract_entities_from_ner_file(content)
            elif "input_eng_out_hindi" in file.name:
                file_mapping["input_eng_out_hindi"] = extract_entities_from_ner_file(content)

        if len(file_mapping) != 7:
            st.error("कृपया सुनिश्चित करें कि सभी आवश्यक फ़ाइलें अपलोड की गई हैं।")
            return

        results = []
        sentence_ids = set(file_mapping["input_eng_out_hindi"].keys())

        for idx in sentence_ids:
            input_entities = file_mapping["input_eng_out_hindi"].get(idx, set())
            reference_entities = file_mapping["ReferenceSummary"].get(idx, set())

            for model in ["facebookbart-large-cnn", "googlepegasus-xsum", "gpt-3.5-turbo", "t5-large", "RefSum"]:
                generated_entities = file_mapping[model].get(idx, set())
                EHI, EF, PH, OF, NH, LF, EF1 = Compute_EHI(input_entities, reference_entities, generated_entities)

                results.append({
                    'Model': model,
                    'Sentence ID': idx,
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

            # Provide download options
            st.download_button("डाउनलोड करें (Download CSV)", result_df.to_csv(index=False).encode("utf-8"), "EHI_results.csv", "text/csv")
            st.download_button("डाउनलोड करें (Download JSON)", result_df.to_json(orient="records", indent=4).encode("utf-8"), "EHI_results.json", "application/json")

            # Plot functions
            st.header("Plots")
            plot_ehi_vs_entity_f1(result_df)
            plot_mean_variance(result_df)
            plot_cdf_ehi_entity_f1(result_df)
            plot_all_avg_metrics(result_df)

if __name__ == "__main__":
    main()
