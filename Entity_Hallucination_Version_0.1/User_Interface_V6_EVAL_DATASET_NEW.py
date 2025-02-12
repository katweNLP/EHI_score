import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from typing import List, Tuple, Union
import spacy
from typing import Tuple

# Load a pre-trained NER model
nlp = spacy.load("en_core_web_sm")  
import spacy
import re
from spacy.matcher import Matcher

# Load the SpaCy model
nlp = spacy.load("en_core_web_sm")

def extract_entities(text: str) -> set:
    """
    Extract named entities from a given text using SpaCy and custom rules.
    
    Args:
        text (str): Input text to process.
    
    Returns:
        set: A set containing all unique entities.
    """
    # Process the text
    doc = nlp(text.lower())
    
    # Tokenize and POS tagging
    tokens = [(token.text, token.pos_) for token in doc]
    # print("Tokenized and POS-tagged tokens:", tokens)
    
    # Extract named entities recognized by SpaCy
    entities = {ent.text for ent in doc.ents}
    
    # Custom regex patterns for additional entities
    custom_patterns = {
        "PERCENT": re.compile(r"\b\d+(\.\d+)?%\b"),  # Match percentages like "1.6%"
        "MONEY": re.compile(r"\b€?\$?\d+(?:,\d{3})*(?:\.\d+)?(?: billion| million)?\b"),  # Match money values
        "ORDINAL": re.compile(r"\b(first|second|third|[0-9]+(?:st|nd|rd|th))\b", re.IGNORECASE),  # Ordinals
    }
    
    # Extract entities matching custom patterns
    for label, pattern in custom_patterns.items():
        matches = pattern.findall(text)
        entities.update(matches)
    
    # Use SpaCy's Matcher for additional patterns
    matcher = Matcher(nlp.vocab)
    patterns = [
        [{"LOWER": "gross"}, {"LOWER": "domestic"}, {"LOWER": "product"}],  # Match "gross domestic product"
        [{"LOWER": "public"}, {"LOWER": "sector"}],  # Match "public sector"
    ]
    matcher.add("CUSTOM_ENTITIES", patterns)
    matches = matcher(doc)
    for match_id, start, end in matches:
        entity_text = doc[start:end].text
        entities.add(entity_text)
    
    # Print extracted entities
    # print("Extracted Entities:", entities)
    
    return entities



GLOBAL_MATCHED_ENTITIES = []
GLOBAL_UNMATCHED_ENTITIES = []

def update_global_lists(matched, unmatched):
    """
    Update global lists for matched and unmatched entities.
    """
    GLOBAL_MATCHED_ENTITIES.extend(matched)
    GLOBAL_UNMATCHED_ENTITIES.extend(unmatched)

def calculate_exact_entity_matches_for_two(list1, list2):
    """
    Calculate the count of exact matches between two lists of entities.
    This function avoids duplicates in matches and updates global lists for matched and unmatched entities.

    Args:
        list1 (list): The first list of entities.
        list2 (list): The second list of entities.

    Returns:
        int: The count of exact matches between the two lists.
    """
    matched_entities = []
    unmatched_entities = list1.copy()  # Start with all entities from list1 as unmatched

    for entity in list1:
        if entity in list2 and entity not in matched_entities:
            matched_entities.append(entity)
            if entity in unmatched_entities:
                unmatched_entities.remove(entity)

    # Update global matched and unmatched entity lists
    update_global_lists(matched_entities, unmatched_entities)

    # Return the count of matched entities
    return len(matched_entities)

def calculate_entity_matches_for_three_match(list1, list2, list3):
    """
    Calculate the count of exact matches across three lists of entities.

    Args:
        list1 (list): The first list of entities.
        list2 (list): The second list of entities.
        list3 (list): The third list of entities.

    Returns:
        int: The count of exact matches across all three lists.
    """
    # Start with empty matched and unmatched lists
    matched_all_three = []  # Matches in all three lists
    unmatched_entities = list1.copy()  # Initialize with all entities from list1

    # Exact matches in all three lists
    for entity in list1:
        if entity in list2 and entity in list3 and entity not in matched_all_three:
            matched_all_three.append(entity)
            if entity in unmatched_entities:
                unmatched_entities.remove(entity)

    # Update global matched and unmatched lists
    update_global_lists(matched_all_three, unmatched_entities)

    # Return the count of matched entities across all three lists
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
# Function to compute EHI for a single instance
def Compute_EHI(input_text: str, reference_text: str, generated_text: str) -> Tuple[float, float, float, float, float, float,float]:
    # input_entities = set(input_text.split())
    # reference_entities = set(reference_text.split())
    # generated_entities = set(generated_text.split())
    
    input_entities = extract_entities(input_text)
    reference_entities = extract_entities(reference_text)
    generated_entities = extract_entities(generated_text)
    
    
    # st.write("Input Entities: ",input_entities)
    # st.write("Reference Entities: ",reference_entities)
    # st.write("Generated Entities: ",generated_entities)
    
    I = input_entities
    R = reference_entities
    G = generated_entities
    
    
    I_R_G = calculate_entity_matches_for_three_match(input_entities, reference_entities,generated_entities)
     
    I_R = calculate_exact_entity_matches_for_two(input_entities, reference_entities)
    R_G = calculate_exact_entity_matches_for_two(reference_entities, generated_entities)
    I_G = calculate_exact_entity_matches_for_two(input_entities, generated_entities) 
    
    # I_R_G = len(I.intersection(R).intersection(G))
    # I_R = len(I.intersection(R))
    # R_G = len(R.intersection(G))
    # I_G = len(I.intersection(G))
    # st.write("I_R_G values: ",I.intersection(R).intersection(G))
    # st.write("I_R values: ",I.intersection(R))
    # st.write("R_G values: ",R.intersection(G))
    # st.write("I_G values: ",I.intersection(G))
    
    # st.write("I_R_G: ",I_R_G)
    # st.write("I_R: ",I_R)
    # st.write("R_G: ",R_G)
    # st.write("I_G: ",I_G)
    EF = 3 * I_R_G / (len(I) + len(R) + len(G)) if len(I) + len(R) + len(G) > 0 else 0
    PH = 2 * R_G / (len(R) + len(G)) if len(R) + len(G) > 0 else 0
    OF = (2 * (I_G)-(I_R_G))/ (len(I) + len(G)) if len(I) + len(G) > 0 else 0
    NH = (len(G) - (R_G + I_G - I_R_G)) / len(G) if len(G) > 0 else 0
    LF = (I_R - (I_R_G)) / len(G) if  len(G) > 0 else 0
    
    # EHI = 1 + (EF * PH) - (OF + NH + LF)
    EHI = softmax_ehi(PH, EF, NH, OF, LF)
    
    

     # Entity F1 Score Calculation
    # Precision: Fraction of relevant entities in the generated entities
    Entity_Precision = len(R.intersection(G)) / len(G) if len(G) > 0 else 0
    # Recall: Fraction of relevant entities that were successfully generated
    Entity_Recall = len(R.intersection(G)) / len(R) if len(R) > 0 else 0
    # F1 Score: Harmonic mean of Precision and Recall
    Entity_F1 = (2 * Entity_Precision * Entity_Recall) / (Entity_Precision + Entity_Recall) if (Entity_Precision + Entity_Recall) > 0 else 0

    EF1=Entity_F1

    return EHI, EF, PH, OF, NH, LF,EF1

import pandas as pd
import numpy as np
import streamlit as st

import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from io import BytesIO
def calculate_entity_counts(I, R, G):
    """
    Calculate entity counts based on intersections of sets I, R, and G.

    Args:
        I (set): Entities from the Input Text.
        R (set): Entities from the Reference Summary.
        G (set): Entities from the Generated Summary.

    Returns:
        dict: A dictionary with the counts of intersections.
    """
    # Calculate entity counts
    I_R_G = len(I.intersection(R).intersection(G))  # Intersection of I, R, and G
    I_R = len(I.intersection(R))                    # Intersection of I and R
    R_G = len(R.intersection(G))                    # Intersection of R and G
    I_G = len(I.intersection(G))                    # Intersection of I and G

    # Return results as a dictionary
    return {
        "I_R_G": I_R_G,
        "I_R": I_R,
        "R_G": R_G,
        "I_G": I_G
    }

def compute_and_save_metrics(data: pd.DataFrame):
    # Extract models from the column names
    models = ['google/pegasus-xsum', 't5-large', 'gpt-3.5-turbo', 'facebook/bart-large-cnn']
    
    # Initialize a list to store results for each row and model
    results = []
    
    # Loop through each row in the DataFrame
    for index, row in data.iterrows():
        input_text = row['InputText']
        reference_text = row['ReferenceSummary']
        
        for model in models:
            generated_text = row[model]
            
            # Compute metrics for the current row and model
            EHI, EF, PH, OF, NH, LF, EF1 = Compute_EHI(input_text, reference_text, generated_text)
            
            # Append the results to the list
            results.append({
                'Index': index,
                'Model': model,
                'EHI': EHI,
                'Extractiveness Factor (EF)': EF,
                'Positive Hallucination (PH)': PH,
                'Over Focus (OF)': OF,
                'Negative Hallucination (NH)': NH,
                'Lost Focus (LF)': LF,
                'Entity_F1 Score': EF1
            })
    
    # Convert the results list into a DataFrame
    results_df = pd.DataFrame(results)
    
    # Calculate model-wise averages for each metric
    averages_df = results_df.groupby('Model').mean().reset_index()
    
    # Save and display model-wise averages as a table
    st.write("Model-wise Average Metrics:")
    st.dataframe(averages_df)
    
    # Plot the averages table as an image
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(
        cellText=averages_df.values,
        colLabels=averages_df.columns,
        cellLoc='center',
        loc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.auto_set_column_width(col=list(range(len(averages_df.columns))))
    
    # Save the table as an image to a BytesIO object
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=300)
    buf.seek(0)
    
    # Display the image in Streamlit
    st.image(buf, caption="Model-wise Averages", use_column_width=True)
    
    # Provide a download button for the image
    st.download_button(
        label="Download Model-wise Averages as Image",
        data=buf,
        file_name="model_wise_averages.png",
        mime="image/png"
    )
    
    # Provide a download button for the averages in CSV format
    st.download_button(
        label="Download Model-wise Averages (CSV)",
        data=averages_df.to_csv(index=False),
        file_name="model_wise_averages.csv",
        mime="text/csv"
    )

# Function to compute EHI for batch data
def Compute_EHI_Batch(input_texts: List[str], reference_texts: List[str], generated_texts: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    EHIs, EFs, PHs, OFs, NHs, LFs, Entity_F1s = [], [], [], [], [], [], []
    
    for input_text, reference_text, generated_text in zip(input_texts, reference_texts, generated_texts):
        ehi, ef, ph, of, nh, lf, ef1 = Compute_EHI(input_text, reference_text, generated_text)
        EHIs.append(ehi)
        EFs.append(ef)
        PHs.append(ph)
        OFs.append(of)
        NHs.append(nh)
        LFs.append(lf)
        Entity_F1s.append(ef1)
    
    return np.array(EHIs), np.array(EFs), np.array(PHs), np.array(OFs), np.array(NHs), np.array(LFs), np.array(Entity_F1s)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from io import BytesIO

# Function to plot CDFs and allow downloading as image
def plot_cdfs(data: pd.DataFrame):
    # Create the CDF plot
    plt.figure(figsize=(10, 8))
    st.write("Columns: ", data.columns)
    factors = ['EHI', 'Extractiveness Factor (EF)', 'Positive Hallucination (PH)',
               'Over Focus (OF)', 'Negative Hallucination (NH)', 'Lost Focus (LF)', 'Entity_F1 Score']
    
    for factor in factors:
        if factor in data.columns:  # Ensure factor exists in the data
            sorted_data = np.sort(data[factor])
            cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
            plt.plot(sorted_data, cdf, label=factor)
    
    plt.title('CDFs of EHI Factors')
    plt.xlabel('Factor Values')
    plt.ylabel('CDF')
    plt.legend()
    plt.grid(True)
    
    # Display the plot in the Streamlit app
    st.pyplot(plt)
    
    # Save the plot to a BytesIO buffer
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=300)
    buf.seek(0)  # Move to the beginning of the buffer
    
    # Provide a download button for the plot image
    st.download_button(
        label="Download CDF Plot as Image",
        data=buf,
        file_name="cdf_plot.png",
        mime="image/png"
    )
    plt.close()  

# Close the plot to prevent overlap in Streamlit
# Function to create a table with average scores for all metrics
def display_avg_scores(data: pd.DataFrame):
    st.write("Columns: ", data.columns)
    
    # List of metrics for which averages need to be calculated
    metrics = ['EHI', 'Extractiveness Factor (EF)', 'Positive Hallucination (PH)', 
               'Over Focus (OF)', 'Negative Hallucination (NH)', 'Lost Focus (LF)', 
               'Entity_F1 Score']
    
    # Calculate the mean for each metric
    avg_scores = {metric: data[metric].mean() for metric in metrics}
    
    # Create a DataFrame to display in table format
    avg_scores_df = pd.DataFrame.from_dict(avg_scores, orient='index', columns=['Average Score'])
    avg_scores_df.reset_index(inplace=True)
    avg_scores_df.rename(columns={'index': 'Metric'}, inplace=True)
    
    # Display the table in Streamlit
    st.write("Average Scores for Each Metric:")
    st.dataframe(avg_scores_df)
    
    # Provide an option to download the table as a CSV file
    csv_file_path = "avg_scores_table.csv"
    avg_scores_df.to_csv(csv_file_path, index=False)
    
    st.download_button(
        label="Download Average Scores Table (CSV)",
        data=avg_scores_df.to_csv(index=False),
        file_name=csv_file_path,
        mime="text/csv"
    )

def plot_cdfs1(data: pd.DataFrame):
    plt.figure(figsize=(10, 8))
    st.write("Columns: ",data.columns)
    factors = ['EHI', 'Entity_F1 Score']
    for factor in factors:
        sorted_data = np.sort(data[factor])
        cdf = np.arange(1, len(sorted_data)+1) / len(sorted_data)
        plt.plot(sorted_data, cdf, label=factor)
    
    plt.title('CDF Plot for Entity_F1 and EHI Metric')
    plt.xlabel('Factor Values')
    plt.ylabel('CDF')
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)
# # Function to plot comparisons between EHI and Entity F1 Score
# def plot_comparison(data: pd.DataFrame):
#     # Plotting EHI vs Entity F1 Score
#     plt.figure(figsize=(10, 8))
#     st.write("Columns: ", data.columns)
    
#     # Comparing EHI and Entity F1 Score
#     plt.subplot(2, 1, 1)
#     plt.plot(data['EHI'], label='EHI', color='b', linestyle='-', marker='o')
#     plt.plot(data['Entity_F1 Score'], label='Entity F1 Score', color='r', linestyle='-', marker='x')
#     plt.title('Comparison of EHI and Entity F1 Score')
#     plt.xlabel('Index')
#     plt.ylabel('Values')
#     plt.legend()
#     plt.grid(True)
    
#     # Plotting variance and mean comparison
#     ehi_variance = np.var(data['EHI'])
#     ehi_mean = np.mean(data['EHI'])
#     entity_f1_variance = np.var(data['Entity_F1 Score'])
#     entity_f1_mean = np.mean(data['Entity_F1 Score'])
    
#     # Plotting the variance and mean of both metrics
#     plt.subplot(2, 1, 2)
#     bar_width = 0.35
#     indices = np.arange(2)
#     values = [ehi_mean, entity_f1_mean]
#     variance_values = [ehi_variance, entity_f1_variance]
    
#     plt.bar(indices, values, bar_width, label='Mean', color='g')
#     plt.bar(indices + bar_width, variance_values, bar_width, label='Variance', color='y')
    
#     plt.title('Mean and Variance Comparison')
#     plt.xticks(indices + bar_width / 2, ['EHI', 'Entity F1 Score'])
#     plt.ylabel('Value')
#     plt.legend()
#     plt.grid(True)
    
#     st.pyplot(plt)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
# ORIFINAL PLOT COMPARISON FUNCTION BEFORE ADDING NORMALIZATION PLOT
# # Function to plot comparisons between EHI and Entity F1 Score
# def plot_comparison(data: pd.DataFrame):
#     # Create a figure for EHI vs Entity F1 Score
#     fig, axes = plt.subplots(2, 1, figsize=(10, 12))
#     st.write("Columns: ", data.columns)
    
#     # Subplot 1: Comparing EHI and Entity F1 Score
#     axes[0].plot(data['EHI'], label='EHI', color='b', linestyle='-', marker='o')
#     axes[0].plot(data['Entity F1 Score'], label='Entity F1 Score', color='r', linestyle='-', marker='x')
#     axes[0].set_title('Comparison of EHI and Entity F1 Score')
#     axes[0].set_xlabel('Index')
#     axes[0].set_ylabel('Values')
#     axes[0].legend()
#     axes[0].grid(True)
    
#     # Subplot 2: Plotting mean and variance comparison
#     ehi_variance = np.var(data['EHI'])
#     ehi_mean = np.mean(data['EHI'])
#     entity_f1_variance = np.var(data['Entity F1 Score'])
#     entity_f1_mean = np.mean(data['Entity F1 Score'])
    
#     bar_width = 0.35
#     indices = np.arange(2)
#     values = [ehi_mean, entity_f1_mean]
#     variance_values = [ehi_variance, entity_f1_variance]
    
#     axes[1].bar(indices, values, bar_width, label='Mean', color='g')
#     axes[1].bar(indices + bar_width, variance_values, bar_width, label='Variance', color='y')
#     axes[1].set_title('Mean and Variance Comparison')
#     axes[1].set_xticks(indices + bar_width / 2)
#     axes[1].set_xticklabels(['EHI', 'Entity F1 Score'])
#     axes[1].set_ylabel('Value')
#     axes[1].legend()
#     axes[1].grid(True)
    
#     # Adjust layout and display the plot in Streamlit
#     plt.tight_layout()
#     st.pyplot(fig)
    
#     # Save the plot as a high-quality image
#     graph_file_path = "comparison_plot.png"
#     fig.savefig(graph_file_path, dpi=300, bbox_inches='tight')
    
#     # Saving mean and variance data to CSV
#     summary_data = {
#         'Metric': ['EHI', 'Entity F1 Score'],
#         'Mean': [ehi_mean, entity_f1_mean],
#         'Variance': [ehi_variance, entity_f1_variance]
#     }
#     summary_df = pd.DataFrame(summary_data)
    
#     # Saving the summary DataFrame to a CSV file
#     csv_file_path = "mean_variance_comparison.csv"
#     summary_df.to_csv(csv_file_path, index=False)
    
#     # Display the CSV and provide download links
#     st.write("CSV file with Mean and Variance values:")
#     st.dataframe(summary_df)  # Show the CSV data in the app
    
#     # Provide download link for the CSV
#     st.download_button(
#         label="Download CSV",
#         data=summary_df.to_csv(index=False),
#         file_name=csv_file_path,
#         mime="text/csv"
#     )
    
#     # Provide download link for the graph
#     st.write("Download the plot as an image:")
#     with open(graph_file_path, "rb") as img_file:
#         st.download_button(
#             label="Download Plot (PNG)",
#             data=img_file,
#             file_name=graph_file_path,
#             mime="image/png"
#         )

# Function to plot comparisons between EHI and Entity F1 Score
# Function to normalize a column
def normalize_column(column: pd.Series) -> pd.Series:
    """
    Normalize a given column using min-max normalization.
    Args:
        column (pd.Series): The column to be normalized.
    Returns:
        pd.Series: The normalized column.
    """
    return (column - column.min()) / (column.max() - column.min())

def plot_comparison(data: pd.DataFrame):
    # Normalize the EHI column
    data['Normalized_EHI'] = normalize_column(data['EHI'])

    # Create a figure for plots with sufficient subplots (3 subplots per model for 4 models)
    fig, axes = plt.subplots(12, 1, figsize=(10, 30))  # 3 subplots per model for 4 models = 12 total

    st.write("Columns: ", data.columns)
    
    # List of model names
    models = ['facebook/bart-large-cnn', 'google/pegasus-xsum', 't5-large', 'gpt-3.5-turbo']
    
    # Subplot 1: Comparing EHI and Entity F1 Score for each model
    for i, model in enumerate(models):
        axes[i * 3].plot(data['EHI'], label='EHI', color='b', linestyle='-', marker='o')
        axes[i * 3].plot(data['Entity F1 Score'], label='Entity F1 Score', color='r', linestyle='-', marker='x')
        axes[i * 3].set_title(f'Comparison of EHI and Entity F1 Score for {model}')
        axes[i * 3].set_xlabel('Index')
        axes[i * 3].set_ylabel('Values')
        axes[i * 3].legend()
        axes[i * 3].grid(True)
    
    # Subplot 2: Normalized EHI vs Entity F1 Score for each model
    for i, model in enumerate(models):
        axes[i * 3 + 1].plot(data['Normalized_EHI'], label='Normalized EHI', color='purple', linestyle='-', marker='s')
        axes[i * 3 + 1].plot(data['Entity F1 Score'], label='Entity F1 Score', color='r', linestyle='-', marker='x')
        axes[i * 3 + 1].set_title(f'Comparison of Normalized EHI and Entity F1 Score for {model}')
        axes[i * 3 + 1].set_xlabel('Index')
        axes[i * 3 + 1].set_ylabel('Values')
        axes[i * 3 + 1].legend()
        axes[i * 3 + 1].grid(True)
    
    # Subplot 3: Mean and variance comparison for each model
    for i, model in enumerate(models):
        ehi_variance = np.var(data['EHI'])
        ehi_mean = np.mean(data['EHI'])
        entity_f1_variance = np.var(data['Entity F1 Score'])
        entity_f1_mean = np.mean(data['Entity F1 Score'])
        
        bar_width = 0.35
        indices = np.arange(2)
        values = [ehi_mean, entity_f1_mean]
        variance_values = [ehi_variance, entity_f1_variance]
        
        axes[i * 3 + 2].bar(indices, values, bar_width, label='Mean', color='g')
        axes[i * 3 + 2].bar(indices + bar_width, variance_values, bar_width, label='Variance', color='y')
        axes[i * 3 + 2].set_title(f'Mean and Variance Comparison for {model}')
        axes[i * 3 + 2].set_xticks(indices + bar_width / 2)
        axes[i * 3 + 2].set_xticklabels(['EHI', 'Entity F1 Score'])
        axes[i * 3 + 2].set_ylabel('Value')
        axes[i * 3 + 2].legend()
        axes[i * 3 + 2].grid(True)
    
    # Adjust layout and display the plot in Streamlit
    plt.tight_layout()
    st.pyplot(fig)
    
    # Save the plot as a high-quality image
    graph_file_path = "modelwise_comparison_plot.png"
    fig.savefig(graph_file_path, dpi=300, bbox_inches='tight')
    
    # Saving mean and variance data to CSV for all models
    summary_data = {
        'Model': models,
        'Metric': ['EHI', 'Entity F1 Score'],
        'Mean': [ehi_mean, entity_f1_mean],
        'Variance': [ehi_variance, entity_f1_variance]
    }
    summary_df = pd.DataFrame(summary_data)
    
    # Saving the summary DataFrame to a CSV file
    csv_file_path = "mean_variance_comparison_modelwise.csv"
    summary_df.to_csv(csv_file_path, index=False)
    
    # Display the CSV and provide download links
    st.write("CSV file with Mean and Variance values for each model:")
    st.dataframe(summary_df)  # Show the CSV data in the app
    
    # Provide download link for the CSV
    st.download_button(
        label="Download CSV",
        data=summary_df.to_csv(index=False),
        file_name=csv_file_path,
        mime="text/csv"
    )
    
    # Provide download link for the graph
    st.write("Download the model-wise comparison plot as an image:")
    with open(graph_file_path, "rb") as img_file:
        st.download_button(
            label="Download Plot (PNG)",
            data=img_file,
            file_name=graph_file_path,
            mime="image/png"
        )
def process_json(data: Union[dict, List[dict]]):
    if isinstance(data, dict):
        # Single record
        data = [data]
    
    results = []
    for item in data:
        input_text = item['InputText']
        reference_text = item['Reference']
        model_metrics = {}

        # Updated model column names
        model_columns = [
            "outputs_baseline_generator_network",
            "outputs_c4_cnn_dailymail_pegasus",
            "outputs_cnndm_bart",
            "outputs_cnndm_transformerabs_pretrained_encoders"
        ]

        # Compute metrics for each model separately
        for model in model_columns:
            generated_text = item[model]
            ehi, ef, ph, of, nh, lf, ef1 = Compute_EHI(input_text, reference_text, generated_text)
            model_metrics[model] = {
                "EHI": ehi,
                "Extractiveness Factor (EF)": ef,
                "Positive Hallucination (PH)": ph,
                "Over Focus (OF)": of,
                "Negative Hallucination (NH)": nh,
                "Lost Focus (LF)": lf,
                "Entity F1 Score": ef1
            }
        
        # Add metrics to the item
        item["ModelMetrics"] = model_metrics
        results.append(item)
    
    return results
# Streamlit app
def main():
    st.title("Entity Hallucination Index (EHI) Calculator")

    # Single Text Evaluation
    st.header("Single Text Evaluation")
    input_text = st.text_area("Input Text")
    reference_text = st.text_area("Reference Text")
    generated_text = st.text_area("Generated Text")
    
    if st.button("Compute EHI for Single Text"):
        ehi, ef, ph, of, nh, lf, EF1 = Compute_EHI(input_text, reference_text, generated_text)
        st.write(f"EHI: {ehi}")
        st.write(f"Extractiveness Factor (EF): {ef}")
        st.write(f"Positive Hallucination (PH): {ph}")
        st.write(f"Over Focus (OF): {of}")
        st.write(f"Negative Hallucination (NH): {nh}")
        st.write(f"Lost Focus (LF): {lf}")
        st.write(f"Entity F1 Score: {EF1}")

    # Batch Text Evaluation
    st.header("Batch Text Evaluation")
    uploaded_file = st.file_uploader("Upload a CSV or JSON file", type=["csv", "json"])
    
    if uploaded_file is not None:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
    
            # Ensure correct column names exist
            required_columns = ['InputText', 'Reference', 'id']
            model_columns = [
                "outputs_baseline_generator_network",
                "outputs_c4_cnn_dailymail_pegasus",
                "outputs_cnndm_bart",
                "outputs_cnndm_transformerabs_pretrained_encoders",
                "outputs_base.aligned_text_to_text"
            ]
            
            missing_columns = [col for col in required_columns + model_columns if col not in df.columns]
            if missing_columns:
                st.error(f"Missing required columns: {missing_columns}")
                return
            
            # Initialize lists to store results
            results = []

            for _, row in df.iterrows():
                input_text = row['InputText']
                reference_text = row['Reference']

                for model in model_columns:
                    generated_text = row[model]

                    # Compute the metrics
                    EHI, EF, PH, OF, NH, LF, EF1 = Compute_EHI(input_text, reference_text, generated_text)

                    # Append results
                    results.append({
                        'id': row['id'],
                        'Model': model,
                        'EHI': EHI,
                        'Extractiveness Factor (EF)': EF,
                        'Positive Hallucination (PH)': PH,
                        'Over Focus (OF)': OF,
                        'Negative Hallucination (NH)': NH,
                        'Lost Focus (LF)': LF,
                        'Entity F1 Score': EF1
                    })

            # Convert results to DataFrame
            result_df = pd.DataFrame(results)
            st.write(result_df)

            # Allow the user to download results as CSV
            st.download_button("Download Results as CSV", result_df.to_csv(index=False).encode('utf-8'), "EHI_results.csv", "text/csv")

            # Plotting options
            if st.button("Plot CDFs"):
                plot_cdfs(result_df)
            if st.button("Plot Comparison"):
                plot_comparison(result_df)
            if st.button("Plot CDFs for Entity_F1 and EHI Metric"):
                plot_cdfs1(result_df)
            if st.button("Display Average Scores in Table"):
                display_avg_scores(result_df)

            # Compute and save model-wise average metrics
            compute_and_save_metrics(result_df)

        elif uploaded_file.name.endswith('.json'):
            data = json.load(uploaded_file)
            processed_data = process_json(data)

            # Convert to DataFrame for better visualization
            processed_df = pd.json_normalize(processed_data, 'ModelMetrics', ['id'], sep='_')
            st.write(processed_df)

            # Allow downloading JSON results
            st.download_button("Download Results as JSON", json.dumps(processed_data, indent=4).encode('utf-8'), "EHI_results.json", "application/json")

    else:
        st.info("Please upload a CSV or JSON file to evaluate the models.")

if __name__ == "__main__":
    main()
