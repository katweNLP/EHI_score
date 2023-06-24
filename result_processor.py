import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json 
# Load JSON file


# Load JSON file
@st.cache
def load_data(model):
    df = pd.read_json(model+".json").transpose()
    # Normalize 'EHI' field to a range between 0 and 1
    df["EHI"] = (df["EHI"] - np.min(df["EHI"])) / (np.max(df["EHI"]) - np.min(df["EHI"]))
    return df


modellist=['BART','PEGASUS','T5L','GPT3']

def plot(model):
    
    data = load_data(model)

# Create the 10 figures
    figs = []

# Specify the fields for which you want to create graphs
    fields = ["r1", "r2", "rl", "EF", "PH", "OF", "NH", "LF", "EHI", "EF1"]

    for field in fields:
        fig, ax = plt.subplots()
        ax.plot(data["id"], data[field])
        ax.set_title(model+":"+field)
        ax.set_xlabel('id')
        ax.set_ylabel(field)
        figs.append(fig)
 
# Display the figures
    for i, fig in enumerate(figs):
        st.write(f"Plot for {fields[i]}")
        st.pyplot(fig)


    st.write(data)
# Calculate averages for each field for the first and second half of the data
    first_half_avg = data[:200].mean()
    second_half_avg = data[200:].mean()
    
    st.write(model+":Average scores for Xsum 200 entries:")
    st.write(first_half_avg[1:11])

    st.write(model+":Average scores for XLsum 200 entries:")
    st.write(second_half_avg[1:11])
    


for model in modellist:
    plot(model)

