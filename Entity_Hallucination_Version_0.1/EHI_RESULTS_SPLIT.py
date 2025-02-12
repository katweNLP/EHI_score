import streamlit as st
import pandas as pd
import io

# Streamlit app title
st.title("CSV Indexing & Splitting App")

# Upload CSV file
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    # Read CSV into DataFrame
    df = pd.read_csv(uploaded_file)

    # Check if CSV has at least 400 rows
    if len(df) < 400:
        st.error("Error: The uploaded CSV must have at least 400 rows.")
    else:
        # Add index column from 0 to 399
        df.insert(0, "Index", range(0, 400))

        # Split the DataFrame
        df_0_199 = df.iloc[:200]  # Rows 0 to 199
        df_200_399 = df.iloc[200:400]  # Rows 200 to 399

        # Reset index in the second file (200-399 → 0-199)
        df_200_399["Index"] = range(0, 200)

        # Function to convert DataFrame to CSV for download
        def convert_df_to_csv(df):
            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False)
            return csv_buffer.getvalue()

        # Convert DataFrames to CSV
        csv_0_199 = convert_df_to_csv(df_0_199)
        csv_200_399 = convert_df_to_csv(df_200_399)

        # Download buttons
        st.download_button("Download CSV (Index 0-199)", csv_0_199, "data_0_199.csv", "text/csv")
        st.download_button("Download CSV (Index 0-199) [Renamed]", csv_200_399, "data_200_399.csv", "text/csv")

        # Show preview of both files
        st.subheader("Preview of Index 0-199 CSV:")
        st.dataframe(df_0_199.head())

        st.subheader("Preview of Index 0-199 CSV (Previously 200-399):")
        st.dataframe(df_200_399.head())
