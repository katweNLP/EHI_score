import streamlit as st
import json

def rate_summaries(article_data, items_per_page):
    article_ids = list(article_data.keys())
    total_articles = len(article_ids)
    total_pages = (total_articles + items_per_page - 1) // items_per_page

    page = st.sidebar.number_input("Page", min_value=1, max_value=total_pages, value=1, step=1)
    start_idx = (page - 1) * items_per_page
    end_idx = min(start_idx + items_per_page, total_articles)

    for article_id in article_ids[start_idx:end_idx]:
        article_info = article_data[article_id]
        input_text = article_info["InputText"]
        reference_text = article_info["ReferenceSummary"]
        summaries = [
            article_info["facebook/bart-large-cnn"],
            article_info["google/pegasus-xsum"],
            article_info["t5-large"],
            article_info["gpt-3.5-turbo"]
        ]

        st.subheader(f"Article ID: {article_id}")

        col1, col2, col3 = st.beta_columns([20, 15, 10])
        
        with col1:
            st.subheader("Input Text:")
            st.write(input_text)

            st.subheader("Reference Text:")
            st.write(reference_text)

        with col2:
            st.subheader("Generated Summaries:")
            for i, summary in enumerate(summaries):
                st.subheader(f"Model {i + 1} Summary:")
                st.write(summary)
                st.write("-" * 50)

        with col3:
            st.subheader("Rate Summaries:")
            ratings = {}
            rating=[]
            with open("ratings.json", "r", encoding="utf-8") as file:
                oldrates = json.load(file)
                
            #st.write(oldrates[str(i)])
            
            for i, summary in enumerate(summaries):
               # st.write(oldrates)
               # st.write(article_id)
                rating_key = f"rating_{article_id}_{i}"
                rate = st.slider(f"Rate Model {i + 1} Summary", min_value=1, max_value=10, step=1, key=rating_key, value=int(oldrates[article_id][mlist[i]]))
                st.write(f"You rated Model {i + 1} Summary: {rate}")
                rating.append(rate)
                st.write("-" * 50)

            ratings={
              f"{mlist[0]}": str(rating[0]),
              f"{mlist[1]}": str(rating[1]),
              f"{mlist[2]}": str(rating[2]),
              f"{mlist[3]}": str(rating[3])
              
            }

            # Save ratings to JSON file
            save_ratings(article_id, ratings)

    return total_pages


def save_ratings( id, ratings):
    with open("ratings.json", "r", encoding="utf-8") as file:
        ratings_data = json.load(file)
   # st.write("From file : ")
   # st.write(ratings_data)
   # st.write("From call :")
   # st.write(ratings)
    
    ratings_data[str(id)] =  ratings
   # st.write("printing")
    
   # st.write(ratings_data)
    with open("ratings.json", "w",encoding="utf-8") as file:
        
        json.dump( (ratings_data), file, ensure_ascii=False)
        file.write("\n")



mlist=["facebook/bart-large-cnn","google/pegasus-xsum","t5-large","gpt-3.5-turbo"]

# Load article data from JSON file
with open("articles.json", "r", encoding="utf-8") as file:
    article_data = json.load(file)
st.set_page_config(layout="wide") 
# Streamlit app
st.title("Summary Rating App")

st.markdown(
    """
    <style>
    .sidebar .sidebar-content {
        width: 250px;
    }
    </style>
    """,
    unsafe_allow_html=True
)
items_per_page = 1  # Number of articles to display per page
total_pages = rate_summaries(article_data, items_per_page)

st.sidebar.write(f"Total Pages: {total_pages}")
