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
        st.subheader("Input Text:")
        st.write(input_text)

        st.subheader("Reference Text:")
        st.write(reference_text)

        ratings = []

        st.subheader("Generated Summaries:")
        for i, summary in enumerate(summaries):
            st.subheader(f"Summary {i + 1}:")
            st.write(summary)
            default_rating = 3  # Default rating value
            rating_key = f"rating_{article_id}_{i}"
            rating = st.slider(f"Rate Summary {i + 1}", min_value=1, max_value=5, step=1,
                               value=default_rating, key=rating_key)
            st.write(f"You rated Summary {i + 1}: {rating}")
            st.write("-" * 50)

            ratings.append({
                "summary": summary,
                "rating": rating
            })

        # Save ratings to JSON file
        save_ratings(article_id, ratings)

    return total_pages


def save_ratings(article_id, ratings):
    with open("ratings.json", "a", encoding="utf-8") as file:
        json.dump({article_id: ratings}, file, ensure_ascii=False)
        file.write("\n")


# Load article data from JSON file
with open("articles.json", "r", encoding="utf-8") as file:
    article_data = json.load(file)

# Streamlit app
st.title("Summary Rating App")

items_per_page = 1  # Number of articles to display per page
total_pages = rate_summaries(article_data, items_per_page)

st.sidebar.write(f"Total Pages: {total_pages}")
