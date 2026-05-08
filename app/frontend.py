import streamlit as st
import requests

# Page configuration
st.set_page_config(page_title="Movie/Book Genre Classifier", page_icon="🎬")

st.title("Movie/Book Genre Classifier")
st.markdown("""
This application uses a **RoBERTa-base** model to predict the genre of book summaries or movie plots.
""")

# Input section
text_input = st.text_area("Enter the movie Plot or Book Summary:", placeholder="A retired soldier is forced back into action...", height=200)
threshold = st.slider("Confidence Threshold", 0.1, 1.0, 0.75, help="Lower values increase sensitivity, higher values increase precision.")

if st.button("Classify Genre"):
    if text_input.strip() == "":
        st.error("Please enter some text to classify.")
    else:
        with st.spinner("Analyzing text with RoBERTa..."):
            try:
                backend_url = "http://backend-service:8000/predict" 
                
                payload = {
                    "text": text_input,
                    "threshold": threshold
                }
                
                response = requests.post(backend_url, json=payload)
                
                if response.status_code == 200:
                    data = response.json()
                    genres = data.get("genres", [])
                    probs = data.get("probabilities", {})

                    # Display Results
                    st.subheader("Predicted Genres")
                    if genres:
                        genre_cols = st.columns(len(genres))
                        for i, g in enumerate(genres):
                            genre_cols[i].success(f"**{g}**")
                    else:
                        st.info("No genres met the confidence threshold.")

                    # Probability Visualization
                    st.subheader("Confidence Scores")
                    # Sort probabilities for better visualization
                    sorted_probs = dict(sorted(probs.items(), key=lambda item: item[1], reverse=True))
                    st.bar_chart(sorted_probs)
                else:
                    st.error(f"Backend Error: {response.text}")
                    
            except Exception as e:
                st.error(f"Could not connect to backend. Is FastAPI running? Error: {e}")

st.divider()
st.caption("Made with ❤️ by Abhishek Ranjan and Abhishek Mandal")