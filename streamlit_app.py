# streamlit_app.py
import streamlit as st
import requests
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# -----------------------
# Load the model
# -----------------------
hf_repo_id = "MaksICK1287/Product_NLP"
model_subfolder = "product_ner_model_distilbert_4"

@st.cache_resource(show_spinner=True)
def load_model():
    try:
        with st.spinner("Tokenizer load..."):
            tokenizer = AutoTokenizer.from_pretrained(
                hf_repo_id,
                subfolder=model_subfolder,
                use_fast=True
            )
        
        with st.spinner("Model load..."):
            model = AutoModelForTokenClassification.from_pretrained(
                hf_repo_id,
                subfolder=model_subfolder  
            )

        ner_pipe = pipeline(
            "ner",
            model=model,
            tokenizer=tokenizer,
            aggregation_strategy="simple"
        )
        return ner_pipe
        
    except Exception as e:
        st.error(f"Erorr model: {str(e)}")
        with st.expander("Details"):
            st.code(str(e))
        return None


ner_pipeline = load_model()

if ner_pipeline is None:
    st.error("Not able to load the module")
    st.stop()


@st.cache_resource(show_spinner=True)
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForTokenClassification.from_pretrained(model_name_or_path)

    ner_pipe = pipeline(
        "ner",
        model=model,
        tokenizer=tokenizer,
        aggregation_strategy="simple"  # merges tokens into full entities
    )

    return ner_pipe


ner_pipeline = load_model()

# -----------------------
# Streamlit Interface
# -----------------------
st.title("Product Extractor NER")
st.write("Paste a URL and the model will extract products from the page.")

url = st.text_input("Enter URL:")

if st.button("Extract products"):
    
    if not url:
        st.warning("Please enter a URL")

    else:
        try:
            # Get page text
            r = requests.get(url, timeout=10)
            r.raise_for_status()

            soup = BeautifulSoup(r.text, "html.parser")
            text = soup.get_text(separator=" ", strip=True)

            # Apply NER model
            ner_results = ner_pipeline(text)

            products = [
                ent["word"]
                for ent in ner_results
                if ent["entity_group"] == "PRODUCT"
            ]

            products = list(dict.fromkeys(products))  # remove duplicates

            if products:
                st.success(f"Products found ({len(products)}):")
                st.write(products)

            else:
                st.info("No products found.")

        except Exception as e:
            st.error(f"Error processing URL: {e}")
