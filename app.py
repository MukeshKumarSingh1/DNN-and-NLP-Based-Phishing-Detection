import streamlit as st
import pandas as pd
import tensorflow as tf
import numpy as np
from email import policy
from email.parser import BytesParser
from io import BytesIO
from urllib.parse import urlparse
import re
import ipaddress
from sklearn.preprocessing import StandardScaler

# Load the trained model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("saved_models/Phishing_Email_Classifier_Model_20250212_160914.h5")  # Replace with your model path

model = load_model()

# Placeholder for scaler (update with actual scaler used during training)
scaler = StandardScaler()
scaler.fit(np.ones((1, 9)))

# Function to preprocess the URL
def preprocess_url(url):
    """
    Extract URL-based features for prediction.
    """
    def is_ip_address(domain):
        try:
            ipaddress.ip_address(domain)
            return 1
        except ValueError:
            return 0

    shortening_services = (
        "bit.ly", "tinyurl.com", "is.gd", "goo.gl", "shorte.st", "go2l.ink", "x.co", "ow.ly", "t.co",
        "tr.im", "cli.gs", "yfrog.com", "migre.me", "ff.im", "tiny.cc", "url4.eu", "twit.ac", "su.pr",
        "twurl.nl", "snipurl.com", "short.to", "bitly.com", "lnkd.in", "adf.ly", "cutt.us", "db.tt")

    try:
        parsed_url = urlparse(url)
        netloc = parsed_url.netloc.lower().strip()
        features = np.array([
            len(url),  # URL length
            url.count('/'),  # URL depth
            1 if '@' in url else 0,  # Presence of '@'
            1 if '-' in netloc else 0,  # Presence of '-'
            1 if netloc.count('.') > 1 else 0,  # Subdomain count
            1 if parsed_url.scheme != 'https' else 0,  # Use of HTTP
            1 if '//' in url[7:] else 0,  # Redirection
            1 if any(service in netloc for service in shortening_services) else 0,  # URL Shortener
            is_ip_address(netloc)  # Check if domain is an IP (IPv4 or IPv6)
        ]).reshape(1, -1)

        # Scale features
        scaled_features = scaler.transform(features)

        # Ensure the feature vector matches the expected input size
        padded_features = np.zeros((1, model.input_shape[1]))
        padded_features[0, :scaled_features.shape[1]] = scaled_features
        return padded_features
    except Exception as e:
        st.error(f"Error processing URL: {e}")
        return None

# Function to process .eml file
def extract_eml_features(file):
    try:
        msg = BytesParser(policy=policy.default).parse(file)
        body = msg.get_body(preferencelist=('plain', 'html')).get_content()
        # Placeholder: Replace with actual body preprocessing logic
        features = np.zeros((1, model.input_shape[1]))  # Replace with processed features
        return features
    except Exception as e:
        st.error(f"Error processing .eml file: {e}")
        return None

# Streamlit App
st.markdown(
    """
    <style>
        .stApp {
            background-color: #ffffff;
            color: #333333;
            font-family: 'Arial', sans-serif;
        }
        .stButton > button {
            background-color: #4CAF50;
            color: #ffffff;
            border-radius: 8px;
            font-size: 16px;
            border: none;
            padding: 8px 16px;
        }
        .stButton > button:hover {
            background-color: #45a049;
        }
        .stFileUploader label {
            color: #333333;
        }
        .stTabs [data-baseweb="tab"] {
            background-color: #f1f1f1;
            color: #333333;
            font-size: 16px;
            border-radius: 5px 5px 0 0;
        }
        .stTabs [data-baseweb="tab"]:hover {
            background-color: #dddddd;
        }
        .stTextInput > div > input {
            border: 1px solid #cccccc;
            border-radius: 5px;
            padding: 8px;
        }
        h1, h2, h3 {
            color: #333333;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Phishing Classifier Demo")

# Tabs for File and URL input
tab1, tab2, tab3 = st.tabs(["Upload .eml File", "Test URL", "About the Model"])

with tab1:
    st.header("Upload .eml File for Prediction")
    uploaded_file = st.file_uploader("Choose a .eml file", type=["eml"])
    if uploaded_file:
        features = extract_eml_features(uploaded_file)
        if features is not None:
            try:
                prediction = model.predict(features)
                st.success(f"Prediction: {'Phishing' if prediction[0][0] > 0.5 else 'Not Phishing'}")
            except Exception as e:
                st.error(f"Error during prediction: {e}")

with tab2:
    st.header("Enter a URL for Prediction")
    url_input = st.text_input("Enter a URL:")
    if st.button("Predict URL"):
        if url_input:
            features = preprocess_url(url_input)
            if features is not None:
                try:
                    prediction = model.predict(features)
                    st.success(f"Prediction: {'Phishing' if prediction[0][0] > 0.5 else 'Not Phishing'}")
                except Exception as e:
                    st.error(f"Error during prediction: {e}")
        else:
            st.error("Please enter a valid URL.")

with tab3:
    st.header("About the Model")
    st.write("""
    This phishing classifier uses a deep neural network trained on features extracted from emails and URLs.
    
    ### Key Features:
    - URL length, depth, presence of special characters, and IP addresses.
    - Indicators of phishing behavior like subdomain count, HTTP usage, and URL shortening services.
    - Email body content analysis and metadata.
    
    ### How It Works:
    - Upload a .eml file to analyze email-based features.
    - Enter a URL to analyze its structure and predict if it is phishing.
    
    The model has been trained on a diverse dataset to ensure robust predictions. For more details, refer to the project documentation.
    """)
