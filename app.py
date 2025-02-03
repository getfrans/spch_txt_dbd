import streamlit as st
import duckdb
import pandas as pd
import numpy as np
from datetime import datetime
import whisper
from transformers import MistralForCausalLM, AutoTokenizer

pip install openai-whisper transformers
import torch
import tempfile
import os

# Initialize Streamlit page configuration
st.set_page_config(
    page_title="Audio Analytics Dashboard",
    layout="wide"
)

# Initialize models
@st.cache_resource
def load_models():
    whisper_model = whisper.load_model("base")
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
    mistral_model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")
    return whisper_model, mistral_model, tokenizer

# Initialize DuckDB
def init_database():
    conn = duckdb.connect('analytics.db')
    
    # Create tables if they don't exist
    conn.execute("""
        CREATE TABLE IF NOT EXISTS transcriptions (
            id INTEGER PRIMARY KEY,
            filename VARCHAR,
            transcription TEXT,
            timestamp TIMESTAMP,
            analysis TEXT
        )
    """)
    
    # Create sample data if needed
    conn.execute("""
        INSERT OR IGNORE INTO transcriptions 
        SELECT 
            1 as id,
            'sample_audio.wav' as filename,
            'This is a sample transcription.' as transcription,
            CURRENT_TIMESTAMP as timestamp,
            'Sample analysis' as analysis
        WHERE NOT EXISTS (SELECT 1 FROM transcriptions WHERE id = 1)
    """)
    
    return conn

# Streamlit UI
def main():
    st.title("Audio Analytics Dashboard")
    
    # Initialize models and database
    try:
        whisper_model, mistral_model, tokenizer = load_models()
        conn = init_database()
    except Exception as e:
        st.error(f"Error initializing models or database: {str(e)}")
        return
    
    # Sidebar for file upload and controls
    with st.sidebar:
        st.header("Controls")
        uploaded_file = st.file_uploader("Upload Audio File", type=['wav', 'mp3'])
        
        if uploaded_file:
            if st.button("Process Audio"):
                with st.spinner("Processing audio..."):
                    # Save uploaded file temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        audio_path = tmp_file.name
                    
                    # Transcribe audio
                    try:
                        result = whisper_model.transcribe(audio_path)
                        transcription = result["text"]
                        
                        # Generate analysis with Mistral
                        prompt = f"Analyze this transcription and provide key insights: {transcription}"
                        inputs = tokenizer(prompt, return_tensors="pt")
                        outputs = mistral_model.generate(**inputs, max_length=200)
                        analysis = tokenizer.decode(outputs[0])
                        
                        # Store in DuckDB
                        conn.execute("""
                            INSERT INTO transcriptions (filename, transcription, timestamp, analysis)
                            VALUES (?, ?, CURRENT_TIMESTAMP, ?)
                        """, (uploaded_file.name, transcription, analysis))
                        
                        st.success("Audio processed successfully!")
                    except Exception as e:
                        st.error(f"Error processing audio: {str(e)}")
                    finally:
                        os.unlink(audio_path)
    
    # Main content area
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("Recent Transcriptions")
        transcriptions_df = conn.execute("""
            SELECT filename, transcription, timestamp
            FROM transcriptions
            ORDER BY timestamp DESC
            LIMIT 5
        """).df()
        st.dataframe(transcriptions_df)
    
    with col2:
        st.header("Analysis Insights")
        analysis_df = conn.execute("""
            SELECT filename, analysis, timestamp
            FROM transcriptions
            ORDER BY timestamp DESC
            LIMIT 5
        """).df()
        st.dataframe(analysis_df)
    
    # Analytics section
    st.header("Analytics")
    
    # Word count analysis
    word_counts = conn.execute("""
        SELECT 
            filename,
            LENGTH(transcription) - LENGTH(REPLACE(transcription, ' ', '')) + 1 as word_count
        FROM transcriptions
        ORDER BY timestamp DESC
        LIMIT 10
    """).df()
    
    st.bar_chart(word_counts.set_index('filename'))

if __name__ == "__main__":
    main()
