import streamlit as st
import os
import pandas as pd
from sdv.datasets.local import load_csvs
from sdv.metadata import SingleTableMetadata
from sdv.single_table import GaussianCopulaSynthesizer

def main():

    st.set_page_config(page_title='Synthesize Data')
    st.header('Synthesize Data')
    train_data= st.file_uploader('upload CSV', type='csv')
    if train_data is not None:
        user_text= st.text_input('enter the texts')
        df = pd.read_csv(train_data)
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(data=df)
        st.write(metadata)
        st.write("Synthesized data (100 records)")
        # Step 1: Create the synthesizer
        synthesizer = GaussianCopulaSynthesizer(metadata)

        # Step 2: Train the synthesizer
        synthesizer.fit(df)

        # Step 3: Generate synthetic data
        synthetic_data = synthesizer.sample(num_rows=100)
        st.write(synthetic_data)

if __name__== "__main__":
    main()
