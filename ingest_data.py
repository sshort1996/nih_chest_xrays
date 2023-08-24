import pandas as pd
import streamlit as st
import openai
import os


# Specify the path to your CSV file
csv_file_path = "sample/sample_labels.csv"

# Read the CSV file into a pandas DataFrame
df = pd.read_csv(csv_file_path)

# Create a new DataFrame with each condition in its own row
df['Finding Labels'] = df['Finding Labels'].str.split('|')
df_exploded = df.explode('Finding Labels')

# Set the number of rows to display for each condition
n_rows_per_condition = 10
selected_examples = {}

for label in df_exploded['Finding Labels'].unique():
    selected_examples[label] = df_exploded[df_exploded['Finding Labels'] == label].head(n_rows_per_condition)
    

# Set up OpenAI API credentials
# Replace 'YOUR_API_KEY' with your actual OpenAI API key
openai.api_key = os.environ['openai_key']

# Generate short descriptions using OpenAI API
descriptions = {}
for cond in selected_examples.keys():    
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=f"""Describe the condition: {cond}\nKeep your responses reasonably brief.\n
        Also mention how one might identify this condition on a chest xray""",
        max_tokens=100,
        n=1
    )
    description = response.choices[0].text.strip()
    descriptions[cond] = description

# Display images and short descriptions
for cond in selected_examples.keys():    
    st.write(f'## {cond}')
    cols = st.columns(5)
    to_plot = selected_examples[cond]['Image Index']
    for i, path_stem in enumerate(to_plot):
        image_path = f"sample/images/{path_stem}"
        cols[i % 5].image(image_path, caption=f"Image Index: {path_stem}", use_column_width=True)
    st.write(descriptions[cond])