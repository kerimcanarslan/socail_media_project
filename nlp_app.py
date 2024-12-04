import streamlit as st
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import base64  # Arka plan resmi için gerekli


# Model Import
model_load_path = "./bert_fine_tuned_model"

tokenizer = BertTokenizer.from_pretrained(model_load_path)
model = BertForSequenceClassification.from_pretrained(model_load_path)

def load_data():
    conclusions = pd.read_csv("conclusions.csv")
    opinions = pd.read_csv("opinions.csv")
    return conclusions, opinions

conclusions, opinions = load_data()

# Tür haritası
type_mapping = {0: "Claim", 1: "Counterclaim", 2: "Evidence", 3: "Rebuttal"}

# Metni sınıflandırma fonksiyonu
def predict_type(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()
    return predicted_class

# İlgili metni bulma fonksiyonu
def get_related_text(text, final_output, grouped_column="grouped_texts"):
    predicted_class = predict_type(text)
    predicted_type_name = type_mapping[predicted_class]

    topic_id_row = final_output[final_output['text'] == text]
    if topic_id_row.empty:
        return "Text not found in the dataset."

    topic_id = topic_id_row.iloc[0]['topic_id']

    related_texts = final_output[final_output['topic_id'] == topic_id].iloc[0][grouped_column][predicted_type_name]
    if len(related_texts) > 0:
        for related_text in related_texts:
            if related_text != text:
                return related_text
        return f"No other '{predicted_type_name}' texts available in the same topic."
    else:
        return f"No '{predicted_type_name}' texts found in grouped_texts."

# Veri işleme
test_topic_ids = conclusions['topic_id'].unique()[:10]

test_conclusions = conclusions[conclusions['topic_id'].isin(test_topic_ids)].copy()
test_conclusions['predicted_type_name'] = test_conclusions['text'].apply(predict_type)

grouped_opinions = opinions[opinions['topic_id'].isin(test_topic_ids)].groupby('topic_id').apply(
    lambda x: {
        'Claim': x[x['type'] == 'Claim']['text'].tolist(),
        'Counterclaim': x[x['type'] == 'Counterclaim']['text'].tolist(),
        'Evidence': x[x['type'] == 'Evidence']['text'].tolist(),
        'Rebuttal': x[x['type'] == 'Rebuttal']['text'].tolist()
    }
).reset_index(name='grouped_texts')

final_output = pd.merge(
    test_conclusions[['topic_id', 'text', 'predicted_type_name']],
    grouped_opinions,
    on='topic_id',
    how='left'
)

# Arka plan resmi ayarlama
def set_background_image(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded_string}");
            background-size: cover;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Arka plan resmi dosya yolunu belirtin
set_background_image("newmind_ai.png")

# Streamlit arayüzü
st.title("Text Classification and Related Text Finder")

# Kullanıcıdan bir metin seçmesini iste
st.write("Select a text from the dataset to classify and find related answers:")
selected_text = st.selectbox("Available Texts:", test_conclusions['text'].unique())

# Sınıflandırma butonu
if st.button("Classify Text"):
    if selected_text.strip() == "":
        st.warning("Please select a text to classify.")
    else:
        predicted_type_name = type_mapping[predict_type(selected_text)]
        st.write(f"**Selected Text:** {selected_text}")
        st.write(f"**Predicted Type:** {predicted_type_name}")

# İlgili yanıt bulma butonu
if st.button("Find Related Text"):
    if selected_text.strip() == "":
        st.warning("Please select a text to find related answers.")
    else:
        related_text = get_related_text(selected_text, final_output)
        st.write(f"**Related Text:** {related_text}")
