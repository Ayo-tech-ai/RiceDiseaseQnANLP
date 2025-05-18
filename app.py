import streamlit as st
from transformers import pipeline
import time

# -------------------------------
# 1. Load Q&A Model
# -------------------------------

@st.cache_resource
def load_qa_model():
    return pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

qa_pipeline = load_qa_model()

# -------------------------------
# 2. Context Sections (Full Knowledge Base)
# -------------------------------

sections = {
    "bacterial leaf blight": """Bacterial Leaf Blight is caused by the bacterium Xanthomonas oryzae pv. oryzae. It presents symptoms like water-soaked lesions, yellowing, and wilting, and may lead to up to 70% yield loss. The disease spreads through rain, irrigation, and contaminated seeds or tools. Control involves growing resistant varieties such as IR20 and IR64, and applying copper sprays and crop rotation.""",

    "bacterial leaf streak": """Bacterial Leaf Streak is caused by Xanthomonas oryzae pv. oryzicola. It causes narrow, water-soaked streaks between leaf veins, resulting in 10–30% yield loss. The pathogen spreads via wind-driven rain and infected seeds. Management includes planting resistant varieties like IR24 and IR50, and spraying copper hydroxide.""",

    "bakanae": """Bakanae Disease, also known as “Foolish Seedling,” is caused by the fungus Fusarium fujikuroi. It leads to tall, thin, fragile seedlings with hollow stems, resulting in 20–70% yield loss. The disease is transmitted through seeds and contaminated soil. Prevention includes hot water seed treatment and fungicide application with carbendazim.""",

    "brown spot": """Brown Spot, or Helminthosporiosis, is caused by Bipolaris oryzae and shows up as circular brown lesions with a bullseye pattern. It may cause 10–90% yield loss and is spread by windborne spores and poor soil fertility. Management includes applying potassium or silicon fertilizers and fungicides like mancozeb.""",

    "grassy stunt": """Rice Grassy Stunt Virus (RGSV) is a viral disease spread by the brown planthopper. It results in stunted growth, pale leaves, and sterile tillers, with potential for 100% crop loss. Control strategies include using resistant varieties such as IR36 and IR56, and practicing synchronized planting.""",

    "narrow brown spot": """Narrow Brown Spot, caused by the fungus Cercospora janseana, leads to thin brown streaks on leaves and yield losses between 5–40%. It spreads via windborne spores and potassium deficiency. Balanced fertilization and fungicides like propiconazole help manage the disease.""",

    "ragged stunt": """Rice Ragged Stunt Virus (RRSV), also spread by brown planthoppers, causes ragged and twisted leaves, and stunting. It can cause 30–70% yield loss. Using resistant varieties like IR36 and IR72, along with Integrated Pest Management (IPM), can reduce its impact.""",

    "rice blast": """Rice Blast, caused by the fungus Magnaporthe oryzae, forms diamond-shaped lesions and can result in "rotten neck" symptoms. It may lead to total crop failure. Airborne spores and high humidity facilitate its spread. Management includes using resistant varieties and tricyclazole fungicide.""",

    "false smut": """Rice False Smut, caused by Ustilaginoidea virens, produces greenish-black spore balls that replace rice grains and reduce yield by 5–40%. It also poses a mycotoxin risk. It spreads during flowering via airborne spores and is managed by spraying propiconazole at the booting stage.""",

    "sheath blight": """Sheath Blight is caused by Rhizoctonia solani and results in collar rot, sheath lesions, and lodging. Yield losses range from 20–70%. It spreads through soil-borne sclerotia and is managed by wider spacing and fungicides like validamycin.""",

    "sheath rot": """Sheath Rot, caused by Sarocladium oryzae, results in sheath rotting and shriveled grains, leading to 20–70% yield loss. It spreads via airborne spores and wounds. Treatment includes carbendazim and biological control with Trichoderma.""",

    "stem rot": """Stem Rot, caused by Sclerotium oryzae, is characterized by black sclerotia in the stem and can cause 30–80% yield loss. The disease spreads through sclerotia in the soil. Management strategies include applying silicon and rotating crops.""",

    "tungro": """Rice Tungro Virus is a dual infection caused by Rice Tungro Bacilliform and Spherical Viruses. Spread by green leafhoppers, it causes yellow-orange leaves and stunting and can lead to total crop failure. Control methods include planting resistant varieties like IR36 and using neem-based repellents."""
}

# -------------------------------
# 3. Streamlit UI
# -------------------------------

st.title("Rice Disease Q&A Assistant")
st.caption("Ask a question about rice diseases. Include the disease name for best results (e.g., 'What causes rice blast?').")

question = st.text_input("Your question:")
submit = st.button("Submit")

# -------------------------------
# 4. Process the Question
# -------------------------------

if submit and question:
    with st.spinner("Analyzing your question..."):
        progress = st.progress(0)
        for i in range(50):
            time.sleep(0.01)
            progress.progress(i + 1)

        selected_context = None
        for keyword, section in sections.items():
            if keyword in question.lower():
                selected_context = section
                break

        if selected_context:
            result = qa_pipeline(question=question, context=selected_context)
            progress.progress(100)
            st.success("Answer:")
            st.write(result["answer"])
        else:
            progress.progress(100)
            st.warning("I couldn’t match your question to a specific disease. Try including the name, like 'blast', 'blight', or 'tungro'.")
