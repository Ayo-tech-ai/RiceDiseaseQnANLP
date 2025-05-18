import streamlit as st
from transformers import pipeline

@st.cache_resource
def load_qa_model():
    return pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

qa_pipeline = load_qa_model()

context = """
Rice is affected by various diseases with significant impacts on yield. Bacterial Leaf Blight is caused by the bacterium Xanthomonas oryzae pv. oryzae. It presents symptoms like water-soaked lesions, yellowing, and wilting, and may lead to up to 70% yield loss. The disease spreads through rain, irrigation, and contaminated seeds or tools. Control involves growing resistant varieties such as IR20 and IR64, and applying copper sprays and crop rotation.

Bacterial Leaf Streak is another bacterial disease caused by Xanthomonas oryzae pv. oryzicola. It causes narrow, water-soaked streaks between leaf veins, resulting in 10–30% yield loss. The pathogen spreads via wind-driven rain and infected seeds. Management includes planting resistant varieties like IR24 and IR50, and spraying copper hydroxide.

Bakanae Disease, also known as “Foolish Seedling,” is caused by the fungus Fusarium fujikuroi. It leads to tall, thin, fragile seedlings with hollow stems, resulting in 20–70% yield loss. The disease is transmitted through seeds and contaminated soil. Prevention includes hot water seed treatment and fungicide application with carbendazim.

Brown Spot, or Helminthosporiosis, is caused by Bipolaris oryzae and shows up as circular brown lesions with a bullseye pattern. It may cause 10–90% yield loss and is spread by windborne spores and poor soil fertility. Management includes applying potassium or silicon fertilizers and fungicides like mancozeb.

Rice Grassy Stunt Virus (RGSV) is a viral disease spread by the brown planthopper. It results in stunted growth, pale leaves, and sterile tillers, with potential for 100% crop loss. Control strategies include using resistant varieties such as IR36 and IR56, and practicing synchronized planting.

Narrow Brown Spot, caused by the fungus Cercospora janseana, leads to thin brown streaks on leaves and yield losses between 5–40%. It spreads via windborne spores and potassium deficiency. Balanced fertilization and fungicides like propiconazole help manage the disease.

Rice Ragged Stunt Virus (RRSV), also spread by brown planthoppers, causes ragged and twisted leaves, and stunting. It can cause 30–70% yield loss. Using resistant varieties like IR36 and IR72, along with Integrated Pest Management (IPM), can reduce its impact.

Rice Blast, caused by the fungus Magnaporthe oryzae, forms diamond-shaped lesions and can result in "rotten neck" symptoms. It may lead to total crop failure. Airborne spores and high humidity facilitate its spread. Management includes using resistant varieties and tricyclazole fungicide.

Rice False Smut, caused by Ustilaginoidea virens, produces greenish-black spore balls that replace rice grains and reduce yield by 5–40%. It also poses a mycotoxin risk. It spreads during flowering via airborne spores and is managed by spraying propiconazole at the booting stage.

Sheath Blight is caused by Rhizoctonia solani and results in collar rot, sheath lesions, and lodging. Yield losses range from 20–70%. It spreads through soil-borne sclerotia and is managed by wider spacing and fungicides like validamycin.

Sheath Rot, caused by Sarocladium oryzae, results in sheath rotting and shriveled grains, leading to 20–70% yield loss. It spreads via airborne spores and wounds. Treatment includes carbendazim and biological control with Trichoderma.

Stem Rot, caused by Sclerotium oryzae, is characterized by black sclerotia in the stem and can cause 30–80% yield loss. The disease spreads through sclerotia in the soil. Management strategies include applying silicon and rotating crops.

Rice Tungro Virus is a dual infection caused by Rice Tungro Bacilliform and Spherical Viruses. Spread by green leafhoppers, it causes yellow-orange leaves and stunting and can lead to total crop failure. Control methods include planting resistant varieties like IR36 and using neem-based repellents.

Effective management includes preventive measures like resistant varieties, certified seeds, balanced fertilization, removal of infected plants, targeted chemical applications, and long-term practices such as crop rotation and community-based pest control.
"""

st.title("Rice Disease Q&A")

question = st.text_input("Ask a question about rice diseases:")
submit = st.button("Submit")

if submit and question:
    result = qa_pipeline(question=question, context=context)
    st.write("Answer:", result["answer"])
