import streamlit as st
from transformers import pipeline

@st.cache_resource
def load_qa_model():
    return pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

qa_pipeline = load_qa_model()

context = """
Rice Diseases Knowledge Base
Structured for NLP Question Answering (QA) Models

Bacterial Leaf Blight
Cause: Xanthomonas oryzae pv. oryzae
Symptoms: Water-soaked lesions with yellowing, wilting, and wavy margins
Impact: Up to 70% yield loss; affects milling quality
Transmission: Rain splash, irrigation water, contaminated tools and seeds
Management: Resistant varieties (IR20, IR64), copper-based sprays, crop rotation

Bacterial Leaf Streak
Cause: Xanthomonas oryzae pv. oryzicola
Symptoms: Narrow water-soaked streaks between veins
Impact: 10–30% yield loss; leads to chalky grains
Transmission: Wind-driven rain, infected seeds
Management: Resistant varieties (IR24, IR50), copper hydroxide sprays

Bakanae Disease (Foolish Seedling)
Cause: Fusarium fujikuroi (fungus)
Symptoms: Tall, weak seedlings; hollow stems
Impact: 20–70% yield loss; seedling death
Transmission: Seedborne and contaminated soil
Management: Hot water seed treatment, carbendazim fungicide

Brown Spot
Cause: Bipolaris oryzae
Symptoms: Circular brown lesions resembling bullseyes on leaves/grains
Impact: 10–90% yield loss; causes “dirty rice”
Transmission: Windborne spores, poor soil fertility
Management: Potassium/silicon fertilization, mancozeb fungicides

Rice Grassy Stunt Virus (RGSV)
Cause: Viral disease transmitted by brown planthopper
Symptoms: Stunted growth, pale leaves, sterile tillers
Impact: Up to 100% yield loss in severe cases
Transmission: Brown planthopper vector
Management: Resistant varieties (IR36, IR56), synchronized planting

Narrow Brown Spot
Cause: Cercospora janseana
Symptoms: Thin brown streaks on leaves
Impact: 5–40% yield loss; accelerates leaf aging
Transmission: Windborne spores; potassium deficiency
Management: Balanced fertilization, propiconazole sprays

Rice Ragged Stunt Virus (RRSV)
Cause: Virus transmitted by brown planthopper
Symptoms: Ragged/twisted leaves, stunted growth
Impact: 30–70% yield loss
Transmission: Persistent planthopper feeding
Management: Resistant varieties (IR36, IR72), Integrated Pest Management (IPM)

Rice Blast
Cause: Magnaporthe oryzae
Symptoms: Diamond-shaped lesions, neck rot ("rotten neck")
Impact: Up to 100% loss in severe cases
Transmission: Airborne spores; favored by high humidity
Management: Resistant varieties (IR64), tricyclazole fungicides

Rice False Smut
Cause: Ustilaginoidea virens
Symptoms: Green-black spore balls in place of grains
Impact: 5–40% yield loss; potential mycotoxin contamination
Transmission: Airborne spores during flowering
Management: Propiconazole during booting stage

Sheath Blight
Cause: Rhizoctonia solani
Symptoms: Collar rot, lodging, lesions on sheath
Impact: 20–70% yield loss
Transmission: Soil-borne sclerotia; irrigation water
Management: Wider plant spacing, validamycin sprays

Sheath Rot
Cause: Sarocladium oryzae
Symptoms: Rotting sheaths, poor grain fill
Impact: 20–70% yield loss
Transmission: Airborne spores, wounds
Management: Carbendazim fungicides, Trichoderma biocontrol

Stem Rot
Cause: Sclerotium oryzae
Symptoms: Black sclerotia in stem, plant lodging
Impact: 30–80% yield loss
Transmission: Soilborne sclerotia
Management: Silicon fertilization, crop rotation

Rice Tungro Virus
Cause: Combination of RTBV + RTSV; spread by green leafhopper
Symptoms: Yellow-orange leaf discoloration, stunted growth
Impact: 50–100% yield loss
Transmission: Leafhopper feeding
Management: Resistant varieties (IR36), neem-based repellents

General Rice Disease Management Strategies
Prevention: Use resistant varieties, certified seeds, balanced fertilization
Control: Remove diseased plants, apply copper and fungicide treatments
Long-Term: Practice crop rotation, maintain soil health, adopt community-based pest control
"""

st.title("Rice Disease Q&A")

question = st.text_input("Ask a question about rice diseases:")

if st.button("Submit"):
    if question.strip() != "":
        result = qa_pipeline(question=question, context=context)
        st.write("Answer:", result['answer'])
    else:
        st.warning("Please enter a question before clicking Submit.")
