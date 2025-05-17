import streamlit as st
from transformers import pipeline

@st.cache_resource
def load_qa_model():
    return pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

qa_pipeline = load_qa_model()

context = """
Rice Leaf Diseases Summary

Bacterial Leaf Blight is caused by the bacterium Xanthomonas oryzae pv. oryzae. It spreads through wind-blown rain, irrigation water, infected seeds, and mechanical contact. Symptoms appear 1 to 2 weeks after infection, causing yellowing, wilting, and drying of leaves. It can reduce yield by up to 70%. Prevention involves using resistant varieties, proper field drainage, and avoiding over-fertilization. Farmers should remove infected plants, avoid contaminated water, and apply copper-based bactericides if needed.

Bacterial Leaf Streak is caused by Xanthomonas oryzae pv. oryzicola and spreads through rain splashes, contaminated tools, and seedborne transmission. Symptoms show within 10 to 14 days, characterized by water-soaked streaks that reduce photosynthesis, leading to moderate yield loss. Prevention includes using certified seeds, crop rotation, and resistant varieties. Infected debris should be destroyed, and overhead irrigation avoided.

Bakanae is caused by the fungus Fusarium fujikuroi. It spreads via infected seeds and contaminated water. Symptoms appear 7 to 10 days after sowing, with abnormal elongation, thin and pale seedlings that often die. Prevention requires seed treatment with fungicides before planting. Infected seedlings must be uprooted, and seeds from affected crops avoided.

Brown Spot is caused by the fungus Bipolaris oryzae and spreads through infected seeds, rain, and wind. Symptoms appear 10 to 12 days after infection as brown lesions on leaves, leading to weak plants and reduced grain quality and yield. Balanced fertilization, especially with potassium and silicon, helps prevent it. Resistant varieties and seed fungicide treatment are advised.

Grassy Stunt Virus, transmitted by the brown planthopper insect, causes stunted growth, excessive tillering, and failure to produce grain. It spreads through insect vectors, with symptoms showing 15 to 20 days after infection. Vector control using insecticides or resistant varieties is key. Farmers should monitor hopper outbreaks and promptly remove infected plants.

Narrow Brown Spot, caused by the fungus Cercospora janseana, spreads by spores via wind and rain and can be seedborne. Symptoms develop within 10 days, producing narrow brown lesions that reduce photosynthesis and yield. Prevention involves seed treatment and timely fungicide spraying. Good field drainage and removal of crop residues are recommended.

Ragged Stunt Virus, transmitted by the brown planthopper, causes ragged leaves, twisting, stunting, and poor grain filling. Symptoms appear 14 to 20 days after infection. Preventative measures include using vector-resistant varieties and avoiding continuous cropping. Insecticide-treated seeds and crop rotation are advised.

Rice Blast is caused by the fungus Magnaporthe oryzae, spreading through wind, water, and infected residues. Lesions appear 5 to 10 days post-infection, affecting leaves, necks, and nodes, leading to severe yield loss or plant death. Using blast-resistant varieties and applying fungicides during critical stages helps prevent it. Field sanitation and early control are important.

Rice False Smut, caused by Ustilaginoidea virens, spreads via airborne spores and infected seeds. It appears at the flowering stage (40 to 50 days after sowing), replacing grains with greenish fungal balls that contaminate the harvest. Timely fungicide application at the booting stage is essential. Harvesting early and cleaning equipment thoroughly help manage it.

Sheath Blight, caused by Rhizoctonia solani, is soilborne and spreads via water and plant contact. Symptoms appear 3 to 5 days after infection, causing collapsing sheaths, lodging, and reduced yield. Moderate nitrogen use, good plant spacing, and resistant varieties prevent it. Fungicides and improved air flow between plants assist control.

Sheath Rot, caused by Sarocladium oryzae, spreads through infected plant parts, rain splash, and insects. Symptoms appear at booting or heading stages, resulting in panicle emergence failure and poor grain set. Prevention includes avoiding insect damage, controlling nitrogen levels, and clean harvesting. Removing infected panicles and fungicide use are recommended.

Stem Rot, caused by Sclerotium oryzae, is soilborne, surviving in stubble. It appears from late tillering to heading stages, causing lodging, reduced grain filling, and tiller death. Crop rotation, deep plowing, and silicon fertilizer help prevent it. Burning crop residues and early fungicide application aid control.

Tungro Virus, caused by Rice Tungro Bacilliform and Spherical Viruses, is spread by the green leafhopper. Symptoms appear 10 to 14 days after infection, including yellow-orange leaf discoloration, stunting, and poor tillering. Prevention involves using resistant varieties and early vector control. Infected plants should be rogued and overlapping crops avoided.
"""

st.title("Rice Disease Q&A")

question = st.text_input("Ask a question about rice diseases:")

if question:
    result = qa_pipeline(question=question, context=context)
    st.write("Answer:", result['answer'])
