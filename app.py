import streamlit as st
from transformers import pipeline

@st.cache_resource
def load_qa_model():
    return pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

qa_pipeline = load_qa_model()

context = """
Rice Leaf Diseases Summary

1. Bacterial Leaf Blight is a serious bacterial infection that affects rice plants, causing symptoms such as leaf wilting and yellowing. It is caused by the bacterium Xanthomonas oryzae pv. oryzae, and it significantly impacts the plant by drying out the leaves and potentially leading to yield losses as high as 70%. While the disease can affect all rice varieties, some resistant strains like IR20 and IR64 are available to help manage its spread.

The disease first appears as water-soaked lesions that gradually turn yellow. Unlike fungal spots, these lesions often have wavy margins that make them distinguishable. It can appear at any growth stage, though it tends to be most severe during the tillering stage. The disease spreads rapidly, especially under warm and humid conditions.

Conditions that favor the development of Bacterial Leaf Blight include high humidity, the presence of standing water, and the overuse of nitrogen-based fertilizers. The disease is transmitted through several means, including rain splashes, irrigation water, and contaminated tools or seeds. It can even spread from one farm to another through wind-blown rain, making containment a challenge in densely planted regions.

If left untreated, the disease can cause yield losses of up to 70%. In severe cases, it can even kill seedlings. Additionally, it affects grain quality by reducing milling recovery. Preventative strategies include cultivating resistant rice varieties like IR20 and IR64 and employing sound farming practices such as avoiding waterlogging and ensuring balanced fertilizer use.

When the disease is identified, the immediate course of action is to remove and destroy infected plants to prevent further spread. Although there is no cure, copper-based bactericides have been found to slow the progression of the infection. However, organic treatment options have not yet proven effective.

For a more sustainable and long-term approach, crop rotation with non-rice crops for one to two seasons is recommended, along with proper soil and water management practices like improved drainage. The cost of treatment is generally low, with copper sprays being the most affordable option. The most effective overall strategy combines the use of resistant varieties with good field hygiene.

2. Bacterial Leaf Streak is a bacterial disease that causes water-soaked streaks to appear on rice leaves. This infection, caused by Xanthomonas oryzae pv. oryzicola, affects the plant by reducing its ability to carry out photosynthesis, which leads to moderate yield losses. The disease weakens the leaves and hampers grain filling. While all rice varieties are susceptible, some have shown resistance, including IR24 and IR50.

Symptoms first appear as small, water-soaked streaks located between the veins of the leaf. These streaks remain narrow and, unlike those seen in Bacterial Leaf Blight, do not cause significant wilting. The disease is active from the seedling stage through to maturity and tends to spread more quickly in warm, wet conditions.

Favorable conditions for the development of Bacterial Leaf Streak include high humidity levels above 80%, temperatures ranging from 25 to 30 degrees Celsius, and extended periods of leaf wetness. It spreads through rain splashes, contaminated tools, and infected seeds. The disease can also spread between farms through wind-driven rain or shared equipment.

If not managed, the disease can lead to yield losses between 10% and 30%. While it rarely kills plants, it does weaken them, and in severe cases, it results in chalky or unfilled grains, affecting grain quality.

Preventative measures include using resistant varieties like IR24 and IR50, planting certified disease-free seeds, avoiding overhead irrigation, and rotating rice crops with non-host plants. When the disease is detected, infected plant debris should be removed and burned. Though not curable, copper-based sprays such as copper hydroxide or, where permitted, streptomycin sulfate, can help slow the spread. However, no organic treatments have proven highly effective.

For long-term control, it is recommended to rotate rice with legumes or vegetables and avoid practices like ratooning. Maintaining proper drainage is also important to reduce humidity and leaf wetness. The cost of treatment is generally low to moderate, with copper sprays averaging about $5 to $10 per acre. The most effective and affordable method of control combines the use of resistant varieties with good field sanitation. However, excessive use of copper can lead to a buildup in the soil, which may harm beneficial microbes, so it should be applied cautiously.

3. Bakanae Disease, also known as "Foolish Seedling Disease," is a fungal infection that causes abnormal elongation and often death of rice seedlings. The term "Bakanae" comes from Japanese, meaning "foolish seedling," which refers to the unnaturally tall and weak appearance of infected plants. This disease is caused by the fungus Fusarium fujikuroi. Infected rice plants grow excessively tall but remain fragile, often collapsing under their own weight. While all rice varieties are susceptible, the severity of infection can vary depending on the variety and environmental conditions.

Symptoms of Bakanae appear early, usually 7 to 10 days after sowing. Affected seedlings become pale, thin, and abnormally tall—often two to three times the normal height. Unlike plants suffering from nutrient deficiency, the stems of infected seedlings are hollow and break easily. The disease spreads rapidly in warm temperatures between 25 and 30°C and in humid conditions, making it particularly aggressive in poorly drained fields.

Bakanae is primarily seedborne, but it can also spread through contaminated soil or irrigation water. Infected seeds and shared irrigation systems are the main routes of transmission between farms. Poor drainage and high temperatures further contribute to its spread.

The impact on yield can be significant, ranging from 20% to 50% loss if left untreated, and rising to 70% in severe cases. Death of infected seedlings is common, and even those plants that survive to maturity often produce fewer filled grains, leading to poor grain quality.

Prevention is key to managing Bakanae. Some hybrid varieties, like Arize 6444, show partial resistance. Seed treatment is especially important—soaking seeds in hot water at 53 to 54°C for 10 to 15 minutes or treating with fungicides can greatly reduce infection. It’s also critical to avoid using seeds from previously infected fields and to ensure proper drainage in seedbeds and fields.

If the disease appears, immediate action is required. Infected seedlings should be uprooted and burned to prevent the spread of fungal spores. Although the disease is not curable once a plant is infected, fungicides can help prevent its spread. Seed treatment with fungicides like carbendazim, thiram, or prochloraz is effective. In some regions, benomyl can be used for soil drenching. Organic options such as bio-control agents like Trichoderma harzianum and neem seed extract have shown limited efficacy, but may offer some protection.

For long-term control, crop rotation with non-host crops such as legumes for two to three years is recommended. Soil solarization, which involves covering moist soil with plastic for four to six weeks to kill pathogens, is also effective. Managing water to avoid stagnant conditions, especially in nurseries, is essential to preventing the disease.

In terms of cost and risk management, seed treatment usually costs about $1 to $3 per kilogram of seeds, while fungicide sprays cost around $10 to $15 per acre. The most cost-effective and reliable method is combining hot water seed treatment with the use of resistant varieties. However, repeated use of certain fungicides can harm beneficial soil microbes, so it's important to rotate chemicals and monitor their effects on the soil ecosystem.

4. Brown Spot, also known as Helminthosporiosis, is a widespread fungal disease of rice that leads to the appearance of circular brown lesions on leaves and grains. Caused by the fungus Bipolaris oryzae (synonym: Cochliobolus miyabeanus), it weakens the plant by reducing its photosynthetic ability and grain development, often resulting in what is commonly referred to as “dirty rice.” Although it can infect all rice varieties, it tends to be more severe in fields with poor soil fertility, particularly potassium and silicon deficiencies.

Early signs include small, dark brown spots (1–2 mm) on the leaves. As the disease progresses, these spots enlarge to 5–10 mm, developing characteristic gray centers and brown margins, creating a “bullseye” or halo-like pattern. Infected grains show black or brown discoloration, reducing quality and market value. Unlike rice blast disease, the lesions from brown spot are more uniform in shape and lack the pointed or tapered ends. The disease can appear at any growth stage but causes the most damage during the flowering stage. It spreads moderately quickly, especially in humid conditions with prolonged leaf wetness.

Brown Spot thrives in warm, humid environments (20–30°C and >80% humidity), particularly where soil nutrients are unbalanced or deficient. The fungus spreads through infected seeds, crop debris, and spores carried by wind or rain. It can also persist in the soil for years on leftover straw or debris, enabling long-term survival. As such, contaminated seeds and windborne spores are the main drivers of farm-to-farm transmission.

Yield losses due to brown spot typically range from 10% to 50%, but in severe outbreaks that affect the grains directly, losses can reach as high as 90%. While the disease does not often kill the plant outright, heavy infections can cause leaf death, weaken the plant overall, and lead to discolored, lightweight grains with high breakage rates during milling.

Preventive measures are essential. Some rice varieties like IR36 and BRRI dhan49 show partial resistance. Key farming practices include balanced fertilization—with an emphasis on potassium and silicon—using certified disease-free seeds, and removing and burning infected crop residues after harvest to reduce sources of future infection.

If symptoms appear, fungicides should be applied immediately to contain the spread. While the disease cannot be fully cured, sprays with fungicides such as mancozeb, azoxystrobin, or propiconazole are effective at suppressing further infection. Seeds can be treated with carbendazim or thiram to prevent initial infection. Organic treatments such as applying silicon-rich rice husk ash, neem oil, or Pseudomonas biofungicides have shown limited but helpful effects in disease management.

For long-term prevention, farmers are encouraged to rotate rice with non-host crops like pulses or vegetables for one to two years and improve soil health by incorporating organic matter. Managing irrigation to avoid excessive humidity can also help reduce disease pressure.

Treatment costs are relatively manageable, with fungicide sprays typically costing $8–12 per acre per application and seed treatments priced around $0.50–1 per kilogram of seeds. The most affordable and effective strategy is a combination of potassium fertilization and seed treatment. However, overuse of fungicides may negatively affect soil microbial life, so rotating chemical types and integrating organic methods where possible is recommended.

5. Rice Grassy Stunt Virus (RGSV) is a highly destructive viral disease of rice that results in severe stunting, pale yellow-green foliage, and excessive tillering—with infected plants producing no edible grain. Caused by the Rice grassy stunt virus (RGSV) and spread exclusively by the brown planthopper (Nilaparvata lugens), this disease can lead to total crop failure in heavily infested fields. Though all varieties are vulnerable, certain cultivars like IR36, IR56, and IR72 offer partial resistance.

Early symptoms begin subtly with yellowing at the leaf tips. As the infection advances, plants remain significantly shorter—about a third of their normal height—while producing narrow, pale yellow leaves and an unusually high number of sterile tillers (sometimes over 100 per plant). The absence of panicle formation (grain heads) seals the fate of infected plants. Unlike rice tungro, another viral disease, RGSV-infected plants maintain narrow, upright leaves and do not exhibit orange discoloration.

The disease spreads rapidly when brown planthopper populations are high, particularly in warm, humid environments (25–30°C) and in regions with continuous rice cultivation. The virus is persistent within the planthopper, meaning once a hopper is infected, it can spread RGSV for life. These vectors can migrate from field to field, especially when triggered by overcrowding or pesticide disruption of natural predators.

RGSV has one of the most devastating impacts on rice yield: infected plants may survive but they produce zero grain, leading to 100% yield loss in some fields. The economic toll can be massive, especially in regions lacking early detection and coordinated pest control efforts.

Prevention relies heavily on resistant varieties and good agronomic practices. Synchronous planting among neighboring farmers can disrupt the planthopper life cycle and reduce their numbers. Avoiding staggered planting and plowing under infected stubble can limit breeding grounds for the vector.

Once plants are infected, they cannot be cured. Immediate response includes removing infected plants (roguing) and managing vector populations using selective insecticides like imidacloprid or buprofezin. Care must be taken to avoid overuse of broad-spectrum insecticides, which may harm natural predators such as spiders and mirid bugs that help control planthoppers. Organic measures like neem-based sprays and habitat conservation for beneficial insects may offer some vector suppression.

For long-term control, community-wide pest management strategies are essential. These include rotating crops, incorporating fallow periods, and removing alternate hosts like Leersia grass, which can harbor both planthoppers and the virus. Maintaining ecological balance in the field helps reduce reliance on chemical inputs.

Treatment costs vary depending on pest pressure, with insecticide applications typically costing $10–15 per acre. Fortunately, resistant seed varieties usually come at no significant additional cost. The most effective and affordable strategy remains the combination of resistant varieties and synchronized planting, paired with intelligent vector management.

6. Narrow Brown Spot, caused by the fungus Cercospora janseana, is a foliar disease of rice that presents as thin, elongated brown lesions, primarily on leaves and sometimes on panicles. While it typically doesn’t lead to devastating losses on its own, it can weaken crop health, especially in potassium-deficient fields, and when combined with other stresses, may significantly reduce grain quality and milling yield.

The disease initially appears as tiny brown streaks between leaf veins. As it progresses, these lesions elongate into 2–10 mm long and 0.5–1 mm wide brown strips, giving the characteristic narrow appearance. In advanced stages, affected leaves yellow prematurely, and panicles may develop spotted glumes. These symptoms distinguish it from similar diseases like bacterial streak (which causes wet, oozy lesions) and brown spot (which causes rounder, broader lesions).

Outbreaks typically occur in the late tillering to maturity stage, especially when conditions are humid (over 75%) and temperatures hover between 25–30°C. Imbalanced fertilization—too much nitrogen and too little potassium—creates an ideal environment for infection. The pathogen spreads via windborne spores and contaminated seeds or equipment, with crop residues acting as a primary source of inoculum.

Though narrow brown spot rarely causes plant death, it reduces photosynthesis, accelerates leaf senescence, and leads to poor grain filling, with yield losses ranging from 5–20%, and occasionally up to 40% in neglected fields. Infected grains may be visibly spotted and lighter, which lowers milling recovery and market value.

Preventive practices are crucial and include the use of balanced fertilization (especially ensuring sufficient potassium), removal of infected straw, and avoiding over-application of nitrogen. Some japonica rice varieties show partial resistance, offering an additional line of defense.

Chemical control is not always necessary, but where infections are widespread or persistent, propiconazole or azoxystrobin sprays at the booting stage can protect the crop. Seed treatment with thiram also helps reduce seedborne transmission. Organic approaches such as potassium silicate sprays and compost tea have shown limited efficacy, but may be used in integrated programs.

Long-term control requires improving soil fertility through crop rotation and residue management—either burning or deeply burying infected straw. This helps prevent the disease from overwintering in the field and re-infecting the next crop.

Treatment costs include $10–15/acre for fungicides and $20–30/acre for potassium fertilizers. The most cost-effective strategy remains preventive fertilization and residue handling. However, overuse of fungicides or potassium can disrupt soil microbiota or alter soil pH, so careful management is key.


7. Rice Ragged Stunt Virus (RRSV) is a viral disease transmitted exclusively by the brown planthopper (Nilaparvata lugens), making it a major concern in areas with recurring planthopper infestations. The virus causes severe stunting, twisted and ragged leaves, and significantly reduces tillering and panicle development, with potential yield losses between 30–70%, especially when infection occurs early in the crop’s life.

The disease begins with slight twisting and dark green streaks on young leaves. Within 10–20 days, plants develop distinctly ragged leaf edges, twisted upright leaves, and significant height reduction—often growing 30–50% shorter than healthy plants. Panicle emergence is poor, and when panicles do appear, they frequently produce chalky, unfilled grains. This differentiates RRSV from other viral diseases like tungro (which causes yellowing) or grassy stunt (which leads to excessive tillering).

RRSV thrives in warm (25–30°C), humid environments, especially in fields practicing continuous rice cropping. Unlike many diseases, it does not spread via seeds, soil, or tools—only through persistent transmission by infected planthoppers, which can travel between farms.

To limit losses, it's crucial to implement preventive strategies:

Use resistant varieties such as IR36, IR56, and IR72.

Synchronized planting across farms reduces planthopper breeding cycles.

Remove ratoon crops (leftover stubble regrowth) and grassy weeds that serve as virus reservoirs.


Control involves an integrated pest management (IPM) approach. Since the disease is incurable, early removal of infected plants and targeted spraying of insecticides (e.g., buprofezin, pymetrozine, or imidacloprid) to control young planthoppers is essential. Avoid pyrethroids, which often trigger resistance in planthopper populations. Neem oil and biological agents (like spiders and mirid bugs) can support organic control efforts.

Long-term solutions focus on crop rotation, area-wide community pest control, and field drainage to disrupt vector breeding. Typical treatment costs range from $10–20/acre for insecticides, while using resistant seeds adds little to no extra cost.

The most affordable and effective strategy remains the combination of resistant varieties and coordinated planting. While chemical controls can harm pollinators, RRSV poses no soil persistence risk, since it depends entirely on living vectors.


8. Rice blast, caused by the fungus Magnaporthe oryzae, is recognized as the most destructive fungal disease affecting rice worldwide. It produces lesions on leaves, stems, and panicles, and under favorable conditions, it can lead to complete crop loss. The disease impacts the plant by reducing photosynthesis through leaf blast and by preventing grain filling in the neck blast stage, which results in empty panicles. Although all rice varieties are vulnerable, resistance levels vary depending on the strain.

Early symptoms of rice blast include small, diamond-shaped white or gray lesions bordered by dark edges. As the disease advances, leaf blast manifests as expanding lesions measuring one to two centimeters with gray centers. Node blast causes blackening of the nodes that can lead to stem collapse, while neck blast results in a condition known as "rotten neck," where the neck breaks and panicles fall off. The lesions caused by rice blast can be distinguished from brown spot by their angular shape with tapered ends and from bacterial blight because they do not exude bacteria. The disease generally appears during the seedling to tillering stages for leaf blast and during flowering for neck blast. Under ideal conditions, symptoms can develop rapidly within three to seven days.

Rice blast thrives in environments with high humidity above 90 percent and prolonged leaf wetness. It prefers cool nights ranging from 20 to 24 degrees Celsius coupled with warm daytime temperatures of 28 to 30 degrees Celsius. Excessive nitrogen fertilization also encourages its development. The fungus spreads through airborne spores that can travel up to one kilometer and can survive in infected straw and seeds. This makes farm-to-farm transmission via windborne spores a common occurrence.

The impact on yield is significant. Leaf blast can cause yield losses of 10 to 50 percent, while neck blast can lead to losses between 50 and 100 percent by killing entire panicles. Grain quality also deteriorates, with affected grains appearing chalky and lightweight, increasing the amount of broken rice during milling.

Preventive measures focus on using resistant rice varieties such as IR64 and Pusa Basmati 1121, which have blast resistance. Good farming practices include avoiding late nitrogen applications, maintaining a flood depth of five to ten centimeters in fields to reduce spore dispersal, and destroying crop residues after harvest to minimize sources of inoculum.

Control strategies require prompt action once symptoms appear. Fungicides should be applied at the first sign of infection, and temporarily draining fields can help reduce humidity levels. While the disease cannot be cured, fungicides protect new growth. Chemical treatments like tricyclazole are most effective for prevention, while combinations such as azoxystrobin and difenoconazole have curative effects. Organic options include silicon amendments, which strengthen plant cell walls, and the biofungicide Bacillus subtilis. Removing infected plants is advised, though this is often impractical during severe outbreaks.

For long-term management, crop rotation with legumes for one to two years helps break the disease cycle, and soil solarization can reduce fungal spores in the soil. Monitoring tools such as blast forecasting applications, like "BlastLite," aid farmers in anticipating outbreaks.

Regarding costs and risks, typical fungicide treatments range from $15 to $25 per acre per spray, and resistant seeds cost approximately 5 to 10 percent more. The most affordable and effective approach combines resistant varieties with balanced nitrogen application. However, it is important to note that fungicides may negatively impact beneficial mycorrhizal fungi, although they do not accumulate in the soil since they require living tissue to persist.


9. Rice false smut, caused by the fungus Ustilaginoidea virens, is a disease that transforms rice grains into greenish-black spore balls. This fungal infection reduces both grain quality and yield, with contaminated grains potentially containing harmful mycotoxins. The disease replaces individual grains with masses of fungal spores, affecting all rice varieties, although some hybrids show partial resistance.

Early symptoms of rice false smut include the appearance of small white spore sacs emerging from the florets. As the disease progresses, these develop into bright orange or yellow velvety spore balls measuring one to two centimeters in diameter, which later turn greenish-black as the spores mature. Typically, about 5 to 20 percent of panicles in infected fields are affected. The disease can be distinguished from true smut because its spores remain enclosed within a membrane and is also different from kernel smut caused by Tilletia barclayana. Rice false smut appears exclusively during the flowering and grain filling stages and spreads at a moderate rate, requiring specific humidity conditions.

The disease favors prolonged rainfall during flowering periods, high humidity above 90 percent, and temperatures between 25 and 30 degrees Celsius. Excessive nitrogen fertilization further encourages its development. The fungus spreads through airborne spores originating from soil or infected crop residues and can survive in soil for one to two years. Windborne spores facilitate transmission from farm to farm.

Yield losses from rice false smut range from 5 to 20 percent due to direct grain replacement and can reach up to 40 percent if severe panicle infections occur. Although the disease does not kill the plant, infected panicles are unmarketable. The grains affected may contain ustiloxins, which are toxic mycotoxins, and contamination can cause discoloration in processed rice products.

Preventive measures include planting resistant varieties, with some japonica types demonstrating tolerance. Farmers are advised to avoid late nitrogen applications and to adjust planting schedules to avoid rainy flowering periods. Deep plowing helps by burying infected residues to reduce inoculum.

Control requires immediate action such as removing and burning heavily infected panicles and applying fungicides at the booting stage. There is no cure once the infection is established, making prevention the key strategy. Chemical treatments effective against rice false smut include propiconazole or tebuconazole applied at booting, while copper-based fungicides are less effective. Organic options such as applications of Bacillus subtilis and neem oil offer limited efficacy. Partial removal of infected plants may be feasible during small outbreaks.

Long-term strategies to prevent re-occurrence involve rotating crops for two years with non-cereal plants and applying biological soil amendments like Trichoderma species. Monitoring by scouting fields during flowering in high-risk areas is also recommended.

Regarding cost and risk management, fungicide treatments typically cost between $12 and $18 per acre per application. Yield losses can translate to $50 to $200 per acre depending on severity. The most affordable and effective approach is timely fungicide application at the booting stage. Some fungicides may affect earthworm populations, but there is no persistence of the chemicals in soil beyond two years.

10. Sheath blight, caused by the fungus Rhizoctonia solani AG1-IA, is a highly destructive disease that affects rice by causing collar rot and sheath infections. This disease reduces the photosynthetic area of the plant and often leads to lodging, which can cause significant yield losses. The fungal infection produces lesions on leaf sheaths and leaves, triggers premature senescence, and results in unfilled grains. It poses a threat to all rice varieties, as there are no completely resistant strains known.

In the early stages, sheath blight manifests as oval, water-soaked lesions on the lower leaf sheaths near the waterline, often appearing as greenish-gray spots with irregular margins. As the disease advances, these lesions become large and irregular, featuring white centers surrounded by brown borders. A characteristic symptom is "collar rot" at the base of the panicle, which causes the plant to lodge, and entire tillers may die. Sheath blight lesions can be differentiated from those of blast disease by their distinct zonation and from bacterial blight by the absence of bacterial ooze. The disease is most severe from tillering to heading stages and spreads very rapidly under optimal conditions, potentially infecting entire fields within 7 to 10 days.

Favorable conditions for sheath blight include high humidity above 85%, temperatures between 28 and 32 degrees Celsius, dense plant canopies, and excessive nitrogen fertilization. The fungus spreads via sclerotia present in soil or plant debris, mycelial growth through water, and contamination from farm equipment. Transmission from farm to farm is common through infected soil, water, or tools.

Yield losses due to sheath blight range from 20 to 50 percent under moderate infection and can reach up to 70 percent in severe cases. Partial death of tillers is common, and the disease also negatively impacts grain quality by increasing chalkiness and lowering milling recovery.

To prevent sheath blight, some Jasmine rice varieties exhibit tolerance. Effective farming practices include wider plant spacing (20 by 20 centimeters), controlled nitrogen application, and maintaining shallow water depths of about 5 centimeters. Immediate control measures include temporarily draining fields and applying fungicides. Although the disease is not curable once established, its spread can be managed. Chemical treatments such as validamycin are the most effective, while combinations like azoxystrobin and propiconazole are also used. Organic options include biocontrol agents like Trichoderma harzianum and Pseudomonas fluorescens. Removing infected plants is generally impractical in field conditions.

Long-term strategies focus on preventing re-occurrence through deep plowing to bury sclerotia, crop rotation with pulses, and soil solarization. Regular scouting of fields from tillering onwards is essential for monitoring disease presence.

Regarding costs, fungicide treatments typically range from $15 to $25 per acre, while biocontrol methods cost around $5 to $10 per acre. The most affordable and effective management approach involves integrating moderate nitrogen fertilization with other control measures. Fungicides may impact beneficial soil microbes, and the pathogen can survive in soil for several years.



11. Sheath Rot is a fungal disease caused by Sarocladium oryzae that leads to rotting of the leaf sheath and discoloration of grains. This disease significantly reduces grain filling and quality, resulting in substantial economic losses. The fungus affects the plant by causing rotting of the leaf sheaths, poor grain development, and premature aging of the plant. Although all rice varieties are susceptible to Sheath Rot, the severity of the disease can vary.

Early signs of Sheath Rot include discolored lesions on the upper leaf sheaths and purple-brown streaks near the panicle base. As the disease progresses, white powdery fungal growth may develop inside the sheaths, and the grains become discolored and shriveled, with some panicles failing to emerge altogether. Unlike sheath blight, the lesions caused by Sheath Rot are more localized, and no sclerotia formation occurs, differentiating it from infections caused by Rhizoctonia. The disease primarily affects plants from the booting stage to maturity and spreads at a moderate rate, favored by high humidity conditions.

The conditions that promote Sheath Rot include high humidity above 80%, temperatures between 25 and 30 degrees Celsius, and the presence of wounds caused by insects or mechanical damage. The disease spreads through airborne spores, infected seeds, and crop residues, allowing transmission from farm to farm via infected seeds and wind.

In terms of impact, Sheath Rot can cause yield losses ranging from 20 to 40 percent under moderate infection, and up to 70 percent in severe cases. While plant death is rare, panicle abortion is common, which further reduces yield. The quality of grains is also affected, as infected grains become discolored, lightweight, and there is a higher percentage of broken grains during harvest.

Prevention strategies for Sheath Rot involve using resistant varieties, particularly some tolerant japonica types. Farmers are advised to apply balanced nitrogen fertilization, ensure proper field drainage, and remove crop residues to minimize disease occurrence.

For control and treatment, fungicides such as carbendazim or propiconazole can be applied at the booting stage, alongside seed treatment with thiram. Organic options include sprays made from garlic extract and biocontrol agents based on Trichoderma. In cases of heavy infection, it is recommended to remove infected plants to reduce the spread. Although the disease is not curable once established, these measures can help control its spread.

Long-term strategies focus on preventing re-occurrence through crop rotation lasting two years, soil solarization to kill pathogens, and the use of clean seeds. Regular inspection starting from the booting stage is important for early detection and timely management.

Typical treatment costs for fungicides range from ten to fifteen dollars per acre, while cultural practices incur minimal expenses. The most affordable and effective approach combines resistant varieties with integrated management techniques. Some fungicides used may affect soil biota, but no long-term soil contamination is expected from these treatments.


12. Stem Rot is a fungal disease caused by Sclerotium oryzae that results in stem decay and lodging, which significantly reduces yield due to poor grain filling and difficulties during harvest. The disease weakens the stems, leading to lodging, reduces nutrient transport to the panicles, and promotes premature plant death. While all rice varieties are susceptible, the severity of the disease largely depends on the cultural practices employed.

Early symptoms of Stem Rot include small black lesions near the water line on the stems and yellowing of the lower leaves. As the disease progresses, distinct black sclerotia—resembling mustard seeds—develop inside the stems, which become hollow and easily breakable. Severe lodging often occurs during the grain filling stage. Stem Rot can be differentiated from sheath blight by the presence of these hard sclerotia, and it does not produce the foul odor typically associated with bacterial infections. The disease usually appears from late tillering up to maturity and spreads slowly at first but accelerates during the reproductive stages.

Favorable conditions for Stem Rot include high humidity above 85%, temperatures between 25 and 30 degrees Celsius, standing water in fields, and soils deficient in silica. The fungal sclerotia can survive in soil for two to three years and spread through irrigation water and infected crop residues. Farm-to-farm transmission can occur via contaminated equipment or flood water.

Yield losses due to Stem Rot range from 30 to 50 percent due to lodging and poor grain fill, and losses can reach up to 80 percent if the infection occurs early in the crop cycle. Severely infected tillers often die, and grain quality suffers with increased chalkiness and a higher proportion of unfilled grains.

Prevention measures include the use of varieties with limited resistance, with some indica types showing better tolerance. Cultural practices such as silicon fertilization, particularly using rice hull ash, deep plowing to bury sclerotia, and avoiding water stagnation are recommended to reduce disease incidence.

Control involves draining fields for three to five days and removing lodged plants to reduce inoculum. Although Stem Rot is not curable once established, fungicides can protect healthy tissues. Chemical treatments include carbendazim combined with mancozeb applied at the tillering stage, and validamycin for early infections. Organic options such as soil application of Trichoderma viride and neem cake amendment also help manage the disease. Removing infected plants is essential to reduce the number of sclerotia in the field.

Long-term strategies focus on preventing recurrence through a three-year crop rotation with upland crops, flooding fields for three weeks to drown sclerotia, and burning infected residues. Regular monitoring of the lower stems weekly after tillering is important for early detection.

Typical treatment costs for chemical control range from $20 to $30 per acre, while organic methods cost between $5 and $10 per acre. The most affordable and effective method combines silicon fertilization with residue burning. Fungicide use may affect beneficial soil organisms like earthworms, and although sclerotia persist in the soil, they do not infect non-cereal crops.

13. Rice Tungro Virus is a highly destructive viral disease that causes stunting and leaf discoloration in rice plants. The term "Tungro" is derived from Filipino, meaning "degenerated growth," which accurately describes the severe yield losses caused by this disease. It is caused by the co-infection of two viruses: Rice Tungro Bacilliform Virus (RTBV) and Rice Tungro Spherical Virus (RTSV). Infected plants suffer from a 30 to 50 percent reduction in height, show yellow-orange leaf discoloration, and experience reduced tillering and poor panicle formation. Although all rice varieties are susceptible, some resistant types are available.

Early symptoms of the disease include the yellowing of leaf tips and mild stunting. As it progresses, the older leaves turn a distinct yellow-orange, and the plants become severely stunted, appearing bunched together. Flowering is often delayed, and panicles may be small and sterile. The disease can be differentiated from nutrient deficiency by its patchy distribution across fields and from grassy stunt disease by its deeper yellow-orange coloring instead of pale green. Tungro can infect rice at any growth stage but is most damaging when infection occurs before panicle initiation. The disease spreads rapidly, especially in areas with high populations of the green leafhopper.

The primary cause of the disease is the presence of high populations of the green leafhopper (Nephotettix virescens), particularly in regions practicing continuous rice cropping under warm temperatures between 25 and 30 degrees Celsius. The virus is transmitted solely through persistent feeding by green leafhoppers; it is not spread through seeds, soil, or mechanical means. Migration of infected leafhoppers can cause the disease to spread from one farm to another.

Yield losses due to Rice Tungro Virus can reach 50 to 100 percent if the plants are infected before panicle initiation, and between 10 and 30 percent if infection occurs later in the plant’s development. While plant death is rare, infected plants often produce no grain at all. The resulting grains are often chalky, unfilled, and have a higher likelihood of breaking during milling.

Preventing the disease involves the use of resistant varieties such as IR36, IR50, IR54, and IR72, along with locally adapted resistant strains in endemic areas. Important cultural practices include synchronized planting across large areas, prompt removal (roguing) of infected plants, and avoiding staggered planting schedules to prevent continuous exposure to the virus vector.

To control the disease, farmers must immediately remove infected plants and apply insecticides targeting the green leafhopper. There is no cure for infected plants, making prevention critical. Effective insecticides include imidacloprid and thiamethoxam, although overuse should be avoided. Organic alternatives include neem-based repellents and biological control agents like spiders and mirid bugs. Removing infected plants remains essential to minimize virus sources.

For long-term management, strategies include rotating rice with non-host crops, implementing community-wide pest management plans, and using fallow periods to break the vector’s life cycle. Managing habitats by removing alternate host plants and conserving natural predators also helps reduce leafhopper populations.

Typical treatment costs range from $10 to $20 per acre for insecticides, while resistant varieties come at little to no additional cost. The most affordable and effective control method is combining resistant varieties with synchronized planting. Insecticides may impact pollinators, but the virus does not persist in the soil, making it safe for future cropping cycles once managed properly.
"""



st.title("Rice Disease Q&A")

question = st.text_input("Ask a question about rice diseases:")

if question:
    result = qa_pipeline(question=question, context=context)
    st.write("Answer:", result['answer'])
