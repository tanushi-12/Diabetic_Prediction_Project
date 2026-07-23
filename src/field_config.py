
# Human-readable options for the Streamlit manual-entry form, mapped to
# the numeric codes the CDC BRFSS2015 dataset (and therefore the trained
# models) actually expect. Keys are feature column names; values are
# {numeric_code: display_label}.
 
BINARY_YES_NO = {0: "No", 1: "Yes"}
 
FIELD_OPTIONS = {
    "HighBP": BINARY_YES_NO,
    "HighChol": BINARY_YES_NO,
    "CholCheck": BINARY_YES_NO,
    "Smoker": BINARY_YES_NO,
    "Stroke": BINARY_YES_NO,
    "HeartDiseaseorAttack": BINARY_YES_NO,
    "PhysActivity": BINARY_YES_NO,
    "Fruits": BINARY_YES_NO,
    "Veggies": BINARY_YES_NO,
    "HvyAlcoholConsump": BINARY_YES_NO,
    "AnyHealthcare": BINARY_YES_NO,
    "NoDocbcCost": BINARY_YES_NO,
    "DiffWalk": BINARY_YES_NO,
    "Sex": {0: "Female", 1: "Male"},
 
    "GenHlth": {
        1: "Excellent",
        2: "Very good",
        3: "Good",
        4: "Fair",
        5: "Poor",
    },
 
    "Age": {
        1: "18-24",
        2: "25-29",
        3: "30-34",
        4: "35-39",
        5: "40-44",
        6: "45-49",
        7: "50-54",
        8: "55-59",
        9: "60-64",
        10: "65-69",
        11: "70-74",
        12: "75-79",
        13: "80 or older",
    },
 
    "Education": {
        1: "Never attended school / kindergarten only",
        2: "Elementary (grades 1-8)",
        3: "Some high school (grades 9-11)",
        4: "High school graduate / GED",
        5: "Some college or technical school",
        6: "College graduate",
    },
 
    "Income": {
        1: "Less than $10,000",
        2: "$10,000 to less than $15,000",
        3: "$15,000 to less than $20,000",
        4: "$20,000 to less than $25,000",
        5: "$25,000 to less than $35,000",
        6: "$35,000 to less than $50,000",
        7: "$50,000 to less than $75,000",
        8: "$75,000 or more",
    },
}
 
# Extra context shown as a hover tooltip next to the field label,
# so the dropdown options themselves stay a clean "Yes"/"No".
FIELD_HELP = {
    "CholCheck": "Cholesterol check in the past 5 years",
    "Smoker": "Smoked at least 100 cigarettes in your lifetime",
    "PhysActivity": "Physical activity in past 30 days, excluding job",
    "Fruits": "Consumes fruit 1 or more times per day",
    "Veggies": "Consumes vegetables 1 or more times per day",
    "HvyAlcoholConsump": "Heavy drinker (14+ drinks/week for men, 7+ for women)",
    "AnyHealthcare": "Has any kind of healthcare coverage",
    "NoDocbcCost": "Couldn't see a doctor due to cost in the past 12 months",
    "DiffWalk": "Serious difficulty walking or climbing stairs",
    "HeartDiseaseorAttack": "History of coronary heart disease or heart attack",
}
 
# Continuous / count fields -> (min, max, default, step, help_text)
CONTINUOUS_FIELDS = {
    "BMI": (10.0, 70.0, 27.0, 0.5, "Body Mass Index"),
    "MentHlth": (0, 30, 0, 1, "Days of poor mental health in past 30 days"),
    "PhysHlth": (0, 30, 0, 1, "Days of poor physical health in past 30 days"),
}
 