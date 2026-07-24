
import re
 
import pdfplumber
 
 
def extract_text_from_pdf(uploaded_file):
    
    text = ""
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text
 
 
def _search(patterns, text, group=1):
   
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(group)
    return None
 
 
def extract_fields_from_report(uploaded_file):
    
 
    text = extract_text_from_pdf(uploaded_file)
 
    extracted = {}
 
 
    age_raw = _search([r"age[:\s]+(\d{1,3})"], text)
    if age_raw:
        age_years = int(age_raw)
        
        bracket = min(13, max(1, (age_years - 18) // 5 + 1)) if age_years >= 18 else 1
        extracted["Age"] = bracket
 
   
    sex_raw = _search([r"sex[:\s]+(male|female)", r"gender[:\s]+(male|female)"], text)
    if sex_raw:
        extracted["Sex"] = 1 if sex_raw.lower() == "male" else 0
 
    
    bmi_raw = _search([r"bmi[:\s]+(\d{1,2}\.?\d?)"], text)
    if bmi_raw:
        extracted["BMI"] = float(bmi_raw)
 
   
    bp_raw = _search([r"blood pressure[:\s]+(\d{2,3})\s*/\s*(\d{2,3})"], text, group=0)
    if bp_raw:
        nums = re.findall(r"\d{2,3}", bp_raw)
        if len(nums) == 2:
            systolic, diastolic = int(nums[0]), int(nums[1])
            extracted["HighBP"] = 1 if (systolic >= 130 or diastolic >= 80) else 0
 
   
    chol_raw = _search([r"(?:total )?cholesterol[:\s]+(\d{2,3})"], text)
    if chol_raw:
        extracted["HighChol"] = 1 if int(chol_raw) >= 200 else 0
 
    # ---- Smoking status -> Smoker ----
    if re.search(r"smoking status[:\s]+(current|smoker|yes)", text, re.IGNORECASE):
        extracted["Smoker"] = 1
    elif re.search(r"smoking status[:\s]+(never|non[- ]?smoker|no)", text, re.IGNORECASE):
        extracted["Smoker"] = 0
 
   
    if re.search(r"physical activity[:\s]+(yes|active|regular)", text, re.IGNORECASE):
        extracted["PhysActivity"] = 1
    elif re.search(r"physical activity[:\s]+(no|sedentary|none)", text, re.IGNORECASE):
        extracted["PhysActivity"] = 0
 
    
    if re.search(r"heart (disease|attack)[:\s]*(yes|positive|history of|present)", text, re.IGNORECASE) \
            or re.search(r"history of (heart disease|myocardial infarction|heart attack)", text, re.IGNORECASE):
        extracted["HeartDiseaseorAttack"] = 1
    elif re.search(r"heart (disease|attack)[:\s]*(no|none|negative|not reported|denied)", text, re.IGNORECASE):
        extracted["HeartDiseaseorAttack"] = 0
 
    if re.search(r"stroke[:\s]*(yes|positive|history of|present)", text, re.IGNORECASE):
        extracted["Stroke"] = 1
    elif re.search(r"stroke[:\s]*(no|none|negative|not reported|denied)", text, re.IGNORECASE):
        extracted["Stroke"] = 0
 
    all_fields = [
        "HighBP", "HighChol", "CholCheck", "BMI", "Smoker", "Stroke",
        "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies",
        "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "GenHlth",
        "MentHlth", "PhysHlth", "DiffWalk", "Sex", "Age", "Education", "Income"
    ]
 
    found_fields = [f for f in all_fields if f in extracted]
    missing_fields = [f for f in all_fields if f not in extracted]
 
    return extracted, found_fields, missing_fields
