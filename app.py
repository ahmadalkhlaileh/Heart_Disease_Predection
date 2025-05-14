import gradio as gr
import pandas as pd
import joblib

# تحميل النموذج
model = joblib.load("heart_model.pkl")

label_encoders = {
    'Sex': ['0', '1'],
    'ChestPainType': ['ASY', 'ATA', 'NAP', 'TA'],
    'RestingECG': ['LVH', 'Normal', 'ST'],
    'ExerciseAngina': ['N', 'Y'],
    'ST_Slope': ['Down', 'Flat', 'Up']
}

def encode(feature_name, value):
    return label_encoders[feature_name].index(value)

def predict_ar(age, sex, chest_pain, resting_bp, cholesterol,
               fasting_bs, rest_ecg, max_hr, ex_angina, oldpeak, st_slope):
    try:
        data = {
            "Age": [age],
            "Sex": [encode("Sex", sex)],
            "ChestPainType": [encode("ChestPainType", chest_pain)],
            "RestingBP": [resting_bp],
            "Cholesterol": [cholesterol],
            "FastingBS": [fasting_bs],
            "RestingECG": [encode("RestingECG", rest_ecg)],
            "MaxHR": [max_hr],
            "ExerciseAngina": [encode("ExerciseAngina", ex_angina)],
            "Oldpeak": [oldpeak],
            "ST_Slope": [encode("ST_Slope", st_slope)]
        }
        df = pd.DataFrame(data)
        prob = model.predict_proba(df)[0][1]
        result = "✅ يوجد مرض قلب" if prob >= 0.5 else "❌ لا يوجد مرض قلب"
        return f"{result}\nالنسبة: {prob*100:.2f}%"
    except Exception as e:
        return f"❗ خطأ: {str(e)}"

def predict_en(age, sex, chest_pain, resting_bp, cholesterol,
               fasting_bs, rest_ecg, max_hr, ex_angina, oldpeak, st_slope):
    try:
        data = {
            "Age": [age],
            "Sex": [encode("Sex", sex)],
            "ChestPainType": [encode("ChestPainType", chest_pain)],
            "RestingBP": [resting_bp],
            "Cholesterol": [cholesterol],
            "FastingBS": [fasting_bs],
            "RestingECG": [encode("RestingECG", rest_ecg)],
            "MaxHR": [max_hr],
            "ExerciseAngina": [encode("ExerciseAngina", ex_angina)],
            "Oldpeak": [oldpeak],
            "ST_Slope": [encode("ST_Slope", st_slope)]
        }
        df = pd.DataFrame(data)
        prob = model.predict_proba(df)[0][1]
        result = "✅ Heart Disease Detected" if prob >= 0.5 else "❌ No Heart Disease Detected"
        return f"{result}\nProbability: {prob*100:.2f}%"
    except Exception as e:
        return f"❗ Error: {str(e)}"

# الواجهة
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("<h1 style='text-align: center;'>Heart Disease Prediction / توقع مرض القلب</h1>")
    with gr.Tabs():
        with gr.TabItem("English"):
            with gr.Column():
                age = gr.Number(label="Age")
                sex = gr.Radio(["0", "1"], label="Sex (0 = Female, 1 = Male)")
                chest_pain = gr.Dropdown(["ASY", "ATA", "NAP", "TA"], label="Chest Pain Type")
                resting_bp = gr.Number(label="Resting Blood Pressure")
                cholesterol = gr.Number(label="Cholesterol")
                fasting_bs = gr.Number(label="Fasting Blood Sugar")
                rest_ecg = gr.Dropdown(["LVH", "Normal", "ST"], label="Resting ECG")
                max_hr = gr.Number(label="Max Heart Rate")
                ex_angina = gr.Radio(["N", "Y"], label="Exercise-induced Angina")
                oldpeak = gr.Number(label="Oldpeak")
                st_slope = gr.Dropdown(["Down", "Flat", "Up"], label="ST Slope")
                result = gr.Textbox(label="Result", lines=3)
                btn = gr.Button("Predict")
                clear = gr.Button("Clear")
                btn.click(
                    predict_en,
                    inputs=[age, sex, chest_pain, resting_bp, cholesterol, fasting_bs, rest_ecg, max_hr, ex_angina, oldpeak, st_slope],
                    outputs=result
                )
                clear.click(
                    lambda: [None]*11 + [""],
                    outputs=[age, sex, chest_pain, resting_bp, cholesterol, fasting_bs, rest_ecg, max_hr, ex_angina, oldpeak, st_slope, result]
                )

        with gr.TabItem("العربية"):
            with gr.Column():
                age_ar = gr.Number(label="العمر")
                sex_ar = gr.Radio(["0", "1"], label="الجنس (0 = أنثى، 1 = ذكر)")
                chest_pain_ar = gr.Dropdown(["ASY", "ATA", "NAP", "TA"], label="نوع ألم الصدر")
                resting_bp_ar = gr.Number(label="ضغط الدم أثناء الراحة")
                cholesterol_ar = gr.Number(label="الكوليسترول")
                fasting_bs_ar = gr.Number(label="سكر صائم")
                rest_ecg_ar = gr.Dropdown(["LVH", "Normal", "ST"], label="ECG أثناء الراحة")
                max_hr_ar = gr.Number(label="أقصى معدل ضربات القلب")
                ex_angina_ar = gr.Radio(["N", "Y"], label="الذبحة الصدرية أثناء التمرين")
                oldpeak_ar = gr.Number(label="Oldpeak")
                st_slope_ar = gr.Dropdown(["Down", "Flat", "Up"], label="انحدار ST")
                result_ar = gr.Textbox(label="النتيجة", lines=3)
                btn_ar = gr.Button("توقع")
                clear_ar = gr.Button("تفريغ")
                btn_ar.click(
                    predict_ar,
                    inputs=[age_ar, sex_ar, chest_pain_ar, resting_bp_ar, cholesterol_ar, fasting_bs_ar, rest_ecg_ar, max_hr_ar, ex_angina_ar, oldpeak_ar, st_slope_ar],
                    outputs=result_ar
                )
                clear_ar.click(
                    lambda: [None]*11 + [""],
                    outputs=[age_ar, sex_ar, chest_pain_ar, resting_bp_ar, cholesterol_ar, fasting_bs_ar, rest_ecg_ar, max_hr_ar, ex_angina_ar, oldpeak_ar, st_slope_ar, result_ar]
                )

demo.launch()
