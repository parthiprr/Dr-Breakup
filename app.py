from flask import Flask, render_template, request
import joblib
import pandas as pd
import joblib

app = Flask(__name__)

# Load saved model and encoders
model = joblib.load("breakup_model.pkl")
label_encoders = joblib.load("label_encoders.pkl")
feature_columns = joblib.load("feature_columns.pkl")
categorical_cols = ['communication_frequency', 'cheating_suspected', 'future_plans', 'communication_quality']

@app.route('/')
def home():
    return render_template("form.html")

@app.route('/predict', methods=['POST'])
def predict():
    input_data = {
        'relationship_duration': int(request.form['relationship_duration']),
        'communication_frequency': request.form['communication_frequency'].lower(),
        'fights_per_week': int(request.form['fights_per_week']),
        'trust_score': int(request.form['trust_score']),
        'affection_score': int(request.form['affection_score']),
        'cheating_suspected': int(request.form['cheating_suspected']),
        'future_plans': int(request.form['future_plans']),
        'time_spent_per_week': int(request.form['time_spent_per_week']),
        'emotional_support_score': int(request.form['emotional_support_score']),
        'jealousy_level': int(request.form['jealousy_level']),
        'communication_quality': request.form['communication_quality'].lower(),
        'age_difference': int(request.form['age_difference'])
    }

    df = pd.DataFrame([input_data])

    for col in categorical_cols:
        if col in df.columns:
            val = str(df[col].iloc[0])
            if val in label_encoders[col].classes_:
                df[col] = label_encoders[col].transform([val])[0]
            else:
                df[col] = 0

    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0

    df = df[feature_columns]

    prediction = model.predict(df)[0]
    prediction_proba = model.predict_proba(df)[0]
    result = label_encoders['Breakup'].inverse_transform([prediction])[0]

    return render_template("result.html", result=result, no=round(prediction_proba[0]*100, 2), yes=round(prediction_proba[1]*100, 2))

if __name__ == '__main__':
    app.run(debug=True)
