from flask import Flask, render_template, request, send_file
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

app = Flask(__name__)

# Load models
reg_model = joblib.load('regression_model.pkl')
clf_model = joblib.load('classification_model.pkl')

FEATURES = [
    'Pipe_age', 'Coating', 'Operating_pressure(psi)', 'Pressure cycles',
    'Flow rate (bbl/d)', 'Temperature C', 'H2S', 'CO2', 'Water_cut', 'Previous wall loss '
]

EXTRA_FIELDS = ['Nominal Wall Thickness (mm)', 'Anomaly Length (mm)', 'Pipe Surface Area of Segment']


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/upload')
def index():
    return render_template('index.html')



@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return "No file uploaded"

        file = request.files['file']
        if file.filename == '':
            return "No file selected"

        df = pd.read_csv(file)

        required_cols = FEATURES + EXTRA_FIELDS
        for col in required_cols:
            if col not in df.columns:
                return f"Missing column: {col}"

        # Calculate Corrosion Tendency
        df['Corrosion_Tendency'] = (
            0.02 * df['H2S'] +
            0.5 * df['CO2'] +
            0.01 * df['Water_cut'] +
            0.03 * df['Pipe_age'] +
            0.005 * (df['Pressure cycles'] / 1000) +
            0.02 * df['Temperature C']
        )

        # Model input
        X = df[FEATURES].values
        X_tendency = df[['Corrosion_Tendency']].values

        predictions = []
        wall_losses = reg_model.predict(X)
        corrosion_probs = clf_model.predict_proba(X_tendency)[:, 1]

        for i, row in df.iterrows():
            predicted_wall_loss = wall_losses[i]
            corrosion_prob = corrosion_probs[i]

            wall_loss_percent = (predicted_wall_loss / row['Nominal Wall Thickness (mm)']) * 100
            corroded_area_percent = (predicted_wall_loss * row['Anomaly Length (mm)'] / row['Pipe Surface Area of Segment']) * 100

            if wall_loss_percent > 60 or corrosion_prob > 0.80:
                risk_level = "High 🔴"
            elif 30 <= wall_loss_percent <= 60 or 0.50 <= corrosion_prob <= 0.80:
                risk_level = "Medium 🟠"
            else:
                risk_level = "Low 🟢"

            predictions.append({
                "Segment ID": i + 1,
                "Predicted Wall Loss (mm)": round(predicted_wall_loss, 4),
                "Corrosion Probability": round(corrosion_prob, 4),
                "Wall Loss (%)": round(wall_loss_percent, 4),
                "Corroded Area (%)": round(corroded_area_percent, 4),
                "Risk Level": risk_level
            })

        result_df = pd.DataFrame(predictions)
        result_df.to_csv("predictions.csv", index=False)

        # Create bar graph
        plt.figure(figsize=(10, 6))
        plt.bar(result_df['Segment ID'], result_df['Predicted Wall Loss (mm)'], color='skyblue', edgecolor='black')
        for i, v in enumerate(result_df['Predicted Wall Loss (mm)']):
            plt.text(i + 1, v + 0.01, f"{v:.2f}", ha='center')

        plt.xlabel("Segment ID")
        plt.ylabel("Predicted Wall Loss (mm)")
        plt.title("Segment-wise Predicted Wall Loss")
        plt.tight_layout()
        plt.savefig("static/wall_loss_plot.png")
        plt.close()
        
        # plt.figure(figsize=(10, 6))
        # plt.bar(result_df['Segment ID'], result_df['Corroded Area (%)'], color='orange', edgecolor='black')
        # for i, v in enumerate(result_df['Corroded Area (%)']):
        #     plt.text(i + 1, v + 0.01, f"{v:.2f}", ha='center')

        # plt.xlabel("Segment ID")
        # plt.ylabel("Corroded Area (%)")
        # plt.title("Segment-wise Corroded Area (%)")
        # plt.tight_layout()
        # plt.savefig("static/Corroded Area_(%).png")
        # plt.close()


        return render_template("result.html", tables=result_df.to_html(index=False, classes='table'), graph_url="/static/wall_loss_plot.png", show_download=True)

    except Exception as e:
        return f"Something went wrong: {e}"

@app.route('/download')
def download():
    return send_file('predictions.csv', as_attachment=True)

if __name__ == '__main__':
    app.run()
