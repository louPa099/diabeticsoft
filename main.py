from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Cargar el modelo
model = joblib.load('stacking_clf.pkl')

@app.route("/")
def hello_world():
    # proccesing code
    return "Hello, World!"  # Esta línea debe estar indentada

@app.route("/predict", methods=['GET', 'POST'])
def predict():
    if request.method == "POST":
        data = request.json
        if data is None:
            return jsonify({"error": "No se proporcionaron datos"}), 400

        try:
            # Extraer los inputs del JSON
            imc = float(data['IMC'])
            glucosa = float(data['Glucosasangre'])
            colesterol = float(data['Colesterol'])
            trigliceridos = float(data['Trigliceridos'])
            hba1c = float(data['HbA1c'])

            # Crear un array con los datos de entrada
            input_data = np.array([[imc, glucosa, colesterol, trigliceridos, hba1c]])

            # Hacer la predicción usando el modelo
            prediction = model.predict(input_data)

            # Obtener las probabilidades de predicción
            probabilities = model.predict_proba(input_data)

            # Mapear las clases a sus nombres
            class_names = ["Alterado", "Diabetes Controlada", "Normal", "Muy Alterado"]
            
            # Convertir la predicción a un índice entero si es necesario
            if isinstance(prediction[0], str):
                predicted_class = prediction[0]
            else:
                predicted_class = class_names[int(prediction[0])]

            # Interpretar los resultados
            interpretation = interpret_results(predicted_class, probabilities[0])

            # Devolver la predicción y las probabilidades junto con los datos de entrada
            return jsonify({
                "predicted_class": predicted_class,
                "probabilities": {class_names[i]: float(prob) for i, prob in enumerate(probabilities[0])},
                "input_data": {
                    "IMC": imc,
                    "Glucosasangre": glucosa,
                    "Colesterol": colesterol,
                    "Trigliceridos": trigliceridos,
                    "HbA1c": hba1c
                },
                "interpretation": interpretation
            })

        except KeyError as e:
            return jsonify({"error": f"Falta el campo {str(e)}"}), 400
        except ValueError as e:
            return jsonify({"error": f"Error en el formato de los datos: {str(e)}"}), 400

    return jsonify({"error": "Método no permitido"}), 405

def interpret_results(predicted_class, probabilities):
    class_names = ["Alterado", "Diabetes Controlada", "Normal", "Muy Alterado"]
    prob_dict = {class_names[i]: prob for i, prob in enumerate(probabilities)}
    
    main_message = f"El resultado principal es: {predicted_class}. "
    
    if predicted_class == "Normal":
        message = main_message + "Sus niveles están dentro del rango normal. Mantenga sus hábitos saludables."
    elif predicted_class == "Alterado":
        message = main_message + "Algunos de sus niveles están fuera del rango normal. Se recomienda consultar con un médico para una evaluación más detallada."
    elif predicted_class == "Diabetes Controlada":
        message = main_message + "Sus niveles sugieren que podría tener diabetes, pero está bajo control. Continúe con su tratamiento actual y consulte regularmente con su médico."
    elif predicted_class == "Muy Alterado":
        message = main_message + "Sus niveles están significativamente alterados. Es crucial que consulte con un médico lo antes posible para una evaluación completa y posible ajuste de tratamiento."
    
    # Añadir información sobre la certeza de la predicción
    max_prob = max(prob_dict.values())
    if max_prob > 0.8:
        certainty = "El modelo tiene un alto grado de certeza en esta predicción."
    elif max_prob > 0.6:
        certainty = "El modelo tiene un grado moderado de certeza en esta predicción."
    else:
        certainty = "El modelo tiene un bajo grado de certeza en esta predicción. Se recomienda una evaluación adicional."
    
    # Añadir recomendaciones generales
    recommendations = "\n\nRecomendaciones generales:\n"
    recommendations += "- Mantenga una dieta equilibrada y rica en fibra.\n"
    recommendations += "- Realice ejercicio regularmente.\n"
    recommendations += "- Monitoree sus niveles de glucosa según las indicaciones de su médico.\n"
    recommendations += "- Asista a chequeos médicos regulares."

    return message + " " + certainty + recommendations

if __name__ == "__main__":
    print("Servidor Flask iniciado. Escuchando en http://127.0.0.1:5000/")
    app.run(debug=True)
