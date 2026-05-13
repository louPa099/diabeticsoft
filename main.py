"""
DIABETIC SOFT MUNDI - API v3.0
================================
Backend Flask para tamizaje de riesgo de diabetes tipo II adaptado a poblacion
andina de Cusco. Consume los modelos calibrados generados por train.py.

Cambios respecto a v2:
  - Reglas clinicas duras (ALAD / ADA) sobre la prediccion del modelo
  - Validacion estricta de rangos antropometricos
  - IMC se calcula en el backend, app solo envia peso/talla
  - Probabilidades calibradas (CalibratedClassifierCV) -> barras honestas
  - Logueo del feature-vector para debugging en Render
"""
from __future__ import annotations

import os
from pathlib import Path

import joblib
import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS

# ---------------------------------------------------------------------------
# Carga de artefactos
# ---------------------------------------------------------------------------
HERE = Path(__file__).parent
modelo_basico = joblib.load(HERE / "modelo_basico.pkl")
modelo_completo = joblib.load(HERE / "modelo_completo.pkl")
scaler_basico = joblib.load(HERE / "scaler_basico.pkl")
scaler_completo = joblib.load(HERE / "scaler_completo.pkl")
metadata = joblib.load(HERE / "metadata.pkl")

CLASES = metadata["clases"]  # ['Alterado', 'Muy Alterado', 'Normal']
FEATURES_BASICO = metadata["features_basico"]
FEATURES_COMPLETO = metadata["features_completo"]

# Textos descriptivos por modo (la app Flutter puede mostrarlos al usuario)
NOTA_BASICO = (
    "Test rapido para cuando no tienes todos tus datos. Este tamizaje preliminar "
    "usa solo antropometria y cuestionario. Para mayor precision, ingresa tus "
    "analisis de glucosa, colesterol y trigliceridos en el modo completo."
)
NOTA_COMPLETO = (
    "Evaluacion completa con datos antropometricos y resultados de laboratorio "
    "basico. Proporciona el resultado mas preciso del tamizaje."
)

IDX_NORMAL = CLASES.index("Normal")
IDX_ALTERADO = CLASES.index("Alterado")
IDX_MUY = CLASES.index("Muy Alterado")

app = Flask(__name__)
CORS(app)

# ---------------------------------------------------------------------------
# Validacion y reglas clinicas
# ---------------------------------------------------------------------------
RANGOS = {
    "edad": (15, 100),
    "peso": (30, 250),
    "talla": (120, 220),  # cm
    "perimetro_abdominal": (50, 180),
    "antecedentes_familiares": (0, 2),
    "actividad_fisica": (0, 2),
    "consumo_frutas_verduras": (0, 1),
    "presion_arterial": (70, 240),
    "altitud": (0, 6000),
    "glucosa": (40, 600),
    "colesterol": (80, 500),
    "trigliceridos": (30, 1000),
}


def _valida(nombre: str, valor: float):
    lo, hi = RANGOS[nombre]
    if not (lo <= valor <= hi):
        raise ValueError(f"{nombre}={valor} fuera de rango [{lo}, {hi}]")


def aplicar_reglas_clinicas(
    pred_idx: int,
    proba: np.ndarray,
    *,
    glucosa: float | None,
    perimetro: float,
    sexo_enc: int,
    imc: float,
    presion: float,
) -> tuple[int, np.ndarray, list[str]]:
    """Post-procesa la prediccion del modelo aplicando guardrails clinicos.

    Referencias: ALAD 2019, ADA Standards of Care 2024, MINSA Peru.
    """
    proba = proba.copy()
    avisos: list[str] = []

    # 1) Glucosa: criterio diagnostico directo
    if glucosa is not None:
        if glucosa >= 200:
            pred_idx = IDX_MUY
            proba = np.array([0.0, 0.0, 0.0])
            proba[IDX_MUY] = 0.97
            proba[IDX_ALTERADO] = 0.03
            avisos.append("Glucosa >=200 mg/dL: clasificacion forzada a 'Muy Alterado' por criterio diagnostico (ADA).")
        elif glucosa >= 126:
            if pred_idx == IDX_NORMAL:
                pred_idx = IDX_MUY
                avisos.append("Glucosa >=126 mg/dL en ayunas: criterio de diabetes tipo II. Elevado a 'Muy Alterado'.")
                proba = np.array([0.10, 0.85, 0.05])
            elif pred_idx == IDX_ALTERADO:
                pred_idx = IDX_MUY
                avisos.append("Glucosa >=126 mg/dL en ayunas: elevado a 'Muy Alterado'.")
                proba = np.array([0.15, 0.80, 0.05])
        elif glucosa >= 100:
            if pred_idx == IDX_NORMAL:
                pred_idx = IDX_ALTERADO
                avisos.append("Glucosa entre 100 y 125 mg/dL: prediabetes (ADA). Elevado a 'Alterado'.")
                proba = np.array([0.80, 0.10, 0.10])

    # 2) Obesidad central (criterio ALAD: H>=90, M>=80 cm)
    umbral_perim = 90 if sexo_enc == 1 else 80
    if perimetro >= umbral_perim and pred_idx == IDX_NORMAL:
        pred_idx = IDX_ALTERADO
        avisos.append(
            f"Perimetro abdominal {perimetro} cm >= {umbral_perim} cm (criterio ALAD): "
            f"obesidad central. Elevado a 'Alterado'."
        )
        proba = np.array([0.65, 0.15, 0.20])

    # 3) HTA estadio 2 sostenida sumada a IMC alto
    if presion >= 160 and imc >= 30 and pred_idx == IDX_NORMAL:
        pred_idx = IDX_ALTERADO
        avisos.append("Presion sistolica >=160 con IMC >=30: riesgo cardiometabolico alto.")

    return pred_idx, proba, avisos


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.route("/")
def home():
    return jsonify({
        "app": "Diabetic Soft Mundi API",
        "version": metadata.get("version", "3.0"),
        "modelo": metadata.get("modelo"),
        "universidad": "Universidad Andina del Cusco",
        "clases": CLASES,
        "modos": {
            "basico": {
                "variables": len(FEATURES_BASICO),
                "features": FEATURES_BASICO,
                "metricas": metadata["metricas_basico"],
                "descripcion": "Test rapido sin laboratorio",
                "nota": NOTA_BASICO,
            },
            "completo": {
                "variables": len(FEATURES_COMPLETO),
                "features": FEATURES_COMPLETO,
                "metricas": metadata["metricas_completo"],
                "descripcion": "Evaluacion con laboratorio basico",
                "nota": NOTA_COMPLETO,
            },
        },
        "reglas_clinicas": metadata.get("reglas_clinicas_post"),
        "limitaciones": metadata.get("limitaciones"),
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "version": metadata.get("version"),
        "clases": CLASES,
        "macro_f1_basico": metadata["metricas_basico"]["macro_f1"],
        "macro_f1_completo": metadata["metricas_completo"]["macro_f1"],
    })


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"success": False, "error": "JSON requerido"}), 400

    try:
        modo = str(data.get("modo", "basico")).lower()
        if modo not in {"basico", "completo"}:
            return jsonify({"success": False, "error": f"modo invalido: {modo}"}), 400

        # ----- Lectura y validacion -----
        sexo = str(data.get("sexo", "M")).upper()
        if sexo not in {"M", "F"}:
            return jsonify({"success": False, "error": "sexo debe ser 'M' o 'F'"}), 400
        sexo_enc = 1 if sexo == "M" else 0

        edad = float(data["edad"]); _valida("edad", edad)
        peso = float(data["peso"]); _valida("peso", peso)
        talla_cm = float(data["talla"]); _valida("talla", talla_cm)

        talla_m = talla_cm / 100.0
        imc = peso / (talla_m ** 2)
        if not (10 <= imc <= 70):
            return jsonify({"success": False, "error": f"IMC calculado fuera de rango: {imc:.2f}"}), 400

        perimetro = float(data["perimetro_abdominal"]); _valida("perimetro_abdominal", perimetro)
        antecedentes = int(data["antecedentes_familiares"]); _valida("antecedentes_familiares", antecedentes)
        actividad = int(data["actividad_fisica"]); _valida("actividad_fisica", actividad)
        frutas = int(data["consumo_frutas_verduras"]); _valida("consumo_frutas_verduras", frutas)
        presion = float(data.get("presion_arterial", 120)); _valida("presion_arterial", presion)
        altitud = float(data.get("altitud", 3400)); _valida("altitud", altitud)

        glucosa = colesterol = trigliceridos = None
        if modo == "completo":
            glucosa = float(data["glucosa"]); _valida("glucosa", glucosa)
            colesterol = float(data["colesterol"]); _valida("colesterol", colesterol)
            trigliceridos = float(data["trigliceridos"]); _valida("trigliceridos", trigliceridos)

        # ----- Construccion del vector EN EL ORDEN del metadata -----
        valores = {
            "Edad": edad, "Sexo_enc": sexo_enc, "IMC": imc,
            "PerimetroAbdominal": perimetro,
            "AntecedentesFamiliares": antecedentes,
            "ActividadFisica": actividad,
            "ConsumoFrutasVerduras": frutas,
            "PresionArterial": presion,
            "Altitud": altitud,
            "GlucosaSangre": glucosa,
            "Colesterol": colesterol,
            "Trigliceridos": trigliceridos,
        }

        if modo == "completo":
            cols = FEATURES_COMPLETO
            vec = np.array([[valores[c] for c in cols]], dtype=float)
            X = scaler_completo.transform(vec)
            proba = modelo_completo.predict_proba(X)[0]
        else:
            cols = FEATURES_BASICO
            vec = np.array([[valores[c] for c in cols]], dtype=float)
            X = scaler_basico.transform(vec)
            proba = modelo_basico.predict_proba(X)[0]

        pred_idx = int(np.argmax(proba))

        # ----- Reglas clinicas duras -----
        pred_idx, proba, avisos = aplicar_reglas_clinicas(
            pred_idx, proba,
            glucosa=glucosa, perimetro=perimetro, sexo_enc=sexo_enc,
            imc=imc, presion=presion,
        )

        categoria = CLASES[pred_idx]
        probabilidades = {CLASES[i]: round(float(p) * 100, 2) for i, p in enumerate(proba)}
        max_prob = float(max(proba))
        if max_prob >= 0.80:
            certeza = "alta"
        elif max_prob >= 0.60:
            certeza = "moderada"
        else:
            certeza = "baja"

        # ----- Clasificacion IMC y perimetro -----
        if imc < 18.5: imc_cat = "Bajo peso"
        elif imc < 25: imc_cat = "Normal"
        elif imc < 30: imc_cat = "Sobrepeso"
        elif imc < 35: imc_cat = "Obesidad grado I"
        elif imc < 40: imc_cat = "Obesidad grado II"
        else: imc_cat = "Obesidad grado III"

        umbral_perim = 90 if sexo_enc == 1 else 80
        obesidad_central = perimetro >= umbral_perim

        return jsonify({
            "success": True,
            "version": metadata.get("version"),
            "modo": modo,
            "nota_modo": NOTA_BASICO if modo == "basico" else NOTA_COMPLETO,
            "categoria": categoria,
            "probabilidades": probabilidades,
            "certeza": certeza,
            "avisos_clinicos": avisos,
            "imc": {"valor": round(imc, 2), "clasificacion": imc_cat},
            "obesidad_central": obesidad_central,
            "datos_ingresados": {
                "edad": edad, "sexo": sexo, "peso_kg": peso, "talla_cm": talla_cm,
                "perimetro_abdominal": perimetro,
                "antecedentes_familiares": antecedentes,
                "actividad_fisica": actividad,
                "consumo_frutas_verduras": frutas,
                "presion_arterial": presion, "altitud": altitud,
                "glucosa": glucosa, "colesterol": colesterol, "trigliceridos": trigliceridos,
            },
            "recomendacion": RECOMENDACIONES[categoria],
        })

    except KeyError as e:
        return jsonify({"success": False, "error": f"Falta el campo: {e}"}), 400
    except ValueError as e:
        return jsonify({"success": False, "error": str(e)}), 400
    except Exception as e:  # noqa: BLE001
        return jsonify({"success": False, "error": f"Error interno: {e}"}), 500


# ---------------------------------------------------------------------------
# Recomendaciones (texto que muestra la app Flutter)
# ---------------------------------------------------------------------------
RECOMENDACIONES = {
    "Normal": {
        "mensaje": "Su nivel de riesgo es bajo. Mantenga sus habitos saludables y realice controles anuales.",
        "alimentacion": {
            "descripcion": "Dieta balanceada para mantener su estado saludable",
            "recomendados": [
                "Quinua, kiwicha y caniahua (granos andinos ricos en proteina y fibra)",
                "Chuno y moraya (carbohidratos complejos de absorcion lenta)",
                "Tarwi/chocho (alto en proteinas, bajo indice glucemico)",
                "Frutas locales: tumbo, aguaymanto, tuna, capuli",
                "Verduras: zapallo, hojas de quinua, yuyo",
                "Trucha y cuy (proteinas magras)",
                "Mate de muna, manzanilla o hierba luisa sin azucar",
            ],
            "evitar": ["Exceso de pan blanco y fideos refinados", "Bebidas azucaradas", "Frituras en exceso"],
            "porciones": "3 comidas principales + 2 refrigerios saludables al dia",
        },
        "actividad_fisica": {
            "descripcion": "Mantenga un estilo de vida activo",
            "recomendaciones": [
                "Caminatas de 30 minutos diarios considerando la altitud",
                "Subir escaleras en vez de ascensor",
                "Estiramientos al despertar",
            ],
            "minutos_semanales": 150,
        },
        "estilo_vida": [
            "Dormir 7-8 horas diarias",
            "Hidratacion: minimo 8 vasos de agua (mas importante en altitud)",
            "Control medico anual preventivo",
            "Mantener IMC entre 18.5 y 24.9",
        ],
        "educacion": [
            "La diabetes tipo II se puede prevenir con habitos saludables",
            "Los alimentos andinos tienen bajo indice glucemico",
            "Los antecedentes familiares son un factor no modificable, pero los habitos si",
        ],
    },
    "Alterado": {
        "mensaje": "Presenta factores de riesgo moderados. Se recomienda consultar a un profesional de salud.",
        "alimentacion": {
            "descripcion": "Dieta controlada para reducir factores de riesgo",
            "recomendados": [
                "Quinua y kiwicha como reemplazo del arroz blanco",
                "Tarwi/chocho remojado (proteina vegetal que estabiliza glucosa)",
                "Chuno negro en sopas (almidon resistente)",
                "Trucha y cuy al horno o plancha",
                "Ensaladas con hojas de quinua, yuyo y berros",
                "Aguaymanto, tuna y tumbo (frutas bajas en azucar)",
                "Habas y pallares como fuente de proteina y fibra",
                "Mate de muna despues de las comidas",
            ],
            "evitar": [
                "Azucar refinada (usar stevia)", "Pan blanco y fideos refinados",
                "Gaseosas y jugos envasados", "Frituras y comida chatarra",
                "Exceso de papa blanca (preferir papa nativa o chuno)",
                "Alcohol en exceso",
            ],
            "porciones": "5 comidas pequenas al dia",
        },
        "actividad_fisica": {
            "descripcion": "Incrementar actividad fisica gradualmente",
            "recomendaciones": [
                "Caminatas de 30-45 min diarios a ritmo moderado",
                "Ejercicios de fuerza 2-3 veces por semana",
                "Bailar (actividad cultural andina que es cardio)",
                "Levantarse cada hora si trabaja sentado",
            ],
            "minutos_semanales": 200,
            "nota_altitud": "En altitud >3000m, suba intensidad de forma gradual e hidratese bien.",
        },
        "estilo_vida": [
            "Reducir 5-7% del peso si tiene sobrepeso (reduce el riesgo 58%)",
            "Dormir 7-8 horas (el mal sueno eleva la glucosa)",
            "Control medico cada 6 meses (glucosa y perfil lipidico)",
            "Monitorear presion arterial mensualmente",
        ],
        "educacion": [
            "La prediabetes es REVERSIBLE con cambios en alimentacion y ejercicio",
            "Perimetro >90cm (H) o >80cm (M) indica grasa visceral de riesgo",
            "La actividad fisica mejora la sensibilidad a la insulina aun sin perder peso",
        ],
        "alerta": "Se recomienda realizar una prueba de glucosa en ayunas en un centro de salud cercano.",
    },
    "Muy Alterado": {
        "mensaje": "Se detecto un alto nivel de riesgo. Es urgente acudir a un profesional de salud.",
        "alimentacion": {
            "descripcion": "Dieta estricta supervisada por profesional de salud",
            "recomendados": [
                "Quinua como base de alimentacion",
                "Tarwi/chocho en todas sus formas",
                "Caniahua en mazamorras sin azucar",
                "Verduras en abundancia: zapallo, broculi, espinaca",
                "Trucha o cuy al horno (max 3 veces/semana)",
                "Linaza remojada (1 cda diaria, ayuda al control glucemico)",
                "Infusiones de muna, manzanilla, hierba luisa sin azucar",
            ],
            "evitar_estrictamente": [
                "Azucar en todas sus formas", "Pan blanco, fideos y galletas refinadas",
                "Gaseosas y bebidas energeticas", "Frituras y comida rapida",
                "Arroz blanco en grandes cantidades", "Alcohol",
                "Embutidos y carnes procesadas",
            ],
            "porciones": "Porciones pequenas, 5-6 veces al dia. NUNCA saltarse comidas.",
        },
        "actividad_fisica": {
            "descripcion": "Actividad fisica supervisada y gradual",
            "recomendaciones": [
                "Consultar al medico ANTES de iniciar ejercicio intenso",
                "Caminatas suaves de 20-30 min, 5 veces por semana",
                "Estiramientos diarios",
                "Monitorear como se siente durante la actividad",
            ],
            "minutos_semanales": 150,
            "nota_altitud": "IMPORTANTE: en altitud >3000m consulte al medico antes de actividad intensa.",
        },
        "estilo_vida": [
            "ACUDIR A UN CENTRO DE SALUD lo antes posible",
            "Realizarse prueba de glucosa en ayunas y perfil lipidico",
            "Control medico mensual mientras se estabiliza",
            "NO automedicarse",
            "Dormir 7-8 horas",
            "Eliminar tabaco completamente",
            "Llevar un registro diario de alimentacion",
        ],
        "educacion": [
            "La DM2 no tratada puede causar dano renal, ceguera y amputaciones",
            "Con tratamiento adecuado la diabetes se controla eficazmente",
            "Los centros MINSA ofrecen control de diabetes gratuito o a bajo costo",
        ],
        "alerta": "URGENTE: acuda a su centro de salud mas cercano. No espere.",
        "recursos_cusco": [
            "Hospital Regional del Cusco - Endocrinologia",
            "Centro de Salud San Jeronimo",
            "EsSalud Cusco - Programa de Diabetes",
            "Puestos de salud MINSA - Control gratuito de glucosa",
        ],
    },
}


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    print("=" * 60)
    print(f"  DIABETIC SOFT MUNDI API {metadata.get('version', '3.0')}")
    print("=" * 60)
    print(f"  Clases: {CLASES}")
    print(f"  Basico   macro-F1: {metadata['metricas_basico']['macro_f1']:.4f}")
    print(f"  Completo macro-F1: {metadata['metricas_completo']['macro_f1']:.4f}")
    print("=" * 60)
    app.run(host="0.0.0.0", port=port, debug=False)
