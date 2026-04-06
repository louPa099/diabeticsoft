from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)

# ============================================================
# Cargar modelos al iniciar el servidor
# Archivos que deben estar en el mismo directorio:
#   modelo_basico.pkl     → MLP Neural Net (9 variables)
#   modelo_completo.pkl   → Logistic Regression (12 variables)
#   scaler_basico.pkl     → StandardScaler entrenado
#   scaler_completo.pkl   → StandardScaler entrenado
#   metadata.pkl          → Encoders + configuración
# ============================================================
modelo_basico = joblib.load('modelo_basico.pkl')
modelo_completo = joblib.load('modelo_completo.pkl')
scaler_basico = joblib.load('scaler_basico.pkl')
scaler_completo = joblib.load('scaler_completo.pkl')
metadata = joblib.load('metadata.pkl')

CLASES = metadata['clases']  # ['Alterado', 'Muy Alterado', 'Normal']

# ============================================================
# RUTA PRINCIPAL - Info del API
# ============================================================
@app.route("/")
def home():
    return jsonify({
        "app": "Diabetic Soft Mundi API",
        "version": "2.0",
        "universidad": "Universidad Andina del Cusco",
        "modos": {
            "basico": {
                "variables": 9,
                "modelo": metadata.get('mejor_modelo_basico', 'MLP Neural Net'),
                "accuracy": metadata.get('accuracy_basico'),
                "descripcion": "Sin laboratorio - solo cuestionario + antropometría"
            },
            "completo": {
                "variables": 12,
                "modelo": metadata.get('mejor_modelo_completo', 'Logistic Regression'),
                "accuracy": metadata.get('accuracy_completo'),
                "descripcion": "Con laboratorio básico (glucómetro + lab simple)"
            }
        },
        "clases": CLASES,
        "nota": "HbA1c excluida por data leakage"
    })

# ============================================================
# RUTA DE PREDICCIÓN
# ============================================================
@app.route("/predict", methods=['POST'])
def predict():
    data = request.get_json()
    if data is None:
        return jsonify({"success": False, "error": "No se proporcionaron datos JSON"}), 400

    try:
        modo = data.get('modo', 'basico')  # 'basico' o 'completo'

        # ── Variables comunes (modo básico) ──
        # Sexo: 'M' o 'F'
        sexo = data.get('sexo', 'M').upper()
        sexo_enc = 1 if sexo == 'M' else 0

        # Edad en años
        edad = float(data['edad'])

        # IMC calculado automáticamente desde peso y talla
        peso = float(data['peso'])          # kg
        talla_cm = float(data['talla'])     # cm
        talla_m = talla_cm / 100.0
        imc = peso / (talla_m ** 2)

        # Perímetro abdominal en cm
        perimetro = float(data['perimetro_abdominal'])

        # Antecedentes familiares: 0=Ninguno, 1=Un familiar, 2=Ambos
        antecedentes = int(data['antecedentes_familiares'])

        # Actividad física: 0=Sedentario, 1=Moderado, 2=Activo
        actividad = int(data['actividad_fisica'])

        # Consumo frutas/verduras: 0=No diario, 1=Diario
        frutas = int(data['consumo_frutas_verduras'])

        # Presión arterial sistólica (mmHg) - opcional, default 120
        presion = float(data.get('presion_arterial', 120))

        # Altitud m.s.n.m. - opcional, default 3400 (Cusco)
        altitud = float(data.get('altitud', 3400))

        # ── Orden EXACTO de features (debe coincidir con entrenamiento) ──
        # ['Edad','Sexo_enc','IMC','PerimetroAbdominal','AntecedentesFamiliares',
        #  'ActividadFisica','ConsumoFrutasVerduras','PresionArterial','Altitud']
        features_basico = [
            edad, sexo_enc, imc, perimetro, antecedentes,
            actividad, frutas, presion, altitud
        ]

        if modo == 'completo':
            # ── Variables adicionales (laboratorio básico) ──
            glucosa = float(data['glucosa'])            # mg/dl
            colesterol = float(data['colesterol'])      # mg/dl
            trigliceridos = float(data['trigliceridos']) # mg/dl
            # HbA1c NO se usa (data leakage)

            features = features_basico + [glucosa, colesterol, trigliceridos]
            features_scaled = scaler_completo.transform([features])
            pred = modelo_completo.predict(features_scaled)[0]
            proba = modelo_completo.predict_proba(features_scaled)[0]
            modelo_usado = 'completo'
            n_vars = 12
        else:
            features_scaled = scaler_basico.transform([features_basico])
            pred = modelo_basico.predict(features_scaled)[0]
            proba = modelo_basico.predict_proba(features_scaled)[0]
            modelo_usado = 'basico'
            n_vars = 9

        # ── Resultado ──
        categoria = CLASES[int(pred)]
        probabilidades = {
            CLASES[i]: round(float(p) * 100, 2)
            for i, p in enumerate(proba)
        }

        # ── Certeza ──
        max_prob = float(max(proba))
        if max_prob > 0.8:
            certeza = "alta"
        elif max_prob > 0.6:
            certeza = "moderada"
        else:
            certeza = "baja"

        # ── Recomendaciones detalladas ──
        recomendaciones = {
            'Normal': {
                'mensaje': 'Su nivel de riesgo es bajo. Mantenga sus hábitos saludables y realice controles anuales.',
                'alimentacion': {
                    'descripcion': 'Dieta balanceada para mantener su estado saludable',
                    'recomendados': [
                        'Quinua, kiwicha y cañihua (granos andinos ricos en proteína y fibra)',
                        'Chuño y moraya (fuente de carbohidratos complejos)',
                        'Tarwi/chocho (alto en proteínas, bajo índice glucémico)',
                        'Frutas locales: tumbo, aguaymanto, tuna, capulí',
                        'Verduras: zapallo, calabaza, hojas de quinua, yuyo',
                        'Trucha y cuy (proteínas magras)',
                        'Habas, arvejas y pallares frescos',
                        'Mate de muña, manzanilla o hierba luisa (infusiones sin azúcar)'
                    ],
                    'evitar': [
                        'Exceso de pan blanco y fideos refinados',
                        'Bebidas azucaradas y gaseosas',
                        'Frituras en exceso'
                    ],
                    'porciones': '3 comidas principales + 2 refrigerios saludables al día'
                },
                'actividad_fisica': {
                    'descripcion': 'Mantenga un estilo de vida activo',
                    'recomendaciones': [
                        'Caminatas de 30 minutos diarios (considere la altitud)',
                        'Actividades recreativas: caminatas, paseos en bicicleta',
                        'Ejercicios de estiramiento al despertar',
                        'Subir escaleras en lugar de usar ascensor'
                    ],
                    'minutos_semanales': 150
                },
                'estilo_vida': [
                    'Dormir entre 7 y 8 horas diarias',
                    'Hidratarse bien (mínimo 8 vasos de agua al día, importante en altitud)',
                    'Control médico anual preventivo',
                    'Mantener un peso saludable (IMC entre 18.5 y 24.9)'
                ],
                'educacion': [
                    'La diabetes tipo II se puede prevenir con hábitos saludables',
                    'Los alimentos andinos tienen bajo índice glucémico, son aliados naturales',
                    'La altitud de Cusco puede afectar cómo el cuerpo procesa los azúcares',
                    'Los antecedentes familiares son un factor de riesgo no modificable, pero los hábitos sí lo son'
                ]
            },
            'Alterado': {
                'mensaje': 'Presenta factores de riesgo moderados. Se recomienda consultar a un profesional de salud.',
                'alimentacion': {
                    'descripcion': 'Dieta controlada para reducir factores de riesgo',
                    'recomendados': [
                        'Quinua y kiwicha como reemplazo del arroz blanco (menor índice glucémico)',
                        'Tarwi/chocho remojado (rico en proteína vegetal, estabiliza glucosa)',
                        'Chuño negro en sopas (carbohidrato complejo de absorción lenta)',
                        'Moraya rallada en preparaciones (fuente de almidón resistente)',
                        'Trucha y cuy al horno o a la plancha (proteína magra)',
                        'Ensaladas con hojas de quinua, yuyo y berros',
                        'Aguaymanto, tuna y tumbo (frutas bajas en azúcar)',
                        'Habas, tarwi y pallares como fuente de proteína y fibra',
                        'Mate de muña después de las comidas (digestivo natural)',
                        'Agua de cebada sin azúcar'
                    ],
                    'evitar': [
                        'Azúcar refinada (reemplazar con stevia o pequeñas cantidades de miel)',
                        'Pan blanco y fideos (preferir integrales o de quinua)',
                        'Gaseosas y jugos envasados',
                        'Frituras y comida chatarra',
                        'Exceso de papa blanca (preferir papa nativa o chuño)',
                        'Chicharrón y carnes grasas en exceso',
                        'Alcohol en exceso'
                    ],
                    'porciones': '5 comidas pequeñas al día (evitar comer en exceso)',
                    'ejemplo_menu': {
                        'desayuno': 'Avena con quinua, aguaymanto y canela (sin azúcar)',
                        'media_manana': 'Un puñado de habas tostadas + una tuna',
                        'almuerzo': 'Sopa de moraya + trucha a la plancha con ensalada de quinua',
                        'media_tarde': 'Mate de muña + tarwi con limón',
                        'cena': 'Crema de zapallo con chuño rallado + infusión de manzanilla'
                    }
                },
                'actividad_fisica': {
                    'descripcion': 'Incrementar actividad física gradualmente',
                    'recomendaciones': [
                        'Caminatas de 30-45 minutos diarios (ritmo moderado)',
                        'Subir gradualmente la intensidad (considerar altitud de Cusco)',
                        'Ejercicios de fuerza 2-3 veces por semana (sentadillas, flexiones)',
                        'Estiramiento diario de 10-15 minutos',
                        'Bailar (actividad cultural andina que es ejercicio cardiovascular)',
                        'Evitar el sedentarismo: levantarse cada hora si trabaja sentado'
                    ],
                    'minutos_semanales': 200,
                    'nota_altitud': 'En altitud >3000m, la actividad física requiere más oxígeno. Aumente gradualmente la intensidad y mantenga buena hidratación.'
                },
                'estilo_vida': [
                    'Controlar peso corporal: reducir 5-7% si tiene sobrepeso',
                    'Dormir entre 7 y 8 horas (el mal sueño eleva la glucosa)',
                    'Reducir estrés con técnicas de respiración o meditación',
                    'Hidratarse con 8-10 vasos de agua al día',
                    'Control médico cada 6 meses (glucosa y perfil lipídico)',
                    'Monitorear presión arterial mensualmente',
                    'Evitar fumar y reducir consumo de alcohol'
                ],
                'educacion': [
                    'La prediabetes es REVERSIBLE con cambios en alimentación y ejercicio',
                    'Perder entre 5 y 7% de peso puede reducir el riesgo de diabetes en 58%',
                    'Los granos andinos (quinua, kiwicha, cañihua) tienen menor índice glucémico que el arroz',
                    'El perímetro abdominal >94cm (hombres) o >80cm (mujeres) indica grasa visceral de riesgo',
                    'La actividad física mejora la sensibilidad a la insulina incluso sin perder peso',
                    'En altitud, el cuerpo puede tener respuestas metabólicas diferentes'
                ],
                'alerta': 'Se recomienda realizar una prueba de glucosa en ayunas en un centro de salud cercano.'
            },
            'Muy Alterado': {
                'mensaje': 'Se detectó un alto nivel de riesgo. Es urgente acudir a un profesional de salud.',
                'alimentacion': {
                    'descripcion': 'Dieta estricta supervisada por profesional de salud',
                    'recomendados': [
                        'Quinua como base de alimentación (reemplazar arroz y fideos)',
                        'Tarwi/chocho en todas sus formas (ensalada, sopa, guiso)',
                        'Cañihua en mazamorras sin azúcar (alto valor nutricional)',
                        'Chuño y moraya en cantidades controladas',
                        'Verduras en abundancia: zapallo, calabaza, brócoli, espinaca',
                        'Trucha o cuy al horno (máximo 3 veces por semana)',
                        'Frutas con bajo índice glucémico: aguaymanto, tuna (con moderación)',
                        'Infusiones: muña, manzanilla, hierba luisa (sin azúcar)',
                        'Linaza remojada (1 cucharada diaria, ayuda al control glucémico)'
                    ],
                    'evitar_estrictamente': [
                        'Azúcar en todas sus formas (blanca, rubia, miel en exceso)',
                        'Pan blanco, fideos, galletas refinadas',
                        'Gaseosas, jugos envasados, bebidas energéticas',
                        'Frituras, chicharrón, pollo broaster',
                        'Comida rápida y ultraprocesados',
                        'Arroz blanco en grandes cantidades',
                        'Alcohol',
                        'Embutidos y carnes procesadas'
                    ],
                    'porciones': 'Porciones pequeñas, 5-6 veces al día. NUNCA saltarse comidas.',
                    'ejemplo_menu': {
                        'desayuno': 'Papilla de cañihua con canela + mate de muña',
                        'media_manana': 'Tarwi con limón + 1 aguaymanto',
                        'almuerzo': 'Ensalada de quinua con verduras + trucha al horno + chuño',
                        'media_tarde': 'Habas tostadas (un puñado pequeño) + agua de linaza',
                        'cena': 'Sopa de moraya con verduras + infusión de manzanilla'
                    }
                },
                'actividad_fisica': {
                    'descripcion': 'Actividad física supervisada y gradual',
                    'recomendaciones': [
                        'Consultar con médico ANTES de iniciar ejercicio intenso',
                        'Caminatas suaves de 20-30 minutos, 5 veces por semana',
                        'Aumentar gradualmente (la altitud de Cusco demanda más esfuerzo)',
                        'Ejercicios de estiramiento diarios',
                        'Evitar ejercicio intenso sin supervisión médica',
                        'Monitorear cómo se siente durante la actividad (mareos, fatiga)'
                    ],
                    'minutos_semanales': 150,
                    'nota_altitud': 'IMPORTANTE: En altitud >3000m, consulte a su médico antes de realizar actividad física intensa. Comience lentamente.'
                },
                'estilo_vida': [
                    'ACUDIR A UN CENTRO DE SALUD lo antes posible para evaluación completa',
                    'Realizarse prueba de glucosa en ayunas y perfil lipídico',
                    'Control médico mensual mientras se estabiliza',
                    'NO automedicarse con remedios caseros o fármacos sin prescripción',
                    'Dormir 7-8 horas (el sueño irregular eleva la glucosa)',
                    'Eliminar tabaco completamente',
                    'Eliminar o reducir drásticamente el alcohol',
                    'Controlar peso corporal con guía profesional',
                    'Monitorear presión arterial semanalmente',
                    'Llevar un registro diario de alimentación'
                ],
                'educacion': [
                    'La diabetes tipo II no tratada puede causar complicaciones graves: daño renal, ceguera, amputaciones',
                    'CON TRATAMIENTO ADECUADO, la diabetes se puede controlar efectivamente',
                    'Los cambios en alimentación y ejercicio pueden ser tan efectivos como los medicamentos en etapas iniciales',
                    'La altitud de Cusco puede afectar la lectura de glucómetros y hemoglobina',
                    'Es importante involucrar a la familia en los cambios de alimentación',
                    'Los centros de salud MINSA ofrecen control de diabetes de forma gratuita o a bajo costo',
                    'Aprender a leer etiquetas de alimentos: buscar contenido de azúcar y carbohidratos'
                ],
                'alerta': 'URGENTE: Acuda a su centro de salud más cercano para una evaluación médica completa. No espere.',
                'recursos_cusco': [
                    'Hospital Regional del Cusco - Servicio de Endocrinología',
                    'Centro de Salud San Jerónimo',
                    'EsSalud Cusco - Programa de Diabetes',
                    'Puestos de salud MINSA - Control gratuito de glucosa'
                ]
            }
        }

        # ── Clasificación IMC ──
        if imc < 18.5:
            imc_cat = "Bajo peso"
        elif imc < 25:
            imc_cat = "Normal"
        elif imc < 30:
            imc_cat = "Sobrepeso"
        elif imc < 35:
            imc_cat = "Obesidad grado I"
        elif imc < 40:
            imc_cat = "Obesidad grado II"
        else:
            imc_cat = "Obesidad grado III"

        return jsonify({
            "success": True,
            "modo": modelo_usado,
            "variables_usadas": n_vars,
            "categoria": categoria,
            "probabilidades": probabilidades,
            "certeza": certeza,
            "recomendacion": recomendaciones[categoria],
            "imc": {
                "valor": round(imc, 2),
                "clasificacion": imc_cat
            },
            "datos_ingresados": {
                "edad": edad,
                "sexo": sexo,
                "peso_kg": peso,
                "talla_cm": talla_cm,
                "perimetro_abdominal": perimetro,
                "antecedentes_familiares": antecedentes,
                "actividad_fisica": actividad,
                "consumo_frutas_verduras": frutas,
                "presion_arterial": presion,
                "altitud": altitud
            }
        })

    except KeyError as e:
        return jsonify({
            "success": False,
            "error": f"Falta el campo: {str(e)}"
        }), 400
    except ValueError as e:
        return jsonify({
            "success": False,
            "error": f"Error en formato de datos: {str(e)}"
        }), 400
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Error interno: {str(e)}"
        }), 500

# ============================================================
# RUTA DE SALUD
# ============================================================
@app.route("/health", methods=['GET'])
def health():
    return jsonify({
        'status': 'ok',
        'modelos_cargados': True,
        'clases': CLASES,
        'basico': {
            'modelo': metadata.get('mejor_modelo_basico'),
            'accuracy': metadata.get('accuracy_basico')
        },
        'completo': {
            'modelo': metadata.get('mejor_modelo_completo'),
            'accuracy': metadata.get('accuracy_completo')
        }
    })

# ============================================================
# INICIAR SERVIDOR
# ============================================================
if __name__ == "__main__":
    print("=" * 50)
    print("  DIABETIC SOFT MUNDI API v2.0")
    print("=" * 50)
    print(f"  Clases: {CLASES}")
    print(f"  Básico:   {metadata.get('mejor_modelo_basico')} ({metadata.get('accuracy_basico', 0):.4f})")
    print(f"  Completo: {metadata.get('mejor_modelo_completo')} ({metadata.get('accuracy_completo', 0):.4f})")
    print(f"  HbA1c: EXCLUIDA")
    print("=" * 50)
    app.run(host='0.0.0.0', port=10000, debug=False)
