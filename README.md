# Diabetic Soft Mundi API v2.0
## Universidad Andina del Cusco - Pedraza & Gonzales

### Archivos necesarios en el repositorio:
```
├── main.py                  ← API Flask
├── requirements.txt         ← Dependencias
├── Procfile                 ← Configuración Gunicorn
├── modelo_basico.pkl        ← MLP Neural Net (de Colab)
├── modelo_completo.pkl      ← Logistic Regression (de Colab)
├── scaler_basico.pkl        ← StandardScaler (de Colab)
├── scaler_completo.pkl      ← StandardScaler (de Colab)
└── metadata.pkl             ← Encoders + config (de Colab)
```

### Pasos para desplegar en Render:

1. Crear repositorio en GitHub con todos los archivos
2. En Render → New → Web Service → Conectar repo
3. Configurar:
   - Runtime: Python
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `gunicorn main:app --bind 0.0.0.0:$PORT`
4. Deploy

### Endpoints:

- `GET /` → Info del API
- `GET /health` → Estado del servidor
- `POST /predict` → Predicción

### Ejemplo de request - Modo Básico:
```json
{
  "modo": "basico",
  "edad": 45,
  "sexo": "M",
  "peso": 85,
  "talla": 170,
  "perimetro_abdominal": 98,
  "antecedentes_familiares": 1,
  "actividad_fisica": 0,
  "consumo_frutas_verduras": 0,
  "presion_arterial": 135,
  "altitud": 3400
}
```

### Ejemplo de request - Modo Completo:
```json
{
  "modo": "completo",
  "edad": 45,
  "sexo": "M",
  "peso": 85,
  "talla": 170,
  "perimetro_abdominal": 98,
  "antecedentes_familiares": 1,
  "actividad_fisica": 0,
  "consumo_frutas_verduras": 0,
  "presion_arterial": 135,
  "altitud": 3400,
  "glucosa": 150,
  "colesterol": 230,
  "trigliceridos": 210
}
```

### Ejemplo de response:
```json
{
  "success": true,
  "modo": "basico",
  "variables_usadas": 9,
  "categoria": "Alterado",
  "probabilidades": {
    "Alterado": 62.5,
    "Muy Alterado": 15.3,
    "Normal": 22.2
  },
  "certeza": "moderada",
  "recomendacion": "Presenta factores de riesgo moderados...",
  "imc": {
    "valor": 29.41,
    "clasificacion": "Sobrepeso"
  }
}
```
