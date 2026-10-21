import streamlit as st
import pandas as pd
import numpy as np

# interfaz de usuario
st.title("Red Neuronal: Clasificacion Geografica")
st.header("Clasificacion de Coordenadas GPS (America vs. Asia)")
st.write("Entrenamiento de un MLP utilizando unicamente NumPy.")

# funciones de activacion
def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def derivada_relu(x):
    return np.where(x > 0, 1, 0)

relu = (
    lambda x: x * (x > 0),
    lambda x: derivada_relu(x)
)

sigmoid_tuple = (
    sigmoid,
    lambda x: x * (1 - x)
)

# generar coordenadas GPS aleatorias para Sao Paulo y Tokio
def circulo(num_datos=150, lat_range=(-23.7236123, -23.3763547), lon_range=(-46.9501189, -46.3080714)):
    lat = np.random.uniform(lat_range[0], lat_range[1], num_datos)
    lon = np.random.uniform(lon_range[0], lon_range[1], num_datos)
    return np.column_stack([lat, lon])

np.random.seed(42)
N = 150

# 10 puntos fijos especificos para Sao Paulo (America - Clase 0)
fijos_sp = np.array([
    [-23.5505, -46.6333],
    [-23.5614, -46.6559],
    [-23.5477, -46.6358],
    [-23.5874, -46.6576],
    [-23.6231, -46.6542],
    [-23.5329, -46.6395],
    [-23.5629, -46.6544],
    [-23.5705, -46.6452],
    [-23.5489, -46.6388],
    [-23.5558, -46.6612]
])

# 10 puntos fijos especificos para Tokio (Asia - Clase 1)
fijos_tokio = np.array([
    [35.6762, 139.6503],
    [35.6895, 139.6917],
    [35.6586, 139.7454],
    [35.7101, 139.8107],
    [35.6284, 139.7387],
    [35.6852, 139.7528],
    [35.6938, 139.7034],
    [35.6691, 139.7031],
    [35.7126, 139.7712],
    [35.6581, 139.7017]
])

# 140 puntos aleatorios restantes para completar 150 por ciudad
random_sp = circulo(num_datos=140, lat_range=(-23.7236123, -23.3763547), lon_range=(-46.9501189, -46.3080714))
random_tokio = circulo(num_datos=140, lat_range=(35.5009845, 35.8320508), lon_range=(139.4509760, 139.9442734))

# combinar puntos fijos y aleatorios
datos_sp = np.vstack([fijos_sp, random_sp])
datos_tokio = np.vstack([fijos_tokio, random_tokio])

# matriz combinada
X_raw = np.concatenate([datos_sp, datos_tokio])
Y = np.array([0] * N + [1] * N).reshape(len(X_raw), 1)

# normalizacion Min-Max para el entrenamiento
X_min = X_raw.min(axis=0)
X_max = X_raw.max(axis=0)
X = (X_raw - X_min) / (X_max - X_min)

# mostrar mapa de las coordenadas generadas
st.subheader("1. Visualizacion del Dataset GPS")

map_data = pd.DataFrame(X_raw, columns=['lat', 'lon'])
map_data['Ciudad'] = ['Sao Paulo'] * N + ['Tokio'] * N

st.map(map_data)

st.write("Vista previa de las coordenadas generadas (Sao Paulo y Tokio):")
vista_previa = pd.concat([map_data.head(5), map_data.tail(5)])
st.dataframe(vista_previa)

# entrenamiento de la red neuronal
class Capa():
    def __init__(self, n_anterior, n_neuronas, funcion_act):
        self.funcion_act = funcion_act
        self.b = np.zeros((1, n_neuronas))
        self.W = np.random.randn(n_anterior, n_neuronas) * 0.1

def entrenamiento(X, Y, red_neuronal, lr=0.1):
    # Forward Pass
    output = [X]
    for num_capa in range(len(red_neuronal)):
        z = output[-1] @ red_neuronal[num_capa].W + red_neuronal[num_capa].b
        a = red_neuronal[num_capa].funcion_act[0](z)
        output.append(a)

    # Backpropagation
    back = list(range(len(output)-1))
    back.reverse()
    delta = []

    for capa_idx in back:
        a = output[capa_idx+1]

        if capa_idx == back[0]:
            x = (a - Y) * red_neuronal[capa_idx].funcion_act[1](a)
            delta.append(x)
        else:
            x = (delta[-1] @ W_temp) * red_neuronal[capa_idx].funcion_act[1](a)
            delta.append(x)

        W_temp = red_neuronal[capa_idx].W.transpose()

        # Gradient Descent
        red_neuronal[capa_idx].b -= np.mean(delta[-1], axis=0, keepdims=True) * lr
        red_neuronal[capa_idx].W -= output[capa_idx].transpose() @ delta[-1] * lr

    loss = np.mean((output[-1] - Y) ** 2)
    return output[-1], loss

# proceso y resultados del entrenamiento
st.subheader("2. Entrenamiento e Historial Paso a Paso")

if st.button("Iniciar Entrenamiento"):
    neuronas = [2, 4, 8, 1]
    funciones_activacion = [relu, relu, sigmoid_tuple]
    
    red_neuronal = []
    for paso in range(len(neuronas)-1):
        red_neuronal.append(Capa(neuronas[paso], neuronas[paso+1], funciones_activacion[paso]))

    bar = st.progress(0)
    
    # listas para guardar el historial paso a paso
    historial_loss = []
    historial_precision = []
    registros_epocas = []
    
    epochs = 200
    prediccion_inicial = None

    for epoch in range(epochs):
        prediccion, loss = entrenamiento(X, Y, red_neuronal, lr=0.1)
        
        # guardar prediccion inicial de la epoca 0
        if epoch == 0:
            prediccion_inicial = prediccion.copy()
            
        # calcular precision en cada epoca
        pred_binaria = np.where(prediccion >= 0.5, 1, 0)
        precision_actual = np.mean(pred_binaria == Y) * 100
        
        historial_loss.append(loss)
        historial_precision.append(precision_actual)
        
        # guardar foto del estado cada 50 epocas o en la primera
        if epoch == 0 or (epoch + 1) % 50 == 0:
            registros_epocas.append({
                "Epoca": epoch + 1,
                "Error (Loss)": round(loss, 6),
                "Precision (%)": round(precision_actual, 2),
                "Promedio Prediccion Sao Paulo (Clase 0)": round(float(np.mean(prediccion[0:N])), 4),
                "Promedio Prediccion Tokio (Clase 1)": round(float(np.mean(prediccion[N:])), 4)
            })

        bar.progress((epoch + 1) / epochs)

    st.success("Entrenamiento completado con exito!")
    
    # metricas finales
    col1, col2 = st.columns(2)
    col1.metric(label="Precision Final", value=f"{historial_precision[-1]:.2f}%")
    col2.metric(label="Error Final (MSE Loss)", value=f"{historial_loss[-1]:.6f}")

    st.write("### A. Grafica de Curva de Aprendizaje (Loss vs. Epoca)")
    df_loss = pd.DataFrame({"Loss": historial_loss})
    st.line_chart(df_loss)

    st.write("### B. Tabla de Evolucion del Aprendizaje en Epocas Clave")
    st.dataframe(pd.DataFrame(registros_epocas))

    st.write("### C. Comparativa: Antes vs. Despues del Entrenamiento")
    st.write("Las salidas se acercan a 0 para Sao Paulo y a 1 para Tokio conforme la red aprende:")
    
    df_comparativa = pd.DataFrame({
        "Latitud": X_raw[:, 0],
        "Longitud": X_raw[:, 1],
        "Clase Real": Y.flatten(),
        "Prediccion Inicial (Epoca 1)": np.round(prediccion_inicial.flatten(), 4),
        "Prediccion Final (Epoca 200)": np.round(prediccion.flatten(), 4)
    })
    
    # mostrar solo las primeras 5 y ultimas 5 filas
    st.dataframe(pd.concat([df_comparativa.head(5), df_comparativa.tail(5)]))