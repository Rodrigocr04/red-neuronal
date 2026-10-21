# Cambios para Lograr 100% de Precision

1. **Normalizacion de Coordenadas:** Se transformaron las coordenadas GPS (latitud y longitud) a una escala entre 0 y 1.
2. **Pesos Aleatorios Pequeños:** Se usaron valores aleatorios pequeños en lugar de un numero fijo (0.75) para inicializar las conexiones de la red.
3. **Tasa de Aprendizaje (Learning Rate = 0.1):** Se ajusto la velocidad de aprendizaje a 0.1 para un entrenamiento mas estable.
4. **Puntos Fijos y Aleatorios:** Se fijaron los primeros 10 puntos de Sao Paulo y 10 de Tokio para validar con el Excel, completando los 300 puntos totales con datos aleatorios.