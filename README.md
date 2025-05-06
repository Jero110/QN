Estructura del código

dqn_agent.py: Clase DQNAgent con redes policy_net y target_net, métodos de acción (choose_action), optimización (optimize_model) y actualización de la red objetivo.

models.py: Dos clases de red neuronal: MLPCartPole y MLPLunarLander, adaptadas a las dimensiones de estado de cada entorno.

replay_buffer.py: Implementación de ReplayBuffer circular para almacenar transiciones y muestrear batches.

train_dqn.py: Script principal que:

Carga el entorno pasado por --env.

Inicializa el agente con hiperparámetros especificados.

Ejecuta el bucle de episodios y pasos, guardando el modelo al final.

visualize_model.py / visualize_lunarlander.py: Scripts para cargar pesos entrenados y ejecutar episodios renderizados en pantalla.

figures/: Carpeta donde se exportan las curvas de recompensa durante el entrenamiento.

Hiperparámetros finales utilizados

Entorno

Episodios

LR

Batch size

Gamma

Epsilon Start

Epsilon End

Epsilon Decay

CartPole-v1

500

1e-4

128

0.99

1.0

0.01

0.995

LunarLander-v2

500

1e-4

128

0.99

1.0

0.01

0.995

Curvas de recompensa por episodio

Se muestran las gráficas de recompensa acumulada por episodio, con una media móvil de 100 episodios superpuesta.





Respuestas a ejercicios seleccionados

Parte 2: Ejploración vs. Explotación

Respuesta: Se implementó una estrategia ε-greedy con ε inicial=1.0 que decae exponencialmente hasta 0.01 para balancear exploración y explotación.

Parte 3: Importancia de la red objetivo

Respuesta: La red objetivo estabiliza el entrenamiento evitando que los valores Q cambien demasiado rápido; se actualiza cada 10 episodios copiando pesos de la red principal.

Parte 4: Normalización de recompensas

Respuesta: En LunarLander, se normalizaron las recompensas dividiendo por 200 para mantener magnitudes de gradiente estables.

Desafíos y soluciones

CartPole: Convergencia muy rápida (>195 de media) pero inestabilidad en picos tempranos. 

Solución: Reducir ligeramente la tasa de aprendizaje y añadir decaimiento de ε.

LunarLander: Exploding gradients y recompensas muy variables.

Solución: Normalizar recompensas, aumentar batch size y usar LR más bajo.

Observaciones sobre el rendimiento

CartPole-v1: Converge en ~200 episodios y mantiene media >195 en los últimos 100.

LunarLander-v2: Requiere ~800–1000 episodios para empezar a rendir de forma consistente, con reward medio ~200 al final.
