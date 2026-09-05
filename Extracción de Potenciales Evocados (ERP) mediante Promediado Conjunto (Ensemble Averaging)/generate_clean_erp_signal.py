import numpy as np
import matplotlib.pyplot as plt

# Frecuencia de muestreo
fs = 1000  # Hz

# Duración del trial
T = 0.5  # segundos

# Vector de tiempo
t = np.arange(0, T, 1/fs)

# Parámetros de la señal ERP sintética
A = 100
alpha = 12
f = 10  # Hz

# Señal ERP limpia
s = A * t * np.exp(-alpha * t) * np.sin(2 * np.pi * f * t)

# Graficar
plt.figure(figsize=(10, 4))
plt.plot(t * 1000, s)
plt.xlabel("Tiempo (ms)")
plt.ylabel("Amplitud")
plt.title("ERP sintético limpio s(t)")
plt.grid(True)
plt.show()