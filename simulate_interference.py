import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# Paramètres physiques
# -------------------------
I0 = 1.0
w = 100e-9
d = 450e-9
lam = 2.5e-12
L = 10.0
L1 = 0.25

# Axe spatial
x = np.linspace(-5e-4, 5e-4, 15000)

# Décalage géométrique
x_shift = L * (d / (2 * L1))

# Phase d’interférence
phi = 2 * np.pi * d * x / (lam * L)

# sinc²
def sinc2(u):
    return (np.sinc(u / np.pi))**2

# Intensités individuelles
I1 = I0 * sinc2(np.pi * w * (x - x_shift) / (lam * L))
I2 = I0 * sinc2(np.pi * w * (x + x_shift) / (lam * L))

# Intensité totale (ta formule)
I = I1 + I2 + np.sqrt(I1 * I2) * np.cos(phi)

# Normalisation visuelle
I /= np.max(I)

# -------------------------
# Plot
# -------------------------
plt.figure(figsize=(10,4))
plt.plot(x * 1e6, I, label="I = I₁ + I₂ + √(I₁I₂) cosφ")
plt.xlabel("x (µm)")
plt.ylabel("Intensité normalisée")
plt.title("Interférences avec enveloppes sinc²")
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.savefig("result.png", dpi=200)
