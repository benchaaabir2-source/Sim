# double_sinc_interference.py
# Simulation d'interférence de deux lobes sinc² décalés
# Intensité totale exacte : I_tot = I1 + I2 + 2 V sqrt(I1*I2) cos(phi)

import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# Paramètres physiques
# -------------------------
I0 = 1.0         # Intensité maximale d'un lobe
w  = 100e-9      # Largeur de la fente (m)
d  = 450e-9      # Séparation entre fentes (m)
lam= 2.51e-12    # Longueur d'onde (m) → électrons 200 kV
L  = 10.0        # Distance écran (m)
L1 = 0.25        # Distance pour calcul du décalage
V  = 0.5         # Visibilité globale

# Axe x (échelle ±500 µm)
x = np.linspace(-5e-4, 5e-4, 12000)

# Déplacement des lobes
x_shift = L * (d / (2*L1))

# Phase d'interférence
phi = 2 * np.pi * d * x / (lam * L)

# -------------------------
# Fonction sinc²
# -------------------------
def sinc2(u):
    return (np.sinc(u/np.pi))**2

# Intensités individuelles
I1 = I0 * sinc2(np.pi * w * (x - x_shift) / (lam * L))
I2 = I0 * sinc2(np.pi * w * (x + x_shift) / (lam * L))

# Intensité totale exacte avec interférence
I_tot = I1 + I2 + 2 * V * np.sqrt(I1 * I2) * np.cos(phi)

# -------------------------
# Tracé
# -------------------------
plt.figure(figsize=(12,5))
plt.plot(x*1e6, I1, label="I1 (lobe gauche)")
plt.plot(x*1e6, I2, label="I2 (lobe droit)")
plt.plot(x*1e6, I_tot, label="I_tot (interférence exacte)", color='black', linewidth=1.5)
plt.xlabel("x (µm)")
plt.ylabel("Intensité")
plt.title("Interférence de deux lobes sinc² décalés avec V=0.5")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
