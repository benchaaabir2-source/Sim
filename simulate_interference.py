import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------
# Paramètres physiques
# --------------------------------------------------
I0 = 1.0
w = 100e-9
d = 450e-9
lam = 2.5e-12
L = 10.0

V = 0.5  # visibilité

# Axe spatial
x = np.linspace(-5e-4, 5e-4, 16000)

# Phase d’interférence globale
phi = 2 * np.pi * d * x / (lam * L)

# Phase locale proposée
phi1 = (2 * np.pi / lam) * (d / 2) * (x / L)

# --------------------------------------------------
# sinc²
# --------------------------------------------------
def sinc2(u):
    return (np.sinc(u / np.pi))**2

# --------------------------------------------------
# Diffraction (UNE seule enveloppe)
# --------------------------------------------------
I_single = I0 * sinc2(np.pi * w * x / (lam * L))

# --------------------------------------------------
# Deux "fentes" sans décalage spatial
# (même enveloppe)
# --------------------------------------------------
I1 = I_single.copy()
I2 = I_single.copy()

# --------------------------------------------------
# Interférence standard (forme canonique)
# --------------------------------------------------
I_interf = I1 + I2 + np.sqrt(I1 * I2) * np.cos(phi)

# --------------------------------------------------
# TES NOUVELLES FORMES
# --------------------------------------------------

# 1) modulation locale sur I1
I1_mod = I1 * (1 + V * np.cos(2 * phi1))

# 2) modulation locale sur I2 (symétrique)
I2_mod = I2 * (1 + V * np.cos(-2 * phi1))

# 3) somme des deux
I_mod_sum = I1_mod + I2_mod

# --------------------------------------------------
# Normalisation commune
# --------------------------------------------------
all_curves = [
    I_single, I_interf,
    I1_mod, I2_mod, I_mod_sum
]

norm = max(c.max() for c in all_curves)

I_single /= norm
I_interf /= norm
I1_mod /= norm
I2_mod /= norm
I_mod_sum /= norm

# --------------------------------------------------
# PLOT
# --------------------------------------------------
plt.figure(figsize=(12, 6))

plt.plot(x * 1e6, I_single, label="Diffraction seule (sinc²)")
plt.plot(x * 1e6, I_interf, label="Interférence standard √(I₁I₂)")

plt.plot(x * 1e6, I1_mod, label="I₁ · (1 + V cos(2φ₁))")
plt.plot(x * 1e6, I2_mod, label="I₂ · (1 + V cos(2φ₁))")
plt.plot(x * 1e6, I_mod_sum, label="Somme des deux modulations")

plt.xlabel("x (µm)")
plt.ylabel("Intensité normalisée")
plt.title("Interférence sans décalage géométrique (phase seule)")
plt.grid(True)
plt.legend(fontsize=9)
plt.tight_layout()

plt.savefig("result.png", dpi=200)
