
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
L1 = 0.25

V = 0.5  # visibilité

# Axe spatial
x = np.linspace(-5e-4, 5e-4, 16000)

# Décalage géométrique
x_shift = L * (d / (2 * L1))

# Phase principale
phi = 2 * np.pi * d * x / (lam * L)

# Phase locale
phi1 = (2 * np.pi / lam) * (d / 2) * (x / L)

# --------------------------------------------------
# sinc²
# --------------------------------------------------
def sinc2(u):
    return (np.sinc(u / np.pi))**2

# --------------------------------------------------
# Diffraction seule
# --------------------------------------------------
I_single = I0 * sinc2(np.pi * w * x / (lam * L))

# --------------------------------------------------
# Deux fentes (sans interférence)
# --------------------------------------------------
I1 = I0 * sinc2(np.pi * w * (x - x_shift) / (lam * L))
I2 = I0 * sinc2(np.pi * w * (x + x_shift) / (lam * L))
I_sum = I1 + I2

# --------------------------------------------------
# Interférence standard (référence)
# --------------------------------------------------
I_interf = I1 + I2 + np.sqrt(I1 * I2) * np.cos(phi)

# --------------------------------------------------
# TES NOUVELLES FORMULES (avec SIN)
# --------------------------------------------------

# modulation sur I1
I1_mod = I1 * (1 + V * np.sin(2 * phi1))

# modulation sur I2 (symétrique)
I2_mod = I2 * (1 + V * np.sin(-2 * phi1))

# somme des deux modulations
I_mod_sum = I1_mod + I2_mod

# --------------------------------------------------
# Normalisation globale
# --------------------------------------------------
all_curves = [
    I_single, I_sum, I_interf,
    I1_mod, I2_mod, I_mod_sum
]

norm = max(c.max() for c in all_curves)

I_single /= norm
I_sum /= norm
I_interf /= norm
I1_mod /= norm
I2_mod /= norm
I_mod_sum /= norm

# --------------------------------------------------
# PLOT
# --------------------------------------------------
plt.figure(figsize=(12, 6))

plt.plot(x * 1e6, I_single, label="1 slit (diffraction)")
plt.plot(x * 1e6, I_sum, "--", label="2 slits: I₁ + I₂")
plt.plot(x * 1e6, I_interf, label="Standard interference √(I₁I₂) cosφ")

plt.plot(x * 1e6, I1_mod, label="I₁ · (1 + V sin(2φ₁))")
plt.plot(x * 1e6, I2_mod, label="I₂ · (1 + V sin(2φ₁))")
plt.plot(x * 1e6, I_mod_sum, label="Sum of modulated terms")

plt.xlabel("x (µm)")
plt.ylabel("Normalized intensity")
plt.title("Diffraction and modulation model (sin version)")
plt.grid(True)
plt.legend(fontsize=9)
plt.tight_layout()

plt.savefig("result_sin_model.png", dpi=200)
plt.show()
