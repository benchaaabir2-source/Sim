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

# Décalage géométrique des deux fentes
x_shift = L * (d / (2 * L1))

# Phase d’interférence
phi = 2 * np.pi * d * x / (lam * L)

# sinc²
def sinc2(u):
    return (np.sinc(u / np.pi))**2

# ------------------------------------------------
# 1) Diffraction par UNE seule fente
# ------------------------------------------------
I_single = I0 * sinc2(np.pi * w * x / (lam * L))

# ------------------------------------------------
# 2) Deux fentes (sans interférence)
# ------------------------------------------------
I1 = I0 * sinc2(np.pi * w * (x - x_shift) / (lam * L))
I2 = I0 * sinc2(np.pi * w * (x + x_shift) / (lam * L))
I_sum = I1 + I2

# ------------------------------------------------
# 3) Deux fentes AVEC interférence
# ------------------------------------------------
I_interf = I1 + I2 + np.sqrt(I1 * I2) * np.cos(phi)

# Normalisation commune
norm = max(I_single.max(), I_sum.max(), I_interf.max())
I_single /= norm
I_sum /= norm
I_interf /= norm

# ------------------------------------------------
# PLOT
# ------------------------------------------------
plt.figure(figsize=(11, 5))

plt.plot(x * 1e6, I_single, label="1 fente (diffraction sinc²)", linewidth=2)
plt.plot(x * 1e6, I_sum, "--", label="2 fentes : I₁ + I₂ (sans interférence)")
plt.plot(x * 1e6, I_interf, label="2 fentes : I₁ + I₂ + √(I₁I₂) cosφ")

plt.xlabel("x (µm)")
plt.ylabel("Intensité normalisée")
plt.title("Diffraction et interférences — comparaison physique")
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.savefig("result.png", dpi=200)
