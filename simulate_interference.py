import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# Paramètres physiques
# -------------------------
I0 = 1.0
w = 100e-9          # largeur de fente
lam = 2.5e-12       # longueur d’onde électron
L = 10.0            # distance écran
d = 0.0             # PAS de décalage (D = 0)
V = 0.5             # visibilité

# Axe spatial
x = np.linspace(-5e-4, 5e-4, 15000)

# -------------------------
# Définition sinc²
# -------------------------
def sinc2(u):
    return (np.sinc(u / np.pi))**2

# -------------------------
# Diffraction simple (une seule fente)
# -------------------------
I1 = I0 * sinc2(np.pi * w * x / (lam * L))

# -------------------------
# Phase (même si d = 0 → phi = 0)
# -------------------------
phi1 = (2 * np.pi / lam) * (d / 2) * (x / L)

# -------------------------
# TA NOUVELLE FORMULE
# -------------------------
I_mod = I1 * (1 + V * np.cos(2 * phi1))

# "double courbe" (juste pour comparaison visuelle)
I_double = 2 * I_mod

# Normalisation visuelle
max_val = max(I_mod.max(), I_double.max())
I_mod /= max_val
I_double /= max_val

# -------------------------
# Plot
# -------------------------
plt.figure(figsize=(10, 4))

plt.plot(x * 1e6, I_mod, label="Single slit + modulation")
plt.plot(x * 1e6, I_double, "--", label="Same curve × 2")

plt.xlabel("x (µm)")
plt.ylabel("Normalized intensity")
plt.title("Single-slit modulation model (D = 0)")
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.savefig("single_slit_modulated_D0.png", dpi=200)
plt.show()
