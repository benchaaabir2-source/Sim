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

# --------------------------------------------------
# Fonctions
# --------------------------------------------------
def sinc2(u):
    return (np.sinc(u / np.pi))**2

# --------------------------------------------------
# Phase
# --------------------------------------------------
phi = 2 * np.pi * d * x / (lam * L)
phi1 = (2 * np.pi / lam) * (d / 2) * (x / L)

# --------------------------------------------------
# Décalage géométrique (cas 2 fentes)
# --------------------------------------------------
x_shift = L * (d / (2 * L1))

# --------------------------------------------------
# Diffraction
# --------------------------------------------------
I_single = I0 * sinc2(np.pi * w * x / (lam * L))

I1 = I0 * sinc2(np.pi * w * (x - x_shift) / (lam * L))
I2 = I0 * sinc2(np.pi * w * (x + x_shift) / (lam * L))

# --------------------------------------------------
# Interférence standard
# --------------------------------------------------
I_interf = I1 + I2 + np.sqrt(I1 * I2) * np.cos(phi)

# --------------------------------------------------
# TON MODÈLE — avec décalage
# --------------------------------------------------
I1_mod = I1 * (1 + V * np.cos(2 * phi1))
I2_mod = I2 * (1 + V * np.cos(-2 * phi1))
I_mod_sum = I1_mod + I2_mod

# --------------------------------------------------
# NOUVELLES COURBES : UNE SEULE FENTE CENTRÉE
# --------------------------------------------------

# (A) une seule source + modulation
I_single_mod = I_single * (1 + V * np.cos(2 * phi1))

# (B) "double contribution identique" sans décalage
# (équivalent à deux sources superposées sur l’axe)
I_single_double_mod = 2 * I_single * (1 + V * np.cos(2 * phi1))

# --------------------------------------------------
# Normalisation globale
# --------------------------------------------------
all_curves = [
    I_single,
    I1 + I2,
    I_interf,
    I1_mod,
    I2_mod,
    I_mod_sum,
    I_single_mod,
    I_single_double_mod
]

norm = max(c.max() for c in all_curves)

for i in range(len(all_curves)):
    all_curves[i] /= norm

(
    I_single,
    I_sum,
    I_interf,
    I1_mod,
    I2_mod,
    I_mod_sum,
    I_single_mod,
    I_single_double_mod
) = all_curves

# --------------------------------------------------
# PLOT
# --------------------------------------------------
plt.figure(figsize=(12, 6))

plt.plot(x*1e6, I_single, label="1 fente (sinc²)")
plt.plot(x*1e6, I_sum, "--", label="2 fentes : I₁ + I₂")
plt.plot(x*1e6, I_interf, label="Interférence √(I₁I₂)")

plt.plot(x*1e6, I1_mod, label="I₁ + modulation (déplacée)")
plt.plot(x*1e6, I2_mod, label="I₂ + modulation (déplacée)")
plt.plot(x*1e6, I_mod_sum, label="Somme modulée (2 fentes)")

plt.plot(x*1e6, I_single_mod, label="1 fente centrée + modulation")
plt.plot(x*1e6, I_single_double_mod, label="2 contributions centrées + modulation")

plt.xlabel("x (µm)")
plt.ylabel("Intensité normalisée")
plt.title("Comparaison : diffraction, interférences et modulations locales")
plt.grid(True)
plt.legend(fontsize=8)
plt.tight_layout()

plt.savefig("result.png", dpi=200)
