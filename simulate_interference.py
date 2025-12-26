
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

# Axe spatial
x = np.linspace(-5e-4, 5e-4, 16000)

# Décalage géométrique
x_shift = L * (d / (2 * L1))

# Phase interférentielle standard
phi = 2 * np.pi * d * x / (lam * L)

# Phase locale
phi1 = (2 * np.pi / lam) * (d / 2) * (x / L)

# --------------------------------------------------
# Définition sinc²
# --------------------------------------------------
def sinc2(u):
    return (np.sinc(u / np.pi))**2

# --------------------------------------------------
# Diffraction simple
# --------------------------------------------------
I_single = I0 * sinc2(np.pi * w * x / (lam * L))

# --------------------------------------------------
# Deux fentes sans interférence
# --------------------------------------------------
I1 = I0 * sinc2(np.pi * w * (x - x_shift) / (lam * L))
I2 = I0 * sinc2(np.pi * w * (x + x_shift) / (lam * L))
I_sum = I1 + I2

# --------------------------------------------------
# Interférence classique
# --------------------------------------------------
I_interf = I1 + I2 + np.sqrt(I1 * I2) * np.cos(phi)

# --------------------------------------------------
# Fonction pour créer les plots selon V
# --------------------------------------------------
def plot_modulation(V, filename):
    # Modulations locales
    I1_mod = I1 * (1 + V * np.cos(2 * phi1))
    I2_mod = I2 * (1 + V * np.cos(-2 * phi1))
    I_mod_sum = I1_mod + I2_mod

    plt.figure(figsize=(12,6))
    
    # Tracer toutes les courbes
    plt.plot(x*1e6, I_single, label="1 fente (diffraction)")
    plt.plot(x*1e6, I_sum, "--", label="2 fentes : I₁ + I₂")
    plt.plot(x*1e6, I_interf, label="Interférence standard √(I₁I₂)")
    plt.plot(x*1e6, I1_mod, label=f"I₁ + I₁·V·cos(2φ₁), V={V}")
    plt.plot(x*1e6, I2_mod, label=f"I₂ + I₂·V·cos(2φ₁), V={V}")
    plt.plot(x*1e6, I_mod_sum, label=f"Somme des modulations, V={V}")

    plt.xlabel("x (µm)")
    plt.ylabel("Intensité (non normalisée)")
    plt.title(f"Modèle avec V={V}")
    plt.grid(True)
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.show()

# --------------------------------------------------
# Créer les deux images
# --------------------------------------------------
plot_modulation(V=1.0, filename="result_V1.png")
plot_modulation(V=0.5, filename="result_V0.5.png")
