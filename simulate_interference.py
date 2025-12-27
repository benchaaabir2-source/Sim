import numpy as np
import matplotlib.pyplot as plt

# ==================================================
# PARAMÈTRES PHYSIQUES
# ==================================================
I0 = 1.0
w = 100e-9          # largeur de fente
d = 450e-9          # séparation
lam = 2.5e-12
L = 10.0
L1 = 0.25

# Axe spatial
x = np.linspace(-5e-4, 5e-4, 16000)

# ==================================================
# OUTILS
# ==================================================
def sinc2(u):
    return (np.sinc(u / np.pi))**2


# ==================================================
# CAS GÉNÉRAL (d ≠ 0)
# ==================================================
x_shift = L * (d / (2 * L1))

phi = 2 * np.pi * d * x / (lam * L)
phi1 = (2 * np.pi / lam) * (d / 2) * (x / L)

I_single = I0 * sinc2(np.pi * w * x / (lam * L))

I1 = I0 * sinc2(np.pi * w * (x - x_shift) / (lam * L))
I2 = I0 * sinc2(np.pi * w * (x + x_shift) / (lam * L))
I_sum = I1 + I2


# ==================================================
# FONCTION DE TRACE
# ==================================================
def plot_all(V, filename):

    # ----- interférence standard corrigée
    I_interf = I1 + I2 + 2 * V * np.sqrt(I1 * I2) * np.cos(phi)

    # ----- ton modèle à modulation locale
    I1_mod = I1 * (1 + V * np.cos(2 * phi1))
    I2_mod = I2 * (1 + V * np.cos(2 * phi1))
    I_mod_sum = I1_mod + I2_mod

    # ----- courbes multiplicatives
    I_single_3x = 3 * I_single
    I_sum_2x = 2 * I_sum

    # ==================================================
    # CAS d = 0
    # ==================================================
    d0 = 0.0
    x_shift_0 = 0.0

    # largeur normale
    I_d0 = I0 * sinc2(np.pi * w * x / (lam * L))

    # largeur divisée par 2
    w_half = w / 2
    I_d0_half = I0 * sinc2(np.pi * w_half * x / (lam * L))

    # ==================================================
    # PLOT
    # ==================================================
    plt.figure(figsize=(13, 7))

    # --- cas général
    plt.plot(x*1e6, I_single, label="1 fente (diffraction)")
    plt.plot(x*1e6, I1, label="I₁")
    plt.plot(x*1e6, I2, label="I₂")
    plt.plot(x*1e6, I_sum, "--", label="I₁ + I₂")

    plt.plot(x*1e6, I_interf, label=f"Interférence standard (V={V})")

    plt.plot(x*1e6, I1_mod, label="I₁ modulé")
    plt.plot(x*1e6, I2_mod, label="I₂ modulé")
    plt.plot(x*1e6, I_mod_sum, label="Somme modulations")

    # multiplicateurs
    plt.plot(x*1e6, I_single_3x, ":k", label="3 × diffraction (1 fente)")
    plt.plot(x*1e6, I_sum_2x, ":r", label="2 × (I₁ + I₂)")

    # --- CAS d = 0
    plt.plot(x*1e6, I_d0, "--", linewidth=2,
             label="d = 0 (w normal)")

    plt.plot(x*1e6, I_d0_half, "--", linewidth=2,
             label="d = 0 (w / 2)")

    # --------------------------------------------------
    plt.xlabel("x (µm)")
    plt.ylabel("Intensité (non normalisée)")
    plt.title(f"Modèle complet – V = {V}")
    plt.grid(True)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.show()


# ==================================================
# GÉNÉRATION DES FIGURES
# ==================================================
plot_all(V=1.0, filename="result_V1_full.png")
plot_all(V=0.5, filename="result_V05_full.png")
