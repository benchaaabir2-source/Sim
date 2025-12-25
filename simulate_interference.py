import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# === Paramètres (valeurs fournies) ===
I0 = 1.0
w  = 100e-9        # 100 nm
d  = 450e-9        # 450 nm
lam= 2.51e-12      # 2.51 pm (électrons ~200 keV)
L0 = 10.0          # m
L1 = 0.25          # m

# domaine spatial (±500 µm)
x_min = -5e-4
x_max =  5e-4
n_points = 12000
x = np.linspace(x_min, x_max, n_points)

# sinc safe
def sinc(u):
    return np.where(u == 0.0, 1.0, np.sin(u) / u)

# arguments pour I1 et I2
arg1 = (np.pi * w / lam) * (x / L0 - d / (2 * L1))
arg2 = (np.pi * w / lam) * (x / L0 + d / (2 * L1))

I1 = I0 * sinc(arg1)**2
I2 = I0 * sinc(arg2)**2

# Cas 1 : sans cos (somme incohérente)
I_no_cos = I1 + I2

# Cas 2 : avec cos (interférence cohérente)
phi = (2 * np.pi * d / (lam * L0)) * x
I_with_cos = I1 + I2 + 2 * np.sqrt(I1 * I2) * np.cos(phi)

# Normalisation pour affichage
I_no_cos_n = I_no_cos / np.max(I_no_cos)
I_with_cos_n = I_with_cos / np.max(I_with_cos)

# === Tracé 1 : sans cos ===
plt.figure(figsize=(9,4.5))
plt.plot(x*1e6, I_no_cos_n)
plt.xlabel("x (µm)")
plt.ylabel("Intensité normalisée")
plt.title("Sans cos : I(x) = I1 + I2")
plt.grid(True)
plt.tight_layout()
plt.savefig("simulation_no_cos.png", dpi=200)
plt.show()

# === Tracé 2 : avec cos ===
plt.figure(figsize=(9,4.5))
plt.plot(x*1e6, I_with_cos_n)
plt.xlabel("x (µm)")
plt.ylabel("Intensité normalisée")
plt.title("Avec cos : I(x) = I1 + I2 + 2√(I1 I2) cos(...)")
plt.grid(True)
plt.tight_layout()
plt.savefig("simulation_with_cos.png", dpi=200)
plt.show()

# === Zoom pour voir les franges ===
zoom_half = 150e-6   # ±150 µm
mask = (x >= -zoom_half) & (x <= zoom_half)
plt.figure(figsize=(9,4.5))
plt.plot(x[mask]*1e6, I_with_cos[mask]/np.max(I_with_cos))
plt.xlabel("x (µm)")
plt.ylabel("Intensité normalisée")
plt.title("Zoom (±150 µm) — Avec cos (franges visibles)")
plt.grid(True)
plt.tight_layout()
plt.savefig("simulation_with_cos_zoom.png", dpi=200)
plt.show()

# === Sauvegarde paramètres en CSV ===
params = {
    "I0": I0,
    "w (m)": w,
    "d (m)": d,
    "lambda (m)": lam,
    "L0 (m)": L0,
    "L1 (m)": L1,
    "x_min (m)": x_min,
    "x_max (m)": x_max,
    "n_points": n_points
}
pd.DataFrame.from_dict(params, orient='index', columns=['value']).to_csv("simulation_parameters.csv")

# === Échelles utiles imprimées ===
fringe_spacing = lam * L0 / d       # m
envelope_width = lam * L0 / w       # m
x_shift = d * L1 / L0               # m

print("Fichiers générés: simulation_no_cos.png, simulation_with_cos.png, simulation_with_cos_zoom.png, simulation_parameters.csv")
print(f"Espacement des franges ≈ {fringe_spacing*1e6:.2f} µm")
print(f"Largeur d'enveloppe ≈ {envelope_width*1e6:.2f} µm")
print(f"x_shift (projection) = {x_shift*1e9:.3f} nm")
