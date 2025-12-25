import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# === Paramètres physiques ===
I0 = 1.0
w  = 100e-9        # 100 nm
d  = 450e-9        # 450 nm
lam= 2.51e-12      # 2.51 pm (200 keV electrons)
L0 = 10.0          # m
L1 = 0.25          # m
V  = 0.8           # visibilité, 0 < V <= 1

# Domaine spatial
x_min = -5e-4      # -500 µm
x_max =  5e-4      # 500 µm
n_points = 12000
x = np.linspace(x_min, x_max, n_points)

# sinc fonction safe
def sinc(u):
    return np.where(u == 0.0, 1.0, np.sin(u)/u)

# --- 1️⃣ Version double sinc² décalée ---
arg1 = (np.pi * w / lam) * (x / L0 - d / (2 * L1))
arg2 = (np.pi * w / lam) * (x / L0 + d / (2 * L1))
I1 = I0 * sinc(arg1)**2
I2 = I0 * sinc(arg2)**2
phi = (2 * np.pi * d / (lam * L0)) * x
I_double_sinc = (I1 + I2) * (1 + V * np.cos(phi))
I_double_sinc_n = I_double_sinc / np.max(I_double_sinc)

# --- 2️⃣ Version enveloppe sinc² centrale ---
I_env = I0 * sinc((np.pi * w / lam) * x / L0)**2
I_env_total = I_env * (1 + V * np.cos(phi))
I_env_total_n = I_env_total / np.max(I_env_total)

# --- Tracé comparatif ---
plt.figure(figsize=(10,5))
plt.plot(x*1e6, I_double_sinc_n, label="Double sinc² décalée")
plt.plot(x*1e6, I_env_total_n, label="Enveloppe sinc² centrale")
plt.xlabel("x (µm)")
plt.ylabel("Intensité normalisée")
plt.title(f"Comparaison des modèles (V={V})")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("simulation_double_vs_env.png", dpi=200)
plt.show()

# --- Zoom sur franges centrales ±150 µm ---
zoom_half = 150e-6
mask = (x >= -zoom_half) & (x <= zoom_half)
plt.figure(figsize=(10,4.5))
plt.plot(x[mask]*1e6, I_double_sinc_n[mask], label="Double sinc² décalée")
plt.plot(x[mask]*1e6, I_env_total_n[mask], label="Enveloppe sinc² centrale")
plt.xlabel("x (µm)")
plt.ylabel("Intensité normalisée")
plt.title("Zoom ±150 µm — Franges visibles")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("simulation_double_vs_env_zoom.png", dpi=200)
plt.show()

# --- Sauvegarde paramètres CSV ---
params = {
    "I0": I0,
    "w (m)": w,
    "d (m)": d,
    "lambda (m)": lam,
    "L0 (m)": L0,
    "L1 (m)": L1,
    "V": V,
    "x_min (m)": x_min,
    "x_max (m)": x_max,
    "n_points": n_points
}
pd.DataFrame.from_dict(params, orient='index', columns=['value']).to_csv("simulation_parameters.csv")

# --- Échelles importantes ---
fringe_spacing = lam * L0 / d       # m
envelope_width = lam * L0 / w       # m
x_shift = d * L1 / L0               # m

print("Fichiers générés :")
print(" - simulation_double_vs_env.png")
print(" - simulation_double_vs_env_zoom.png")
print(" - simulation_parameters.csv")
print(f"Espacement des franges ≈ {fringe_spacing*1e6:.2f} µm")
print(f"Largeur d'enveloppe ≈ {envelope_width*1e6:.2f} µm")
print(f"x_shift (projection) = {x_shift*1e9:.3f} nm")
