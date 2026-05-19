import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import log, sqrt, exp, pi, erf

#################### On définit les differentes fonctions nécessaires ####################

def N(x):
    return 0.5 * (1 + erf(x / sqrt(2)))

def d1(S0, K, r, sigma, T, t=0.0):
    return (log(S0/K) + (r + 0.5*sigma**2)*(T-t)) / (sigma*sqrt(T-t))

def d2(S0, K, r, sigma, T, t=0.0):
    return d1(S0, K, r, sigma, T, t) - sigma*sqrt(T-t)

def Vega_BS(S0, K, r, sigma, T, t=0.0):
    d1_ = d1(S0, K, r, sigma, T, t)
    return S0 * sqrt(T - t) * (1/sqrt(2*pi)) * exp(-0.5 * d1_**2)

#################### Fonctions pour Call ####################

def V_BS_Call(S0, K, r, sigma, T, t=0.0):
    return S0 * N(d1(S0, K, r, sigma, T, t)) - K * exp(-r*(T-t)) * N(d2(S0, K, r, sigma, T, t))

def F_Call(M, S0, K, r, sigma, T, t=0.0):
    return V_BS_Call(S0, K, r, sigma, T, t) - M

def newton_Call(M, S0, K, r, T, t=0.0, epsilon=1e-6, max_iter=100):
    # --- test d’arbitrage pour un CALL ---
    borne_inf = max(S0 - K * exp(-r*(T - t)), 0.0)
    borne_sup = S0
    if not (borne_inf <= M <= borne_sup):
        return float("nan")

    sigma = sqrt(2 * abs(log(S0/K) + r*(T-t)) / (T - t))

    while abs(F_Call(M, S0, K, r, sigma, T, t)) > epsilon:
        sigma = sigma - F_Call(M, S0, K, r, sigma, T, t) / Vega_BS(S0, K, r, sigma, T, t)

    return sigma

#################### Fonctions pour Put ####################

def V_BS_Put(S0, K, r, sigma, T, t=0.0):
    return K * exp(-r*(T - t)) * N(-d2(S0, K, r, sigma, T, t)) - S0 * N(-d1(S0, K, r, sigma, T, t))

def F_Put(M, S0, K, r, sigma, T, t=0.0):
    return V_BS_Put(S0, K, r, sigma, T, t) - M

def newton_Put(M, S0, K, r, T, t=0.0, epsilon=1e-6, max_iter=100):
    # --- test d’arbitrage pour un PUT ---
    borne_inf = max(K * exp(-r*(T - t)) - S0, 0.0)
    borne_sup = K * exp(-r*(T - t))
    if not (borne_inf <= M <= borne_sup):
        return float("nan")

    sigma = sqrt(2 * abs(log(S0/K) + r*(T - t)) / (T - t))

    while abs(F_Put(M, S0, K, r, sigma, T, t)) > epsilon:
        sigma = sigma - F_Put(M, S0, K, r, sigma, T, t) / Vega_BS(S0, K, r, sigma, T, t)

    return sigma

#################### Question 1 ####################

S0 = 5430.3
T = 4/12
r = 0.05
K_list = [5125, 5225, 5325, 5425, 5525, 5625, 5725, 5825]
M_list = [475, 405, 340, 280.5, 226, 179.5, 139, 105]

# --- Calcul des volatilités ---
vols = [newton_Call(M, S0, K, r, T) for K, M in zip(K_list, M_list)]

df = pd.DataFrame({"Strike": K_list, "Prix marché": M_list, "Vol implicite": vols})
print(df)

# --- Plot du smile ---
plt.figure(figsize=(7,5))
plt.plot(K_list, vols, marker="o", linestyle="-", color="b")
plt.xlabel("Strike K")
plt.ylabel("Volatilité implicite")
plt.title("Smile de volatilité implicite")
 
plt.grid(True)
plt.show()

#################### Question 2 et 3 ####################

# ====== Lecture du fichier ======
link = "sp-index.txt"
A = np.loadtxt(link, skiprows=1)

S0 = 1260.36
q = 0.0217

T   = A[:, 0]
K   = A[:, 1]
Cb, Ca = A[:, 2], A[:, 3]
Pb, Pa = A[:, 4], A[:, 5]
rcol   = A[:, 6] * 0.01

Call_mid = (Cb + Ca) / 2.0
Put_mid  = (Pb + Pa) / 2.0

# On cherche toutes les maturités uniques dans le fichier
maturites_uniques = np.unique(T)

#pour le 3D
K_3D = []
T_3D = []
Sigma_3D = []
 
#################### 2D : Smiles Call séparés ####################
n = len(maturites_uniques)
ncols = 3
nrows = int(np.ceil(n / ncols))

fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4 * nrows))
axes = axes.flatten()

for i, k in enumerate(maturites_uniques):
    indices = np.where(T == k)
    K_T = K[indices]
    r_T = rcol[indices]
    V_marche_T = Call_mid[indices]

    sigma_imp_T = np.array([
        newton_Call(V, S0 * exp(-q * k), K_j, r_j, k)
        for K_j, r_j, V in zip(K_T, r_T, V_marche_T)
    ])

    K_T_valides = K_T[~np.isnan(sigma_imp_T)]
    sigma_imp_valides = sigma_imp_T[~np.isnan(sigma_imp_T)]

    ax = axes[i]
    ax.plot(K_T_valides, sigma_imp_valides, 'o-', markersize=3)
    ax.set_title(f"T = {k:.3f}")
    ax.set_xlabel("Strike K")
    ax.set_ylabel("Volatilité Implicite")
    ax.grid(True)

fig.suptitle("Smiles Call 2D séparés", fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.97])


#################### 2D : Smiles Call superposés ####################
plt.figure()
for k in maturites_uniques:
    indices = np.where(T == k)
    K_T = K[indices]
    r_T = rcol[indices]
    V_marche_T = Call_mid[indices]

    sigma_imp_T = np.array([
        newton_Call(V, S0 * exp(-q * k), K_j, r_j, k)
        for K_j, r_j, V in zip(K_T, r_T, V_marche_T)
    ])

    K_T_valides = K_T[~np.isnan(sigma_imp_T)]
    sigma_imp_valides = sigma_imp_T[~np.isnan(sigma_imp_T)]

    plt.plot(K_T_valides, sigma_imp_valides, 'o-', markersize=3, label=f"T = {k:.3f}")

plt.title('Smiles Call 2D')
plt.xlabel('Strike K')
plt.ylabel('Volatilité Implicite')
plt.legend(title='Maturité')
plt.grid(True)


#################### 3D : Smiles Call séparés ####################

fig3d, axes3d = plt.subplots(nrows, ncols, subplot_kw={'projection': '3d'}, figsize=(16, 4 * nrows))
axes3d = axes3d.flatten()

for i, k in enumerate(maturites_uniques):
    indices = np.where(T == k)
    K_T = K[indices]
    r_T = rcol[indices]
    V_marche_T = Call_mid[indices]

    sigma_imp_T = np.array([
        newton_Call(V, S0 * exp(-q * k), K_j, r_j, k)
        for K_j, r_j, V in zip(K_T, r_T, V_marche_T)
    ])

    K_T_valides = K_T[~np.isnan(sigma_imp_T)]
    sigma_imp_valides = sigma_imp_T[~np.isnan(sigma_imp_T)]

    ax3 = axes3d[i]
    ax3.plot(K_T_valides, np.full_like(K_T_valides, k, dtype=float), sigma_imp_valides)
    ax3.set_title(f"T = {k:.3f}")
    ax3.set_xlabel('Strike K')
    ax3.set_ylabel('Maturité T')
    ax3.set_zlabel('Volatilité Implicite')
    ax3.view_init(elev=20, azim=-110)

fig3d.suptitle("Smiles Call 3D séparés", fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.95])


#################### 3D : Smiles Call superposés ####################
for k in maturites_uniques:
    indices = np.where(T == k)
    K_T = K[indices]
    r_T = rcol[indices]
    V_marche_T = Call_mid[indices]

    sigma_imp_T = np.array([
        newton_Call(V, S0 * exp(-q * k), K_j, r_j, k)
        for K_j, r_j, V in zip(K_T, r_T, V_marche_T)
    ])

    K_T_valides = K_T[~np.isnan(sigma_imp_T)]
    sigma_imp_valides = sigma_imp_T[~np.isnan(sigma_imp_T)]

    K_3D.extend(K_T_valides)
    T_3D.extend([k] * len(K_T_valides))
    Sigma_3D.extend(sigma_imp_valides)

K_3D = np.array(K_3D)
T_3D = np.array(T_3D)
Sigma_3D = np.array(Sigma_3D)

fig = plt.figure(figsize=(10, 12))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(K_3D, T_3D, Sigma_3D)
ax.set_title('Smiles Call 3D')
ax.set_xlabel('Strike K')
ax.set_ylabel('Maturité T')
ax.set_zlabel('Volatilité Implicite')
ax.view_init(elev=20, azim=-110)

plt.show()

#################### 2D : Smiles Put séparés ####################
n = len(maturites_uniques)
ncols = 3
nrows = int(np.ceil(n / ncols))

fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4 * nrows))
axes = axes.flatten()

for i, k in enumerate(maturites_uniques):
    indices = np.where(T == k)
    K_T = K[indices]
    r_T = rcol[indices]
    V_marche_T = Put_mid[indices]

    sigma_imp_T = np.array([
        newton_Put(V, S0 * exp(-q * k), K_j, r_j, k)
        for K_j, r_j, V in zip(K_T, r_T, V_marche_T)
    ])

    K_T_valides = K_T[~np.isnan(sigma_imp_T)]
    sigma_imp_valides = sigma_imp_T[~np.isnan(sigma_imp_T)]

    ax = axes[i]
    ax.plot(K_T_valides, sigma_imp_valides, 'o-', markersize=3)
    ax.set_title(f"T = {k:.3f}")
    ax.set_xlabel("Strike K")
    ax.set_ylabel("Volatilité Implicite")
    ax.grid(True)

fig.suptitle("Smiles Put 2D séparés", fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.97])


#################### 2D : Smiles Put superposés ####################
plt.figure()
for k in maturites_uniques:
    indices = np.where(T == k)
    K_T = K[indices]
    r_T = rcol[indices]
    V_marche_T = Put_mid[indices]

    sigma_imp_T = np.array([
        newton_Put(V, S0 * exp(-q * k), K_j, r_j, k)
        for K_j, r_j, V in zip(K_T, r_T, V_marche_T)
    ])

    K_T_valides = K_T[~np.isnan(sigma_imp_T)]
    sigma_imp_valides = sigma_imp_T[~np.isnan(sigma_imp_T)]

    plt.plot(K_T_valides, sigma_imp_valides, 'o-', markersize=3, label=f"T = {k:.3f}")

plt.title('Smiles Put 2D')
plt.xlabel('Strike K')
plt.ylabel('Volatilité Implicite')
plt.legend(title='Maturité')
plt.grid(True)


#################### 3D : Smiles Put séparés ####################

fig3d, axes3d = plt.subplots(nrows, ncols, subplot_kw={'projection': '3d'}, figsize=(16, 4 * nrows))
axes3d = axes3d.flatten()

for i, k in enumerate(maturites_uniques):
    indices = np.where(T == k)
    K_T = K[indices]
    r_T = rcol[indices]
    V_marche_T = Put_mid[indices]

    sigma_imp_T = np.array([
        newton_Put(V, S0 * exp(-q * k), K_j, r_j, k)
        for K_j, r_j, V in zip(K_T, r_T, V_marche_T)
    ])

    K_T_valides = K_T[~np.isnan(sigma_imp_T)]
    sigma_imp_valides = sigma_imp_T[~np.isnan(sigma_imp_T)]

    ax3 = axes3d[i]
    ax3.plot(K_T_valides, np.full_like(K_T_valides, k, dtype=float), sigma_imp_valides)
    ax3.set_title(f"T = {k:.3f}")
    ax3.set_xlabel('Strike K')
    ax3.set_ylabel('Maturité T')
    ax3.set_zlabel('Volatilité Implicite')
    ax3.view_init(elev=20, azim=-110)

fig3d.suptitle("Smiles Put 3D séparés", fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.95])


#################### 3D : Smiles Put superposés ####################
K_3D = []
T_3D = []
Sigma_3D = []

for k in maturites_uniques:
    indices = np.where(T == k)
    K_T = K[indices]
    r_T = rcol[indices]
    V_marche_T = Put_mid[indices]

    sigma_imp_T = np.array([
        newton_Put(V, S0 * exp(-q * k), K_j, r_j, k)
        for K_j, r_j, V in zip(K_T, r_T, V_marche_T)
    ])

    K_T_valides = K_T[~np.isnan(sigma_imp_T)]
    sigma_imp_valides = sigma_imp_T[~np.isnan(sigma_imp_T)]

    K_3D.extend(K_T_valides)
    T_3D.extend([k] * len(K_T_valides))
    Sigma_3D.extend(sigma_imp_valides)

K_3D = np.array(K_3D)
T_3D = np.array(T_3D)
Sigma_3D = np.array(Sigma_3D)

fig = plt.figure(figsize=(10, 12))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(K_3D, T_3D, Sigma_3D)
ax.set_title('Smiles Put 3D')
ax.set_xlabel('Strike K')
ax.set_ylabel('Maturité T')
ax.set_zlabel('Volatilité Implicite')
ax.view_init(elev=20, azim=-110)

plt.show()

#################### Question 4 ####################

S_chapeau = []
T_j = []

for tj in maturites_uniques:
    idx = np.where(T == tj)[0]
    Kj  = K[idx]
    rj  = rcol[idx]
    Cj  = Call_mid[idx]
    Pj  = Put_mid[idx]
    # moyenne sur tous les strikes disponibles à maturité tj
    S_chapeau_j = np.mean(Cj - Pj + np.exp(-rj * tj) * Kj)
    S_chapeau.append(S_chapeau_j)
    T_j.append(tj)

S_chapeau = np.array(S_chapeau, dtype=float)
T_j   = np.array(T_j,   dtype=float)

x = T_j.astype(float)                 
y = np.log(S_chapeau).astype(float)  

# Estimation via np.polyfit (moindres carrés) - regression linéaire
beta1, beta2 = np.polyfit(T_j, y, 1)
q_reg  = -beta1
S0_reg = np.exp(beta2)

print("\n Estimation (regression lineaire) ")
print(f"S0_reg = {S0_reg:.6f}")
print(f"q_reg  = {q_reg:.6f}") 
   
# Tracé de la droite correspondante
plt.figure()
plt.scatter(x, y, s=30, label='(T_j, ln (S_chapeau))')
xline = np.linspace(x.min(), x.max(), 200)
plt.plot(xline, beta1*xline + beta2, lw=2, label='Régression')
plt.xlabel('Maturité T'); plt.ylabel('ln(S_chapeau)')
plt.title('Régression linéaire')
plt.grid(True); plt.legend(); plt.show()

# formules théoriques
n   = len(x)
Sx  = np.sum(x)
Sy  = np.sum(y)
Sxx = np.sum(x*x)
Sxy = np.sum(x*y)

# dPhi/dbeta1 = 0, dPhi/dbeta2 = 0
den = n*Sxx - Sx*Sx
beta1 = (n*Sxy - Sx*Sy) / den          
beta2 = (Sy - beta1*Sx) / n            

# Paramètres demandés
q_est  = -beta1
S0_est = np.exp(beta2)

print("\n Estimation par formules théoriques ")
print(f"beta1 = {beta1:.8f}  donc  q = {-beta1:.8f}")
print(f"beta2 = {beta2:.8f}  donc  S0 = {S0_est:.8f}")

#################### Question 5 ####################

# La premiere ligne du fichier est inutile
Data_Google = pd.read_excel("GoogleOrig.xlsx",skiprows=1)

T = Data_Google["Time"].to_numpy(float)
K = Data_Google["Strike"].to_numpy(float)
Call_prix = Data_Google["Call Price"].to_numpy(float)

S0 = 591.66
q_list = [0.00, 0.01, 0.05, 0.10, 0.20, 0.30]
r  = 0.06

# Maturités uniques
maturites_uniques = np.unique(T)

n = len(q_list)
ncols = 3
nrows = int(np.ceil(n / ncols))

fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4 * nrows))
axes = axes.flatten()

for i, q in enumerate(q_list):
    ax = axes[i]
    for k in maturites_uniques:
        indices = np.where(T == k)
        K_T = K[indices]
        V_T = Call_prix[indices]

        sigma_imp_T = np.array([
            newton_Call(V, S0 * np.exp(-q * k), K_j, r, k)
            for K_j, V in zip(K_T, V_T)
        ])

        # Nettoyage NaN
        K_T_valides = K_T[~np.isnan(sigma_imp_T)]
        sigma_imp_valides = sigma_imp_T[~np.isnan(sigma_imp_T)]

        # Tracé des smiles superposés
        ax.plot(K_T_valides, sigma_imp_valides, 'o-', markersize=3, label=f"T = {k:.3f}")

    ax.set_title(f"Smiles Call GoogleOrig - q = {q:.2f}")
    ax.set_xlabel("Strike K")
    ax.set_ylabel("Volatilité implicite")
    ax.grid(True)
    ax.legend(title="Maturité", fontsize=8)

fig.suptitle("Smiles Call GoogleOrig 2D — comparaison selon q", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

#################### Question 6 ####################

fichier = "spx_quotedata.xlsx"
data = pd.read_excel(fichier)

col_date = data.columns[0]
dates = pd.to_datetime(data[col_date].astype(str), dayfirst=True, errors="coerce")

# 1ère date = "aujourd'hui" de la base
date_aujourdhui = dates.iloc[0]

# Maturités en années
T_series = (dates - date_aujourdhui).dt.days / 365.25
T = T_series.to_numpy(float)

# filtrage options vivantes
mask_ok = ~np.isnan(T) & (T > 0)
data = data.loc[mask_ok].reset_index(drop=True)
T = T[mask_ok]

S0 = 3932.69
q = 0.0
r = 0.0255  # taux constant

# passe en NumPy pour un indexage cohérent avec np.where
K  = data.iloc[:, 11].to_numpy(float)
Cb = data.iloc[:, 4].to_numpy(float)
Ca = data.iloc[:, 5].to_numpy(float)
Pb = data.iloc[:, 15].to_numpy(float)
Pa = data.iloc[:, 16].to_numpy(float)

Call_mid = (Cb + Ca) / 2.0
Put_mid  = (Pb + Pa) / 2.0

maturites_uniques = np.unique(np.round(T, 10))
maturites_uniques = maturites_uniques[:13]


#################### 2D : Smiles Call séparés ####################
n = len(maturites_uniques)
ncols = 3
nrows = int(np.ceil(n / ncols))

fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4 * nrows))
axes = axes.flatten()

for i, k in enumerate(maturites_uniques):
    indices = np.isclose(T, k, rtol=0, atol=1e-10)   # <-- au lieu de (T == k)
    K_T = K[indices]
    V_marche_T = Call_mid[indices]

    sigma_imp_T = np.array([
        newton_Call(V, S0 * exp(-q * k), K_j, r, k)
        for K_j, V in zip(K_T, V_marche_T)
    ])

    K_T_valides = K_T[~np.isnan(sigma_imp_T)]
    sigma_imp_valides = sigma_imp_T[~np.isnan(sigma_imp_T)]

    ax = axes[i]
    ax.plot(K_T_valides, sigma_imp_valides, 'o-', markersize=3)
    ax.set_title(f"T = {k:.3f}")
    ax.set_xlabel("Strike K")
    ax.set_ylabel("Volatilité Implicite")
    ax.grid(True)

fig.suptitle("Smiles Call spx_quotedata 2D séparés", fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.97])

# Régression linéaire


S_hat = []
T_j   = []

for tj in maturites_uniques:
    indices = np.isclose(T, tj, rtol=0, atol=1e-10)
    if not np.any(idx):
        continue
    Kj = K[indices]
    Cj = Call_mid[indices]
    Pj = Put_mid[indices]

    # moyenne sur tous les strikes à maturité tj
    S_hat_j = np.mean(Cj - Pj + np.exp(-r * tj) * Kj)
    if S_hat_j > 0:  # sécurité pour le log
        S_hat.append(S_hat_j)
        T_j.append(tj)

S_hat = np.array(S_hat, dtype=float)
T_j   = np.array(T_j,   dtype=float)

# Régression: ln(S_hat) = beta2 + beta1 * T  avec beta1 = -q, beta2 = ln(S0)
y = np.log(S_hat)
x = T_j
beta1, beta2 = np.polyfit(x, y, 1)
q_reg  = -beta1
S0_reg = np.exp(beta2)

print("\nEstimation (régression linéaire via parité Call–Put)")
print(f"S0_reg = {S0_reg:.6f}")
print(f"q_reg  = {q_reg:.6f}")

# formules théoriques
n   = len(x)
Sx  = np.sum(x)
Sy  = np.sum(y)
Sxx = np.sum(x*x)
Sxy = np.sum(x*y)

# dPhi/dbeta1 = 0, dPhi/dbeta2 = 0
den = n*Sxx - Sx*Sx
beta1 = (n*Sxy - Sx*Sy) / den          
beta2 = (Sy - beta1*Sx) / n            

# Paramètres demandés
q_est  = -beta1
S0_est = np.exp(beta2)

print("\n Estimation par formules théoriques ")
print(f"beta1 = {beta1:.8f}  donc  q = {-beta1:.8f}")
print(f"beta2 = {beta2:.8f}  donc  S0 = {S0_est:.8f}")

#################### Question 7 ####################

# Tracé de la droite
plt.figure()
plt.scatter(x, y, s=30, label='(T_j, ln S_hat)')
xline = np.linspace(x.min(), x.max(), 200)
plt.plot(xline, beta1 * xline + beta2, lw=2, label='Régression')
plt.xlabel('Maturité T')
plt.ylabel('ln(S_hat)')
plt.title('Régression linéaire')
plt.grid(True)
plt.legend()
plt.show()

#################### Question 8 ####################

maturites_uniques = np.unique(np.round(T, 10))


#################### 2D : Tout les Smiles Call séparés ####################
n = len(maturites_uniques)
ncols = 3
nrows = int(np.ceil(n / ncols))

fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4 * nrows))
axes = axes.flatten()

for i, k in enumerate(maturites_uniques):
    indices = np.isclose(T, k, rtol=0, atol=1e-10)   # au lieu de (T == k)
    K_T = K[indices]
    V_marche_T = Call_mid[indices]

    sigma_imp_T = np.array([
        newton_Call(V, S0 * exp(-q * k), K_j, r, k)
        for K_j, V in zip(K_T, V_marche_T)
    ])

    K_T_valides = K_T[~np.isnan(sigma_imp_T)]
    sigma_imp_valides = sigma_imp_T[~np.isnan(sigma_imp_T)]

    ax = axes[i]
    ax.plot(K_T_valides, sigma_imp_valides, 'o-', markersize=3)
    ax.set_title(f"T = {k:.3f}")
    ax.set_xlabel("Strike K")
    ax.set_ylabel("Volatilité Implicite")
    ax.grid(True)

fig.suptitle("Smiles Call spx_quotedata 2D séparés", fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.97])

maturites_uniques = maturites_uniques[-12:]


#################### 2D : 12 derniers Smiles Call séparés ####################
n = len(maturites_uniques)
ncols = 3
nrows = int(np.ceil(n / ncols))

fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4 * nrows))
axes = axes.flatten()

for i, k in enumerate(maturites_uniques):
    indices = np.isclose(T, k, rtol=0, atol=1e-10)   # au lieu de (T == k)
    K_T = K[indices]
    V_marche_T = Call_mid[indices]

    sigma_imp_T = np.array([
        newton_Call(V, S0 * exp(-q * k), K_j, r, k)
        for K_j, V in zip(K_T, V_marche_T)
    ])

    K_T_valides = K_T[~np.isnan(sigma_imp_T)]
    sigma_imp_valides = sigma_imp_T[~np.isnan(sigma_imp_T)]

    ax = axes[i]
    ax.plot(K_T_valides, sigma_imp_valides, 'o-', markersize=3)
    ax.set_title(f"T = {k:.3f}")
    ax.set_xlabel("Strike K")
    ax.set_ylabel("Volatilité Implicite")
    ax.grid(True)

fig.suptitle("Smiles Call spx_quotedata 2D séparés", fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.97])

#################### 2D : 12 derniers Smiles Call superposés ####################
plt.figure()
for k in maturites_uniques:
    indices = np.isclose(T, k, rtol=0, atol=1e-10)   # <-- au lieu de (T == k)
    K_T = K[indices]
    V_marche_T = Call_mid[indices]

    sigma_imp_T = np.array([
        newton_Call(V, S0 * exp(-q * k), K_j, r, k)
        for K_j, V in zip(K_T, V_marche_T)
    ])

    K_T_valides = K_T[~np.isnan(sigma_imp_T)]
    sigma_imp_valides = sigma_imp_T[~np.isnan(sigma_imp_T)]

    plt.plot(K_T_valides, sigma_imp_valides, 'o-', markersize=3, label=f"T = {k:.3f}")

plt.title('12 derniers Smiles Call 2D')
plt.xlabel('Strike K')
plt.ylabel('Volatilité Implicite')
plt.legend(title='Maturité')
plt.grid(True)

#################### 2D : Tout les Smiles Call superposés ####################
maturites_uniques = np.unique(np.round(T, 10))
plt.figure()
for k in maturites_uniques:
    indices = np.isclose(T, k, rtol=0, atol=1e-10)   # au lieu de (T == k)
    K_T = K[indices]
    V_marche_T = Call_mid[indices]

    sigma_imp_T = np.array([
        newton_Call(V, S0 * exp(-q * k), K_j, r, k)
        for K_j, V in zip(K_T, V_marche_T)
    ])

    K_T_valides = K_T[~np.isnan(sigma_imp_T)]
    sigma_imp_valides = sigma_imp_T[~np.isnan(sigma_imp_T)]

    plt.plot(K_T_valides, sigma_imp_valides, 'o-', markersize=3, label=f"T = {k:.3f}")

plt.title('Tout les Smiles Call 2D')
plt.xlabel('Strike K')
plt.ylabel('Volatilité Implicite')
plt.legend(title='Maturité')
plt.grid(True)

#################### Question 9 ####################

#################### 2D : 12 premiers Smiles Put séparés ####################
maturites_uniques = maturites_uniques[:12]
n = len(maturites_uniques)
ncols = 3
nrows = int(np.ceil(n / ncols))

fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4 * nrows))
axes = axes.flatten()

for i, k in enumerate(maturites_uniques):
    indices = np.isclose(T, k, rtol=0, atol=1e-10)
    K_T = K[indices]
    V_marche_T = Put_mid[indices]

    sigma_imp_T = np.array([
        newton_Put(V, S0 * exp(-q * k), K_j, r, k)
        for K_j, V in zip(K_T, V_marche_T)
    ])

    K_T_valides = K_T[~np.isnan(sigma_imp_T)]
    sigma_imp_valides = sigma_imp_T[~np.isnan(sigma_imp_T)]

    ax = axes[i]
    ax.plot(K_T_valides, sigma_imp_valides, 'o-', markersize=3)
    ax.set_title(f"T = {k:.3f}")
    ax.set_xlabel("Strike K")
    ax.set_ylabel("Volatilité Implicite")
    ax.grid(True)

fig.suptitle("12 premiers Smiles Put 2D séparés", fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.97])


#################### 2D : 12 premiers Smiles Put superposés ####################
plt.figure()
for k in maturites_uniques:
    iindices = np.isclose(T, k, rtol=0, atol=1e-10)
    K_T = K[indices]
    V_marche_T = Put_mid[indices]

    sigma_imp_T = np.array([
        newton_Put(V, S0 * exp(-q * k), K_j, r, k)
        for K_j, V in zip(K_T, V_marche_T)
    ])

    K_T_valides = K_T[~np.isnan(sigma_imp_T)]
    sigma_imp_valides = sigma_imp_T[~np.isnan(sigma_imp_T)]

    plt.plot(K_T_valides, sigma_imp_valides, 'o-', markersize=3, label=f"T = {k:.3f}")

plt.title('12 premiers Smiles Put 2D superposés')
plt.xlabel('Strike K')
plt.ylabel('Volatilité Implicite')
plt.legend(title='Maturité')
plt.grid(True)

#################### 2D : 12 derniers Smiles Put séparés ####################
maturites_uniques = np.unique(np.round(T, 10))
maturites_uniques = maturites_uniques[-12:]
n = len(maturites_uniques)
ncols = 3
nrows = int(np.ceil(n / ncols))

fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4 * nrows))
axes = axes.flatten()

for i, k in enumerate(maturites_uniques):
    indices = np.isclose(T, k, rtol=0, atol=1e-10)
    K_T = K[indices]
    V_marche_T = Put_mid[indices]

    sigma_imp_T = np.array([
        newton_Put(V, S0 * exp(-q * k), K_j, r, k)
        for K_j, V in zip(K_T, V_marche_T)
    ])

    K_T_valides = K_T[~np.isnan(sigma_imp_T)]
    sigma_imp_valides = sigma_imp_T[~np.isnan(sigma_imp_T)]

    ax = axes[i]
    ax.plot(K_T_valides, sigma_imp_valides, 'o-', markersize=3)
    ax.set_title(f"T = {k:.3f}")
    ax.set_xlabel("Strike K")
    ax.set_ylabel("Volatilité Implicite")
    ax.grid(True)

fig.suptitle("12 derniers Smiles Put 2D séparés", fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.97])


#################### 2D : 12 derniers Smiles Put superposés ####################
plt.figure()
for k in maturites_uniques:
    iindices = np.isclose(T, k, rtol=0, atol=1e-10)
    K_T = K[indices]
    V_marche_T = Put_mid[indices]

    sigma_imp_T = np.array([
        newton_Put(V, S0 * exp(-q * k), K_j, r, k)
        for K_j, V in zip(K_T, V_marche_T)
    ])

    K_T_valides = K_T[~np.isnan(sigma_imp_T)]
    sigma_imp_valides = sigma_imp_T[~np.isnan(sigma_imp_T)]

    plt.plot(K_T_valides, sigma_imp_valides, 'o-', markersize=3, label=f"T = {k:.3f}")

plt.title('12 derniers Smiles Put 2D superposés')
plt.xlabel('Strike K')
plt.ylabel('Volatilité Implicite')
plt.legend(title='Maturité')
plt.grid(True)

#################### Question 10 ####################

fichier = "NEW_spx_quotedata.xlsx"
data = pd.read_excel(fichier, skiprows=1)

col_date = data.columns[0]
dates = pd.to_datetime(data[col_date].astype(str), format="%a %b %d %Y", errors="coerce")

# date d'aujourd'hui fixée au mardi 17 juin 2025
date_aujourdhui = pd.Timestamp(2025, 6, 17)

# Maturités en années
T_series = (dates - date_aujourdhui).dt.days / 365.25
T = T_series.to_numpy(float)

# On filtre
mask_ok = ~np.isnan(T) & (T > 0)
data = data.loc[mask_ok].reset_index(drop=True)
T = T[mask_ok]


r = 0.05       

K  = data.iloc[:, 11].to_numpy(float)
Cb = data.iloc[:, 4].to_numpy(float)
Ca = data.iloc[:, 5].to_numpy(float)
Pb = data.iloc[:, 15].to_numpy(float)
Pa = data.iloc[:, 16].to_numpy(float)

Call_mid = (Cb + Ca) / 2.0
Put_mid  = (Pb + Pa) / 2.0

# On garde les 13 premières maturités distinctes
maturites_uniques = np.unique(np.round(T, 10))

# Régression linéaire pour trouver S_0 et q
S_hat = []
T_j   = []

for tj in maturites_uniques:
    indices = np.isclose(T, tj, rtol=0, atol=1e-10)
    if not np.any(indices):
        continue
    Kj = K[indices]
    Cj = Call_mid[indices]
    Pj = Put_mid[indices]

    S_hat_j = np.mean(Cj - Pj + np.exp(-r * tj) * Kj)
    if S_hat_j > 0:
        S_hat.append(S_hat_j)
        T_j.append(tj)

S_hat = np.array(S_hat, dtype=float)
T_j   = np.array(T_j,   dtype=float)

y = np.log(S_hat)
x = T_j

# polyfit
beta1, beta2 = np.polyfit(x, y, 1)
q_reg  = -beta1
S0_reg = np.exp(beta2)

print("\nEstimation NEW_spx_quotedata.xlsx (régression linéaire via parité Call–Put)")
print(f"S0_reg = {S0_reg:.6f}")
print(f"q_reg  = {q_reg:.6f}")

# Formules fermées
n   = len(x)
Sx  = np.sum(x)
Sy  = np.sum(y)
Sxx = np.sum(x*x)
Sxy = np.sum(x*y)
den = n*Sxx - Sx*Sx
beta1_cf = (n*Sxy - Sx*Sy) / den
beta2_cf = (Sy - beta1_cf*Sx) / n

q_est  = -beta1_cf
S0_est = np.exp(beta2_cf)

print("\n Estimation par formules théoriques NEW_spx_quotedata.xlsx")
print(f"beta1 = {beta1_cf:.8f}  donc  q = {-beta1_cf:.8f}")
print(f"beta2 = {beta2_cf:.8f}  donc  S0 = {S0_est:.8f}")

#################### 2D : Smiles Call séparés ####################
S0 = float(S0_est)
q  = float(q_est)

n = len(maturites_uniques)
ncols = 3
nrows = int(np.ceil(n / ncols))

fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4 * nrows))
axes = axes.flatten()

for i, k in enumerate(maturites_uniques):
    indices = np.isclose(T, k, rtol=0, atol=1e-10)   # <-- au lieu de (T == k)
    K_T = K[indices]
    V_marche_T = Call_mid[indices]

    sigma_imp_T = np.array([
        newton_Call(V, S0 * exp(-q * k), K_j, r, k)
        for K_j, V in zip(K_T, V_marche_T)
    ])

    K_T_valides = K_T[~np.isnan(sigma_imp_T)]
    sigma_imp_valides = sigma_imp_T[~np.isnan(sigma_imp_T)]

    ax = axes[i]
    ax.plot(K_T_valides, sigma_imp_valides, 'o-', markersize=3)
    ax.set_title(f"T = {k:.3f}")
    ax.set_xlabel("Strike K")
    ax.set_ylabel("Volatilité Implicite")
    ax.grid(True)

fig.suptitle("Smiles Call 2D NEW_spx_quotedata.xlsx séparés", fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.97])

#################### 2D : Smiles Call superposés ####################
plt.figure()
for k in maturites_uniques:
    indices = np.isclose(T, k, rtol=0, atol=1e-10)   # <-- au lieu de (T == k)
    K_T = K[indices]
    V_marche_T = Call_mid[indices]

    sigma_imp_T = np.array([
        newton_Call(V, S0 * exp(-q * k), K_j, r, k)
        for K_j, V in zip(K_T, V_marche_T)
    ])

    K_T_valides = K_T[~np.isnan(sigma_imp_T)]
    sigma_imp_valides = sigma_imp_T[~np.isnan(sigma_imp_T)]

    plt.plot(K_T_valides, sigma_imp_valides, 'o-', markersize=3, label=f"T = {k:.3f}")

plt.title(' Smiles Call 2D NEW_spx_quotedata.xlsx')
plt.xlabel('Strike K')
plt.ylabel('Volatilité Implicite')
plt.legend(title='Maturité')
plt.grid(True)

#################### 2D : Smiles Put séparés ####################

fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4 * nrows))
axes = axes.flatten()

for i, k in enumerate(maturites_uniques):
    indices = np.isclose(T, k, rtol=0, atol=1e-10)
    K_T = K[indices]
    V_marche_T = Put_mid[indices]

    sigma_imp_T = np.array([
        newton_Put(V, S0 * exp(-q * k), K_j, r, k)
        for K_j, V in zip(K_T, V_marche_T)
    ])

    K_T_valides = K_T[~np.isnan(sigma_imp_T)]
    sigma_imp_valides = sigma_imp_T[~np.isnan(sigma_imp_T)]

    ax = axes[i]
    ax.plot(K_T_valides, sigma_imp_valides, 'o-', markersize=3)
    ax.set_title(f"T = {k:.3f}")
    ax.set_xlabel("Strike K")
    ax.set_ylabel("Volatilité Implicite")
    ax.grid(True)

fig.suptitle(" Smiles Put 2D NEW_spx_quotedata.xlsx séparés", fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.97])


#################### 2D : Smiles Put superposés ####################
plt.figure()
for k in maturites_uniques:
    iindices = np.isclose(T, k, rtol=0, atol=1e-10)
    K_T = K[indices]
    V_marche_T = Put_mid[indices]

    sigma_imp_T = np.array([
        newton_Put(V, S0 * exp(-q * k), K_j, r, k)
        for K_j, V in zip(K_T, V_marche_T)
    ])

    K_T_valides = K_T[~np.isnan(sigma_imp_T)]
    sigma_imp_valides = sigma_imp_T[~np.isnan(sigma_imp_T)]

    plt.plot(K_T_valides, sigma_imp_valides, 'o-', markersize=3, label=f"T = {k:.3f}")

plt.title(' Smiles Put 2D NEW_spx_quotedata.xlsx superposés')
plt.xlabel('Strike K')
plt.ylabel('Volatilité Implicite')
plt.legend(title='Maturité')
plt.grid(True)


#################### Droite de régression linéaire ####################
plt.figure(figsize=(7, 5))
plt.scatter(T_j, np.log(S_hat), label="Points ln(S_hat)")
x_line = np.linspace(T_j.min(), T_j.max(), 200)
y_line = beta1_cf * x_line + beta2_cf
plt.plot(x_line, y_line, label="Régression linéaire")
plt.title("Régression linéaire NEW_spx_quotedata.xlsx")
plt.xlabel("T (années)")
plt.ylabel("ln(S_hat)")
plt.legend()
plt.grid(True)
plt.show()


#################### Question 11 ####################
#Sur le rapport écrit




