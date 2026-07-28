# src/model_monitoring.py

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, chisquare, entropy

def calculate_psi(reference, current, num_buckets=10):
    """
    Calcula el Population Stability Index (PSI) entre la distribución de referencia (train)
    y la distribución actual (producción).
    PSI < 0.1  -> Sin cambio significativo (Verde)
    PSI 0.1-0.2 -> Cambio moderado (Amarillo)
    PSI > 0.2  -> Drift significativo (Rojo)
    """
    ref_clean = reference.dropna()
    curr_clean = current.dropna()
    
    if len(ref_clean) == 0 or len(curr_clean) == 0:
        return 0.0

    percentiles = np.linspace(0, 100, num_buckets + 1)
    buckets = np.percentile(ref_clean, percentiles)
    buckets[0] -= 1e-5
    buckets[-1] += 1e-5

    ref_counts, _ = np.histogram(ref_clean, bins=buckets)
    curr_counts, _ = np.histogram(curr_clean, bins=buckets)

    ref_pct = ref_counts / len(ref_clean)
    curr_pct = curr_counts / len(curr_clean)

    # Reemplazar 0s para evitar división por cero
    ref_pct = np.where(ref_pct == 0, 1e-4, ref_pct)
    curr_pct = np.where(curr_pct == 0, 1e-4, curr_pct)

    psi_value = np.sum((curr_pct - ref_pct) * np.log(curr_pct / ref_pct))
    return round(psi_value, 4)


def calculate_jensen_shannon(reference, current, num_buckets=10):
    """
    Calcula la divergencia Jensen-Shannon entre dos variables numéricas.
    """
    ref_clean = reference.dropna()
    curr_clean = current.dropna()
    
    if len(ref_clean) == 0 or len(curr_clean) == 0:
        return 0.0

    min_val = min(ref_clean.min(), curr_clean.min())
    max_val = max(ref_clean.max(), curr_clean.max())
    bins = np.linspace(min_val, max_val, num_buckets + 1)

    ref_counts, _ = np.histogram(ref_clean, bins=bins)
    curr_counts, _ = np.histogram(curr_clean, bins=bins)

    p = ref_counts / len(ref_clean) + 1e-6
    q = curr_counts / len(curr_clean) + 1e-6

    # Normalizar
    p /= p.sum()
    q /= q.sum()

    m = 0.5 * (p + q)
    js_div = 0.5 * (entropy(p, m) + entropy(q, m))
    return round(js_div, 4)


def run_drift_analysis(df_reference, df_current, numeric_cols, categoric_cols):
    """
    Recorre todas las variables y calcula KS, PSI, JS Divergence y Chi-cuadrado.
    """
    drift_results = []

    # 1. Variables Numéricas (KS Test, PSI, JS)
    for col in numeric_cols:
        if col in df_reference.columns and col in df_current.columns:
            ref_data = df_reference[col].dropna()
            curr_data = df_current[col].dropna()

            # Kolmogorov-Smirnov Test
            ks_stat, p_value = ks_2samp(ref_data, curr_data)
            
            # PSI & JS
            psi_val = calculate_psi(ref_data, curr_data)
            js_val = calculate_jensen_shannon(ref_data, curr_data)

            # Estado según PSI
            if psi_val < 0.1:
                status = "Estable (Verde)"
            elif psi_val < 0.2:
                status = "Alerta (Amarillo)"
            else:
                status = "Drift Crítico (Rojo)"

            drift_results.append({
                'Variable': col,
                'Tipo': 'Numérica',
                'KS Stat': round(ks_stat, 4),
                'p-value': round(p_value, 4),
                'PSI': psi_val,
                'JS Divergence': js_val,
                'Estado': status
            })

    # 2. Variables Categóricas (Chi-cuadrado y PSI simplificado)
    for col in categoric_cols:
        if col in df_reference.columns and col in df_current.columns:
            ref_counts = df_reference[col].value_counts(normalize=True)
            curr_counts = df_current[col].value_counts(normalize=True)
            
            # Alineación de categorías
            all_cats = list(set(ref_counts.index).union(set(curr_counts.index)))
            ref_pct = np.array([ref_counts.get(c, 1e-4) for c in all_cats])
            curr_pct = np.array([curr_counts.get(c, 1e-4) for c in all_cats])
            
            ref_pct /= ref_pct.sum()
            curr_pct /= curr_pct.sum()

            # Chi-Cuadrado test simplificado
            chi_stat, p_val = chisquare(curr_pct * 100, f_exp=ref_pct * 100)
            psi_val = round(np.sum((curr_pct - ref_pct) * np.log(curr_pct / ref_pct)), 4)

            status = "Estable (Verde)" if psi_val < 0.1 else ("Alerta (Amarillo)" if psi_val < 0.2 else "Drift Crítico (Rojo)")

            drift_results.append({
                'Variable': col,
                'Tipo': 'Categórica',
                'KS Stat': np.nan,
                'p-value': round(p_val, 4),
                'PSI': psi_val,
                'JS Divergence': np.nan,
                'Estado': status
            })

    return pd.DataFrame(drift_results)


if __name__ == "__main__":
    print("Módulo de monitoreo de data drift listo para importarse.")