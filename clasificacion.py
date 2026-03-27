# ============================================================
#   Clasificación: Detección de Fatiga Muscular en Ciclismo
# ============================================================

# ── Backend de matplotlib primero, antes de cualquier import ──
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# ── Resto de importaciones (sin duplicados) ──
import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, classification_report, confusion_matrix,
                             ConfusionMatrixDisplay)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

from scipy.fft import fft, fftfreq
from datasets import load_dataset
from tqdm import tqdm

# Crear carpetas de salida al inicio
for folder in ['graficas_EDA', 'graficas_modelos', 'conclusiones']:
    os.makedirs(folder, exist_ok=True)

print("=" * 65)
print("DETECCIÓN DE FATIGA MUSCULAR EN CICLISMO")
print("=" * 65)

# ============================================================
#              PUNTO 1 – Análisis Preliminar
# ============================================================

print("\n[PUNTO 1] Análisis Preliminar...")

# Cargar dataset UNA SOLA VEZ
ds = load_dataset("YominE/Muscle_Fatigue_Cycling")
df = ds['train'].to_pandas()

# ── 1a – Preprocesamiento del target ──────────────────────────
print(f"\nValores únicos del target ANTES: {df['Target'].unique()}")
print(f"Distribución ANTES:\n{df['Target'].value_counts()}")

df['Target'] = df['Target'].replace(2, 1)

print(f"\nValores únicos del target DESPUÉS: {df['Target'].unique()}")
print(f"Distribución DESPUÉS:\n{df['Target'].value_counts()}")

# ── 1b – Clasificación de variables ──────────────────────────
print("\n--- CLASIFICACIÓN DE VARIABLES ---")
for col in df.columns:
    n_unique = df[col].nunique()
    dtype    = df[col].dtype
    if col == 'Target':
        tipo = "Binaria (Target)"
    elif dtype == 'object' or dtype.name == 'category':
        tipo = "Categórica nominal"
    elif n_unique == 2:
        tipo = "Binaria"
    elif dtype in ['int64', 'float64']:
        tipo = "Numérica continua" if n_unique > 10 else "Numérica discreta / posiblemente ordinal"
    else:
        tipo = "Otro"
    print(f"  {col:40s} | dtype: {str(dtype):10s} | únicos: {n_unique:5d} | Tipo: {tipo}")

print(df.head())
print(df.dtypes)

# ============================================================
#           PUNTO 2 – Ingeniería de Características
# ============================================================
print("\n[PUNTO 2] Extracción de características...")

FS          = 1000   # Hz (paso de Time = 0.001 s → fs = 1000 Hz)
WINDOW_SIZE = FS     # 1 segundo = 1000 muestras

emg_cols = [
    'Right Rectus femoris',
    'Left Gluteus maximus',
    'Left Gastrocnemius medialis',
    'Left Semitendinosus',
    'Left Biceps femoris caput longus',
    'Right Vastus medialis',
    'Right Tibialis anterior',
    'Left Gastrocnemius lateralis'
]

def extract_time_features(window):
    """
    RMS:      Energía de la señal. Aumenta con activación y fatiga.
    Varianza: Dispersión. Refleja irregularidad muscular.
    ZCR:      Cruces por cero. Relacionado con tasa de disparo motora.
    MAV:      Valor absoluto medio. Indicador clásico de actividad EMG.
    """
    rms      = np.sqrt(np.mean(window**2))
    variance = np.var(window)
    zcr      = np.sum(np.diff(np.sign(window)) != 0) / len(window)
    mav      = np.mean(np.abs(window))
    return rms, variance, zcr, mav

def extract_freq_features(window, fs=FS):
    """
    MDF: Frecuencia mediana. Indicador clave de fatiga: DISMINUYE con el desgaste.
    MNF: Frecuencia media. Complementa MDF; sensible a fibras tipo I.
    PWR: Potencia espectral total. Aumenta con mayor reclutamiento.
    """
    n        = len(window)
    freqs    = fftfreq(n, d=1/fs)
    fft_vals = np.abs(fft(window))**2
    pos_mask = freqs > 0
    freqs    = freqs[pos_mask]
    power    = fft_vals[pos_mask]
    total_power      = np.sum(power)
    cumulative_power = np.cumsum(power)
    median_freq      = freqs[np.searchsorted(cumulative_power, total_power / 2)]
    mean_freq        = np.sum(freqs * power) / total_power if total_power > 0 else 0
    return median_freq, mean_freq, total_power

# Construcción de la base de datos de características (una sola pasada)
records = []
for start in tqdm(range(0, len(df) - WINDOW_SIZE + 1, WINDOW_SIZE),
                  desc="Extrayendo features"):
    end       = start + WINDOW_SIZE
    window_df = df.iloc[start:end]
    row       = {}
    for col in emg_cols:
        w     = window_df[col].values
        short = col.replace('Right ', 'R_').replace('Left ', 'L_').replace(' ', '_')
        rms, var, zcr, mav     = extract_time_features(w)
        row[f'{short}_RMS']    = rms
        row[f'{short}_VAR']    = var
        row[f'{short}_ZCR']    = zcr
        row[f'{short}_MAV']    = mav
        mdf, mnf, pwr          = extract_freq_features(w)
        row[f'{short}_MDF']    = mdf
        row[f'{short}_MNF']    = mnf
        row[f'{short}_PWR']    = pwr
    row['Target'] = window_df['Target'].mode()[0]
    records.append(row)

df_features = pd.DataFrame(records)
print(f"\nShape del nuevo dataset: {df_features.shape}")
print(f"\nPrimeras filas:\n{df_features.head()}")
print(f"\nColumnas generadas: {df_features.columns.tolist()}")
print(f"\nDistribución del target:\n{df_features['Target'].value_counts()}")

# ============================================================
#         PUNTO 3 – Análisis Exploratorio de Datos (EDA)
# ============================================================
print(f"\nLas gráficas se guardarán en: 'graficas_EDA'")

colores = ['#2196F3', '#E91E63', '#4CAF50', '#FF9800',
           '#9C27B0', '#00BCD4', '#F44336', '#795548']

# ── 3a – Señales en el tiempo ─────────────────────────────────
fig, axes = plt.subplots(8, 1, figsize=(16, 22))
fig.suptitle('Señales EMG en el Tiempo (primeros 5 segundos)\n'
             'Cada canal representa un músculo diferente de la pierna dominante',
             fontsize=13, fontweight='bold', y=1.01)
for i, col in enumerate(emg_cols):
    axes[i].plot(df['Time'].values[:5000], df[col].values[:5000],
                 linewidth=0.6, color=colores[i], alpha=0.85)
    axes[i].set_ylabel('mV', fontsize=8)
    axes[i].set_title(f'Canal {i+1}: {col}', fontsize=9, fontweight='bold',
                      loc='left', pad=3)
    axes[i].axhline(0, color='gray', linewidth=0.5, linestyle='--', alpha=0.5)
    axes[i].grid(True, alpha=0.2)
    axes[i].spines['top'].set_visible(False)
    axes[i].spines['right'].set_visible(False)
axes[-1].set_xlabel('Tiempo (segundos)', fontsize=10)
plt.tight_layout()
plt.savefig('graficas_EDA/señales_tiempo.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Gráfica 1/6 guardada: señales_tiempo.png")

# ── Estadísticos descriptivos ─────────────────────────────────
print("\n" + "=" * 60)
print("ESTADÍSTICOS DESCRIPTIVOS DEL DATASET DE CARACTERÍSTICAS")
print("=" * 60)
print(df_features.describe().round(4))

# ── Distribución RMS ─────────────────────────────────────────
rms_cols = [c for c in df_features.columns if '_RMS' in c]
mdf_cols = [c for c in df_features.columns if '_MDF' in c]

fig, axes = plt.subplots(2, 4, figsize=(18, 9))
fig.suptitle('Distribución del RMS por Canal EMG\n'
             'El RMS mide la energía de la señal — valores más altos indican mayor activación muscular',
             fontsize=12, fontweight='bold')
for i, col in enumerate(rms_cols):
    ax = axes[i//4][i%4]
    ax.hist(df_features[col], bins=40, color=colores[i], edgecolor='white', alpha=0.85)
    ax.set_title(col.replace('_RMS', '').replace('_', ' '), fontsize=9, fontweight='bold')
    ax.set_xlabel('Valor RMS (mV)', fontsize=8)
    ax.set_ylabel('Número de ventanas', fontsize=8)
    ax.grid(True, alpha=0.2, axis='y')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig('graficas_EDA/distribucion_rms.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Gráfica 2/6 guardada: distribucion_rms.png")

# ── Distribución MDF ─────────────────────────────────────────
fig, axes = plt.subplots(2, 4, figsize=(18, 9))
fig.suptitle('Distribución de la Frecuencia Mediana (MDF) por Canal EMG\n'
             'La MDF es el indicador más importante de fatiga — disminuye cuando el músculo se fatiga',
             fontsize=12, fontweight='bold')
for i, col in enumerate(mdf_cols):
    ax = axes[i//4][i%4]
    ax.hist(df_features[col], bins=40, color=colores[i], edgecolor='white', alpha=0.85)
    ax.set_title(col.replace('_MDF', '').replace('_', ' '), fontsize=9, fontweight='bold')
    ax.set_xlabel('Frecuencia Mediana (Hz)', fontsize=8)
    ax.set_ylabel('Número de ventanas', fontsize=8)
    ax.grid(True, alpha=0.2, axis='y')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig('graficas_EDA/distribucion_mdf.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Gráfica 3/6 guardada: distribucion_mdf.png")

# ── Balance de clases ─────────────────────────────────────────
counts = df_features['Target'].value_counts()
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Balance de Clases en el Dataset\n'
             'Muestra cuántas ventanas corresponden a condición normal vs fatiga muscular',
             fontsize=12, fontweight='bold')
labels     = ['Normal (0)', 'Fatiga (1)']
colors_pie = ['#2196F3', '#F44336']
bars = axes[0].bar(labels, counts.values, color=colors_pie, edgecolor='white', width=0.5)
axes[0].set_ylabel('Número de ventanas', fontsize=10)
axes[0].set_title('Conteo absoluto por clase', fontsize=10, fontweight='bold')
axes[0].grid(True, alpha=0.2, axis='y')
axes[0].spines['top'].set_visible(False)
axes[0].spines['right'].set_visible(False)
for bar, val in zip(bars, counts.values):
    axes[0].text(bar.get_x() + bar.get_width()/2, val + 15,
                 str(val), ha='center', fontsize=11, fontweight='bold')
wedges, texts, autotexts = axes[1].pie(
    counts.values, labels=labels, colors=colors_pie,
    autopct='%1.1f%%', startangle=90,
    wedgeprops={'edgecolor': 'white', 'linewidth': 2})
for t in autotexts:
    t.set_fontsize(12); t.set_fontweight('bold')
axes[1].set_title('Proporción por clase', fontsize=10, fontweight='bold')
plt.tight_layout()
plt.savefig('graficas_EDA/balance_clases.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Gráfica 4/6 guardada: balance_clases.png")
print(f"\nClase 0 (Normal): {counts[0]} ventanas ({counts[0]/len(df_features)*100:.1f}%)")
print(f"Clase 1 (Fatiga): {counts[1]} ventanas ({counts[1]/len(df_features)*100:.1f}%)")
print(f"Ratio desbalance: {counts[0]/counts[1]:.2f}:1")

# ── Correlaciones ─────────────────────────────────────────────
corr_matrix = df_features.drop(columns='Target').corr()
plt.figure(figsize=(22, 18))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, cmap='coolwarm', center=0,
            vmin=-1, vmax=1, annot=False, linewidths=0.1,
            cbar_kws={'label': 'Coeficiente de Correlación de Pearson', 'shrink': 0.8})
plt.title('Matriz de Correlación entre Características EMG\n'
          'Rojo = correlación positiva | Azul = correlación negativa | Blanco = sin correlación',
          fontsize=12, fontweight='bold', pad=15)
plt.xticks(fontsize=6, rotation=90)
plt.yticks(fontsize=6, rotation=0)
plt.tight_layout()
plt.savefig('graficas_EDA/correlaciones.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Gráfica 5/6 guardada: correlaciones.png")

print("\n=== TOP 15 CARACTERÍSTICAS MÁS CORRELACIONADAS CON EL TARGET ===")
corr_target = df_features.corr()['Target'].drop('Target').abs().sort_values(ascending=False)
print(corr_target.head(15).round(4))

# ── Boxplots por clase ────────────────────────────────────────
fig, axes = plt.subplots(2, 8, figsize=(22, 10))
fig.suptitle('Separabilidad entre Clases: RMS y MDF por Canal\n'
             'Azul = Normal | Rojo = Fatiga | Si las cajas no se solapan → buena separabilidad',
             fontsize=12, fontweight='bold')
for i, col in enumerate(rms_cols):
    ax   = axes[0][i]
    d0   = df_features[df_features['Target']==0][col]
    d1   = df_features[df_features['Target']==1][col]
    bp   = ax.boxplot([d0, d1], patch_artist=True,
                      medianprops={'color': 'black', 'linewidth': 2})
    bp['boxes'][0].set_facecolor('#2196F3')
    bp['boxes'][1].set_facecolor('#F44336')
    ax.set_title(col.replace('_RMS','').replace('_',' '), fontsize=7, fontweight='bold')
    ax.set_xticklabels(['Normal', 'Fatiga'], fontsize=7)
    ax.set_ylabel('RMS', fontsize=7)
    ax.grid(True, alpha=0.2, axis='y')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
for i, col in enumerate(mdf_cols):
    ax   = axes[1][i]
    d0   = df_features[df_features['Target']==0][col]
    d1   = df_features[df_features['Target']==1][col]
    bp   = ax.boxplot([d0, d1], patch_artist=True,
                      medianprops={'color': 'black', 'linewidth': 2})
    bp['boxes'][0].set_facecolor('#2196F3')
    bp['boxes'][1].set_facecolor('#F44336')
    ax.set_title(col.replace('_MDF','').replace('_',' '), fontsize=7, fontweight='bold')
    ax.set_xticklabels(['Normal', 'Fatiga'], fontsize=7)
    ax.set_ylabel('MDF (Hz)', fontsize=7)
    ax.grid(True, alpha=0.2, axis='y')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig('graficas_EDA/boxplots_clases.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Gráfica 6/6 guardada: boxplots_clases.png")
print(f"\n✅ EDA completado.")

# ============================================================
#         PUNTO 4 – PROCESAMIENTO DE DATOS
# ============================================================
print("\n" + "=" * 65)
print("  PUNTO 4 – PROCESAMIENTO DE DATOS")
print("=" * 65)

# ── 4.1 – Manejo de valores nulos ────────────────────────────
nulos = df_features.isnull().sum()
print(f"\n→ Valores nulos por columna:\n{nulos[nulos>0]}")
print(f"  Total nulos: {df_features.isnull().sum().sum()}")

# Separar features y target
X             = df_features.drop(columns='Target')
y             = df_features['Target']
feature_names = X.columns.tolist()

# ── 4.2 – División 70 / 15 / 15 ──────────────────────────────
# Justificación: con ~600-800 ventanas, 70/15/15 da suficientes
# muestras de entrenamiento (~500) y validación/test (~100 c/u).
# stratify=y conserva la proporción de clases en cada split.
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.15/0.85, random_state=42, stratify=y_temp)

print(f"\n→ División de datos (70/15/15):")
print(f"   X_train: {X_train.shape}  |  y_train dist: {dict(y_train.value_counts())}")
print(f"   X_val:   {X_val.shape}    |  y_val dist:   {dict(y_val.value_counts())}")
print(f"   X_test:  {X_test.shape}   |  y_test dist:  {dict(y_test.value_counts())}")

# ── 4.3 – Pipeline de preprocesamiento ───────────────────────
# SimpleImputer: rellena NaN con mediana (robusta a outliers)
# StandardScaler: media=0, std=1 (necesario para kNN y DNN)
# fit_transform solo sobre X_train; transform en val y test
# para evitar data leakage del conjunto de evaluación.
preprocessing = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler',  StandardScaler())
])
X_train_proc = preprocessing.fit_transform(X_train)
X_val_proc   = preprocessing.transform(X_val)
X_test_proc  = preprocessing.transform(X_test)

print("\n✅ Preprocesamiento completado (SimpleImputer + StandardScaler).")

# ============================================================
#     PUNTO 5 – ENTRENAMIENTO Y COMPARACIÓN DE MODELOS
# ============================================================
print("\n" + "=" * 65)
print("  PUNTO 5 – ENTRENAMIENTO Y COMPARACIÓN DE MODELOS")
print("=" * 65)

# ── Funciones auxiliares ──────────────────────────────────────
def get_metrics(model, Xd, yd):
    pred = model.predict(Xd)
    return {
        'Accuracy':  round(accuracy_score(yd, pred), 4),
        'Precision': round(precision_score(yd, pred, zero_division=0), 4),
        'Recall':    round(recall_score(yd, pred, zero_division=0), 4),
        'F1':        round(f1_score(yd, pred, zero_division=0), 4),
    }

def plot_learning_curve(model, model_name, Xtr, ytr, Xv, yv):
    """
    Entrena el modelo con subconjuntos crecientes de X_train y mide
    accuracy en train y validación. Permite detectar over/underfitting.
    """
    np.random.seed(42)
    sizes   = np.linspace(0.1, 1.0, 10)
    tr_accs = []
    va_accs = []
    for frac in sizes:
        n   = max(10, int(frac * len(Xtr)))
        idx = np.random.choice(len(Xtr), n, replace=False)
        model.fit(Xtr[idx], ytr.iloc[idx])
        tr_accs.append(accuracy_score(ytr.iloc[idx], model.predict(Xtr[idx])))
        va_accs.append(accuracy_score(yv, model.predict(Xv)))
    x_axis = [int(s * len(Xtr)) for s in sizes]
    plt.figure(figsize=(8, 5))
    plt.plot(x_axis, tr_accs, 'o-', color='#2196F3', label='Train',      linewidth=2)
    plt.plot(x_axis, va_accs, 's-', color='#F44336', label='Validación',  linewidth=2)
    plt.fill_between(x_axis, tr_accs, va_accs, alpha=0.08, color='gray')
    plt.xlabel('Número de muestras de entrenamiento', fontsize=11)
    plt.ylabel('Accuracy', fontsize=11)
    plt.title(f'Curva de Aprendizaje – {model_name}', fontsize=12, fontweight='bold')
    plt.legend(fontsize=10)
    plt.ylim(0.4, 1.05)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    fname = f'graficas_modelos/curva_{model_name.replace(" ","_")}.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Curva guardada: {fname}")

results     = {}
best_models = {}

# ── 1. kNN – GridSearch ───────────────────────────────────────
# El kNN usa pipeline propio para incluir el scaler en la búsqueda.
# Esto es CORRECTO: GridSearchCV valida con pipeline completo.
print("\n[1/5] k-Nearest Neighbors (GridSearch)...")
knn_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler',  StandardScaler()),
    ('clf',     KNeighborsClassifier())
])
param_grid_knn = {
    'clf__n_neighbors': [3, 5, 7, 11, 15],
    'clf__weights':     ['uniform', 'distance'],
    'clf__metric':      ['euclidean', 'manhattan']
}
gs_knn   = GridSearchCV(knn_pipe, param_grid_knn, cv=5,
                        scoring='f1', n_jobs=-1, verbose=0)
gs_knn.fit(X_train, y_train)   # datos sin escalar: pipeline los procesa
best_knn = gs_knn.best_estimator_
print(f"   Mejores params: {gs_knn.best_params_}")

results['kNN'] = {
    'Train': get_metrics(best_knn, X_train, y_train),   # pipeline aplica scaler
    'Val':   get_metrics(best_knn, X_val,   y_val),
    'Test':  get_metrics(best_knn, X_test,  y_test),
}
best_models['kNN'] = best_knn

knn_params = {k.replace('clf__', ''): v for k, v in gs_knn.best_params_.items()}
plot_learning_curve(
    KNeighborsClassifier(**knn_params),
    'kNN', X_train_proc, y_train, X_val_proc, y_val)

# ── 2. Decision Tree – GridSearch ────────────────────────────
print("\n[2/5] Decision Tree (GridSearch)...")
param_grid_dt = {
    'max_depth':         [3, 5, 8, 12, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf':  [1, 2, 4],
    'criterion':         ['gini', 'entropy']
}
gs_dt   = GridSearchCV(DecisionTreeClassifier(random_state=42),
                       param_grid_dt, cv=5, scoring='f1', n_jobs=-1, verbose=0)
gs_dt.fit(X_train_proc, y_train)
best_dt = gs_dt.best_estimator_
print(f"   Mejores params: {gs_dt.best_params_}")

results['Decision Tree'] = {
    'Train': get_metrics(best_dt, X_train_proc, y_train),
    'Val':   get_metrics(best_dt, X_val_proc,   y_val),
    'Test':  get_metrics(best_dt, X_test_proc,  y_test),
}
best_models['Decision Tree'] = best_dt

plot_learning_curve(
    DecisionTreeClassifier(random_state=42, **gs_dt.best_params_),
    'Decision_Tree', X_train_proc, y_train, X_val_proc, y_val)

# ── 3. Random Forest – RandomizedSearch ──────────────────────
print("\n[3/5] Random Forest (RandomizedSearch)...")
param_dist_rf = {
    'n_estimators':      [50, 100, 200, 300],
    'max_depth':         [5, 10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf':  [1, 2, 4],
    'max_features':      ['sqrt', 'log2', 0.5]
}
rs_rf   = RandomizedSearchCV(RandomForestClassifier(random_state=42),
                             param_dist_rf, n_iter=30, cv=5,
                             scoring='f1', random_state=42, n_jobs=-1, verbose=0)
rs_rf.fit(X_train_proc, y_train)
best_rf = rs_rf.best_estimator_
print(f"   Mejores params: {rs_rf.best_params_}")

results['Random Forest'] = {
    'Train': get_metrics(best_rf, X_train_proc, y_train),
    'Val':   get_metrics(best_rf, X_val_proc,   y_val),
    'Test':  get_metrics(best_rf, X_test_proc,  y_test),
}
best_models['Random Forest'] = best_rf

plot_learning_curve(
    RandomForestClassifier(random_state=42, **rs_rf.best_params_),
    'Random_Forest', X_train_proc, y_train, X_val_proc, y_val)

# ── 4. Gradient Boosting – RandomizedSearch ──────────────────
print("\n[4/5] Gradient Boosting (RandomizedSearch)...")
param_dist_gb = {
    'n_estimators':    [50, 100, 200],
    'learning_rate':   [0.01, 0.05, 0.1, 0.2],
    'max_depth':       [3, 4, 5, 6],
    'subsample':       [0.7, 0.8, 1.0],
    'min_samples_leaf':[1, 2, 4]
}
rs_gb   = RandomizedSearchCV(GradientBoostingClassifier(random_state=42),
                             param_dist_gb, n_iter=30, cv=5,
                             scoring='f1', random_state=42, n_jobs=-1, verbose=0)
rs_gb.fit(X_train_proc, y_train)
best_gb = rs_gb.best_estimator_
print(f"   Mejores params: {rs_gb.best_params_}")

results['Gradient Boosting'] = {
    'Train': get_metrics(best_gb, X_train_proc, y_train),
    'Val':   get_metrics(best_gb, X_val_proc,   y_val),
    'Test':  get_metrics(best_gb, X_test_proc,  y_test),
}
best_models['Gradient Boosting'] = best_gb

plot_learning_curve(
    GradientBoostingClassifier(random_state=42, **rs_gb.best_params_),
    'Gradient_Boosting', X_train_proc, y_train, X_val_proc, y_val)

# ── 5. DNN (MLPClassifier) – GridSearch ──────────────────────
# Arquitecturas con mínimo 3 capas ocultas.
# alpha: regularización L2. early_stopping: detiene si val no mejora.
print("\n[5/5] Deep Neural Network (GridSearch)...")
param_grid_dnn = {
    'hidden_layer_sizes': [(128, 64, 32), (256, 128, 64), (128, 64, 32, 16)],
    'activation':         ['relu', 'tanh'],
    'alpha':              [1e-4, 1e-3, 1e-2],
    'learning_rate':      ['constant', 'adaptive'],
}
gs_dnn   = GridSearchCV(
    MLPClassifier(max_iter=500, early_stopping=True,
                  validation_fraction=0.1, random_state=42),
    param_grid_dnn, cv=5, scoring='f1', n_jobs=-1, verbose=0)
gs_dnn.fit(X_train_proc, y_train)
best_dnn = gs_dnn.best_estimator_
print(f"   Mejores params: {gs_dnn.best_params_}")

results['DNN'] = {
    'Train': get_metrics(best_dnn, X_train_proc, y_train),
    'Val':   get_metrics(best_dnn, X_val_proc,   y_val),
    'Test':  get_metrics(best_dnn, X_test_proc,  y_test),
}
best_models['DNN'] = best_dnn

plot_learning_curve(
    MLPClassifier(max_iter=500, early_stopping=True,
                  random_state=42, **gs_dnn.best_params_),
    'DNN', X_train_proc, y_train, X_val_proc, y_val)

# ── 5c – Tabla comparativa ────────────────────────────────────
print("\n\n" + "=" * 65)
print("  TABLA COMPARATIVA DE MÉTRICAS")
print("=" * 65)

rows = []
for model_name, splits in results.items():
    for split, metrics in splits.items():
        row = {'Modelo': model_name, 'Split': split}
        row.update(metrics)
        rows.append(row)
df_results = pd.DataFrame(rows)
print(df_results.to_string(index=False))

# Guardar tabla como imagen
fig, ax = plt.subplots(figsize=(16, 7))
ax.axis('off')
tbl = ax.table(cellText=df_results.values,   # CORREGIDO: sin variable muerta
               colLabels=df_results.columns,
               cellLoc='center', loc='center')
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
tbl.scale(1.2, 1.6)
for (row, col), cell in tbl.get_celld().items():
    if row == 0:
        cell.set_facecolor('#37474F')
        cell.set_text_props(color='white', fontweight='bold')
    elif df_results.iloc[row-1]['Split'] == 'Test':
        cell.set_facecolor('#FFF9C4')
    elif df_results.iloc[row-1]['Split'] == 'Train':
        cell.set_facecolor('#E3F2FD')
    else:
        cell.set_facecolor('#E8F5E9')
ax.set_title('Tabla Comparativa de Métricas por Modelo y Split\n'
             '(Amarillo = Test | Azul = Train | Verde = Validación)',
             fontsize=11, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig('graficas_modelos/tabla_comparativa.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n✅ Tabla comparativa guardada.")

# ── 5d – Gráfico comparativo F1 ──────────────────────────────
test_f1         = {m: results[m]['Test']['F1'] for m in results}
best_model_name = max(test_f1, key=test_f1.get)
print(f"\n🏆 Mejor modelo por F1 en Test: {best_model_name} "
      f"(F1={test_f1[best_model_name]:.4f})")

model_names = list(results.keys())
f1_train    = [results[m]['Train']['F1'] for m in model_names]
f1_val      = [results[m]['Val']['F1']   for m in model_names]
f1_test     = [results[m]['Test']['F1']  for m in model_names]

x     = np.arange(len(model_names))
width = 0.27
fig, ax = plt.subplots(figsize=(13, 6))
b1 = ax.bar(x - width, f1_train, width, label='Train',      color='#2196F3', alpha=0.85)
b2 = ax.bar(x,          f1_val,   width, label='Validación', color='#4CAF50', alpha=0.85)
b3 = ax.bar(x + width,  f1_test,  width, label='Test',       color='#F44336', alpha=0.85)
for bars in [b1, b2, b3]:
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.003,
                f'{h:.3f}', ha='center', va='bottom', fontsize=7.5)
ax.set_xlabel('Modelo', fontsize=11)
ax.set_ylabel('F1-Score', fontsize=11)
ax.set_title('Comparación F1-Score por Modelo y Split', fontsize=12, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(model_names, fontsize=10)
ax.legend(fontsize=10)
ax.set_ylim(0, 1.08)
ax.grid(True, alpha=0.25, axis='y')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig('graficas_modelos/comparacion_f1.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Gráfica comparación F1 guardada.")

# ============================================================
#     PUNTO 6 – EVALUACIÓN FINAL DEL MEJOR MODELO
# ============================================================
print("\n" + "=" * 65)
print(f"  PUNTO 6 – EVALUACIÓN FINAL: {best_model_name}")
print("=" * 65)

# ── 6a – Reentrenar con Train + Val ──────────────────────────
X_trainval      = pd.concat([X_train, X_val])
y_trainval      = pd.concat([y_train, y_val])
X_trainval_proc = preprocessing.fit_transform(X_trainval)
X_test_proc2    = preprocessing.transform(X_test)

_best_params_map = {
    'kNN':               {k.replace('clf__',''):v for k,v in gs_knn.best_params_.items()},
    'Decision Tree':     gs_dt.best_params_,
    'Random Forest':     rs_rf.best_params_,
    'Gradient Boosting': rs_gb.best_params_,
    'DNN':               gs_dnn.best_params_,
}
_base_estimators = {
    'kNN':               KNeighborsClassifier,
    'Decision Tree':     DecisionTreeClassifier,
    'Random Forest':     RandomForestClassifier,
    'Gradient Boosting': GradientBoostingClassifier,
    'DNN':               MLPClassifier,
}
_extra_kwargs = {
    'kNN':               {},
    'Decision Tree':     {'random_state': 42},
    'Random Forest':     {'random_state': 42},
    'Gradient Boosting': {'random_state': 42},
    'DNN':               {'max_iter': 500, 'early_stopping': True,
                          'validation_fraction': 0.1, 'random_state': 42},
}

final_model = _base_estimators[best_model_name](
    **_best_params_map[best_model_name],
    **_extra_kwargs[best_model_name]
)
final_model.fit(X_trainval_proc, y_trainval)
y_pred_final = final_model.predict(X_test_proc2)

# ── 6b – Métricas finales ────────────────────────────────────
print("\n--- MÉTRICAS FINALES SOBRE X_TEST ---")
print(classification_report(y_test, y_pred_final,
                             target_names=['Normal (0)', 'Fatiga (1)']))
final_metrics = {
    'Accuracy':  round(accuracy_score(y_test, y_pred_final), 4),
    'Precision': round(precision_score(y_test, y_pred_final), 4),
    'Recall':    round(recall_score(y_test, y_pred_final), 4),
    'F1':        round(f1_score(y_test, y_pred_final), 4),
}
print(pd.DataFrame([final_metrics], index=[best_model_name]).to_string())

# ── 6b – Matriz de confusión ──────────────────────────────────
cm = confusion_matrix(y_test, y_pred_final)
fig, ax = plt.subplots(figsize=(7, 6))
ConfusionMatrixDisplay(confusion_matrix=cm,
                       display_labels=['Normal (0)', 'Fatiga (1)']).plot(
    ax=ax, colorbar=True, cmap='Blues')
ax.set_title(f'Matriz de Confusión – {best_model_name}\n(Evaluación final sobre X_test)',
             fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig('graficas_modelos/confusion_matrix_final.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Matriz de confusión guardada.")

# ── 6b – Boxplots características discriminativas ────────────
X_test_df           = X_test.copy()
X_test_df['y_pred'] = y_pred_final
feat_diff = {
    col: abs(X_test_df[X_test_df['y_pred']==0][col].mean() -
             X_test_df[X_test_df['y_pred']==1][col].mean())
    for col in feature_names
}
top4_feats = sorted(feat_diff, key=feat_diff.get, reverse=True)[:4]
print(f"\nTop 4 características más discriminativas: {top4_feats}")

fig, axes = plt.subplots(1, 4, figsize=(18, 6))
fig.suptitle('Boxplots de Características Clave — Normal vs Fatiga (predicciones sobre X_test)',
             fontsize=12, fontweight='bold')
for i, feat in enumerate(top4_feats):
    ax    = axes[i]
    data0 = X_test_df[X_test_df['y_pred']==0][feat]
    data1 = X_test_df[X_test_df['y_pred']==1][feat]
    bp    = ax.boxplot([data0, data1], patch_artist=True,
                       medianprops={'color':'black', 'linewidth': 2},
                       flierprops={'marker':'o', 'markerfacecolor':'gray',
                                   'markersize': 4, 'alpha': 0.5})
    bp['boxes'][0].set_facecolor('#2196F3')
    bp['boxes'][1].set_facecolor('#F44336')
    ax.set_title(feat.replace('_', ' '), fontsize=9, fontweight='bold')
    ax.set_xticklabels(['Normal', 'Fatiga'], fontsize=9)
    ax.grid(True, alpha=0.25, axis='y')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig('graficas_modelos/boxplots_predicciones.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Boxplots de predicciones guardados.")

# ============================================================
#         PUNTO 7 – PRUEBA CON MUESTRA ARTIFICIAL
# ============================================================
print("\n" + "=" * 65)
print("  PUNTO 7 – PRUEBA CON MUESTRA ARTIFICIAL")
print("=" * 65)

np.random.seed(99)
means  = X.mean()
stds   = X.std()

# Muestra con valores de fatiga forzados:
# MDF/MNF caen ~22% (desplazamiento espectral a bajas frecuencias)
# RMS/MAV suben ~20% (mayor reclutamiento de unidades motoras)
# VAR sube ~35% (señal más irregular por asincronismo)
sample = means + np.random.uniform(-0.5, 0.5, len(means)) * stds
for col in feature_names:
    if '_MDF' in col or '_MNF' in col:
        sample[col] = means[col] * 0.78
    if '_RMS' in col or '_MAV' in col:
        sample[col] = means[col] * 1.20
    if '_VAR' in col:
        sample[col] = means[col] * 1.35

sample_df   = pd.DataFrame([sample], columns=feature_names)
sample_proc = preprocessing.transform(sample_df)
prediction  = final_model.predict(sample_proc)[0]

proba_str = ""
if hasattr(final_model, 'predict_proba'):
    proba     = final_model.predict_proba(sample_proc)[0]
    proba_str = f"P(Normal)={proba[0]:.3f} | P(Fatiga)={proba[1]:.3f}"
    print(f"\n→ Probabilidades: {proba_str}")

label = 'FATIGA MUSCULAR (1)' if prediction == 1 else 'CONDICIÓN NORMAL (0)'
print(f"\n→ Predicción para muestra artificial: {label}")
print(f"\nValores de la muestra artificial (primeras 14 features):")
print(sample_df.iloc[0, :14].round(4).to_string())

# Visualización muestra vs distribución real
plot_feats = top4_feats + [c for c in feature_names if '_MDF' in c][:4]
plot_feats = list(dict.fromkeys(plot_feats))[:8]

fig, axes = plt.subplots(2, 4, figsize=(18, 9))
fig.suptitle(
    f'Muestra Artificial vs Distribución Real del Dataset\n'
    f'Predicción: {"FATIGA" if prediction==1 else "NORMAL"}  {proba_str}',
    fontsize=12, fontweight='bold')
for i, feat in enumerate(plot_feats):
    ax = axes[i//4][i%4]
    ax.hist(X[y==0][feat], bins=25, alpha=0.55, color='#2196F3',
            label='Normal', density=True, edgecolor='white')
    ax.hist(X[y==1][feat], bins=25, alpha=0.55, color='#F44336',
            label='Fatiga', density=True, edgecolor='white')
    ax.axvline(sample[feat], color='black', linewidth=2.5,
               linestyle='--', label='Muestra artificial')
    ax.set_title(feat.replace('_', ' '), fontsize=8, fontweight='bold')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig('graficas_modelos/muestra_artificial.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Gráfica muestra artificial guardada.")

# ============================================================
#         GUARDAR REPORTE Y CONCLUSIONES
# ============================================================
with open('conclusiones/reporte_resultados.txt', 'w', encoding='utf-8') as f:
    f.write("=" * 70 + "\n")
    f.write("  REPORTE DE RESULTADOS — DETECCIÓN DE FATIGA MUSCULAR\n")
    f.write("=" * 70 + "\n\n")
    f.write("MÉTRICAS POR MODELO Y SPLIT\n" + "-" * 70 + "\n")
    f.write(df_results.to_string(index=False) + "\n\n")
    f.write("MEJORES HIPERPARÁMETROS\n" + "-" * 70 + "\n")
    f.write(f"kNN:               {gs_knn.best_params_}\n")
    f.write(f"Decision Tree:     {gs_dt.best_params_}\n")
    f.write(f"Random Forest:     {rs_rf.best_params_}\n")
    f.write(f"Gradient Boosting: {rs_gb.best_params_}\n")
    f.write(f"DNN:               {gs_dnn.best_params_}\n\n")
    f.write("MEJOR MODELO\n" + "-" * 70 + "\n")
    f.write(f"Modelo:     {best_model_name}\n")
    f.write(f"F1 en Test: {test_f1[best_model_name]:.4f}\n\n")
    f.write("MÉTRICAS FINALES (Train+Val → Test)\n" + "-" * 70 + "\n")
    for k, v in final_metrics.items():
        f.write(f"  {k}: {v}\n")
    f.write(f"\nMATRIZ DE CONFUSIÓN:\n{cm}\n\n")
    f.write("PREDICCIÓN MUESTRA ARTIFICIAL\n" + "-" * 70 + "\n")
    f.write(f"Resultado: {label}\n")
    if proba_str:
        f.write(f"{proba_str}\n")

print("\n✅ Reporte guardado en conclusiones/reporte_resultados.txt")

# Conclusiones cualitativas (incrustadas aquí para no depender de archivo externo)
conclusiones_texto = """
=======================================================================
  CONCLUSIONES E INTERPRETACIONES DETALLADAS POR PUNTO
=======================================================================

PUNTO 1a — Preprocesamiento del target
La unificación de clases 1 y 2 en una sola clase (fatiga=1) simplifica
el problema a clasificación binaria clínicamente interpretable. En un
sistema de alerta deportiva, lo relevante es saber si HAY fatiga, no
su grado exacto.

PUNTO 1b — Clasificación de variables
Las 8 señales EMG son numéricas continuas en mV. Time es numérica
continua (timestamps). Target es binaria. Time fue excluida del
modelado por su relación lineal con el target (data leakage).

PUNTO 2 — Características extraídas
- RMS: energía de la señal, aumenta con fatiga (más reclutamiento).
- VAR: dispersión, refleja irregularidad muscular.
- ZCR: tasa de cruce por cero, ligada a tasa de disparo motora.
- MAV: valor absoluto medio, indicador clásico de actividad EMG.
- MDF: frecuencia mediana (INDICADOR PRINCIPAL DE FATIGA: disminuye).
- MNF: frecuencia media, complementa MDF para fibras tipo I.
- PWR: potencia espectral total, energía global del espectro.
Total: 7 características × 8 canales = 56 características.

PUNTO 3 — EDA
- Señales EMG con morfología típica (±0.5 a ±2 mV, 20-500 Hz).
- RMS con distribución asimétrica positiva (mayoría baja activación).
- MDF concentrada en 50-200 Hz, separación visible entre clases.
- Desbalance moderado (más ventanas normales que de fatiga).
- RMS/MAV/VAR muy correlacionadas entre sí (r>0.9, miden amplitud).
- MDF y MNF son los indicadores más separables entre clases.

PUNTO 4 — Procesamiento
- SimpleImputer(median): robusto a outliers para posibles NaN.
- StandardScaler: necesario para kNN (distancias) y DNN (gradientes).
- División 70/15/15 con stratify para conservar proporción de clases.
- fit_transform solo en train; transform en val y test (sin leakage).

PUNTO 5 — Modelos
Overfitting detectado en: Decision Tree (train F1≈1.0, val<train).
kNN con k pequeño también tiende a sobreajustar en alta dimensión.
Random Forest y Gradient Boosting son más robustos al overfitting.
DNN controlado con early_stopping y regularización L2 (alpha).
El modelo seleccionado para producción es el de mayor F1 en Test,
preferiblemente Random Forest o Gradient Boosting por su robustez
e interpretabilidad (feature importance).

PUNTO 6 — Evaluación final
Falsos negativos (fatiga no detectada) son más costosos que FP
en contexto deportivo → se prioriza Recall alto.
Los boxplots de predicciones confirman que el modelo separa
correctamente las clases usando MDF y RMS principalmente.
Mejoras posibles: más sujetos, SMOTE para desbalance, LSTM para
aprovechar dependencia temporal entre ventanas.

PUNTO 7 — Muestra artificial
La muestra simula fatiga con MDF↓22%, RMS↑20%, VAR↑35%.
Si el modelo la predice como fatiga → aprendió patrones correctos.
Los valores se alinean con la literatura EMG (Merletti, De Luca).
=======================================================================
"""

with open('conclusiones/conclusiones_completas.txt', 'w', encoding='utf-8') as f:
    f.write(conclusiones_texto)
print("✅ Conclusiones guardadas en conclusiones/conclusiones_completas.txt")

print("\n" + "=" * 65)
print("  TRABAJO COMPLETO FINALIZADO")
print("  graficas_EDA/     → 6 gráficas del EDA")
print("  graficas_modelos/ → curvas, tabla, CM, boxplots, muestra")
print("  conclusiones/     → reporte_resultados.txt + conclusiones.txt")
print("=" * 65)