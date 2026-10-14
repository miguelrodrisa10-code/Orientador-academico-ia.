"""
Sistema orientador académico basado en ML (versión mejorada).

Predice la nota media de Bachillerato (por modalidad) a partir de las notas de ESO,
dando mucho peso a los cursos recientes y a la tendencia (si las notas suben o bajan).

Novedades respecto a la versión anterior:
  * Intervalos de predicción reales (conformal split sobre residuos out-of-fold),
    en lugar de un "± MAE".
  * Validación opcional por promoción (--columna-grupo): entrena con unas promociones
    y evalúa en otras, que es el uso real del modelo.
  * Regla de 1 error estándar: entre los modelos empatados se elige el más simple
    (reduce el "winner's curse" de elegir el mejor de 8 por azar).
  * Modo --privado: el .joblib no contiene notas de alumnos (sin KNNImputer ni SVR).
  * Variables derivadas accesibles por NOMBRE (se acabaron los índices fijos).
  * Validación cruzada manual: una sola pasada da MAE, R² y predicciones out-of-fold.
  * Orientación redactada como descripción del perfil, no como recomendación.

Uso:
    python "Proyecto de investigacion.py"                    # entrena si no hay modelo válido
    python "Proyecto de investigacion.py" --reentrenar       # fuerza reentrenamiento
    python "Proyecto de investigacion.py" --privado          # modelo sin datos personales
    python "Proyecto de investigacion.py" --columna-grupo Curso   # validación por promoción
    python "Proyecto de investigacion.py" --datos "otra\\ruta"
"""
import argparse
import os
import re
import threading
import tkinter as tk
import warnings
from pathlib import Path
from tkinter import messagebox, ttk

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin, clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (GradientBoostingRegressor, RandomForestRegressor,
                              VotingRegressor)
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import (GridSearchCV, GroupKFold, KFold,
                                     LeaveOneGroupOut)
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

# ==========================================
# CONFIGURACIÓN
# ==========================================
DIRECTORIO_DATOS_DEFECTO = r"C:\Users\GERMAN\Downloads\Csv datos\drive-download-20260909T171012Z-1-001"
ARCHIVO_MODELO_DEFECTO = str(Path(__file__).resolve().parent / "modelos_bachillerato.joblib")
VERSION_MODELO = 8

SEMILLA = 42
N_FOLDS = 5
MIN_ALUMNOS = 30
FRACCION_MIN_COLUMNA = 0.20
FRACCION_MIN_NOTAS_GUI = 0.30
POTENCIA_PESO_CURSO = 2
UMBRAL_TENDENCIA = 0.15
NIVELES_INTERVALO = (0.80, 0.90)  # cobertura de los intervalos de predicción

RE_ESO = re.compile(r"^E[1-4]_")
RE_BACH = re.compile(r"^B[1-4]_")

DICCIONARIO_MATERIAS = {
    'EFI': 'Educación Física', 'EPV': 'Educación Plástica y Visual',
    'EPVA': 'Educación Plástica, Visual y Audiovisual', 'FIQ': 'Física y Química',
    'FQ': 'Física y Química', 'LCL': 'Lengua Castellana y Literatura',
    'MAT': 'Matemáticas (A/B/Académicas/Aplicadas)', 'MAA': 'Matemáticas A',
    'MAB': 'Matemáticas B', 'MAP': 'Matemáticas Aplicadas',
    'MCS': 'Matemáticas Aplicadas a las CCSS', 'LEI': 'Inglés (Lengua Extranjera)',
    'FRE': 'Francés', 'FIS': 'Física', 'QUI': 'Química', 'HES': 'Historia de España',
    'HMC': 'Historia del Mundo Contemporáneo', 'GEG': 'Geografía',
    'TYD': 'Tecnología y Digitalización',
    'GHI': 'Geografía e Historia', 'GEO': 'Geografía e Historia',
    'HIST': 'Historia', 'ING': 'Inglés', 'TIC': 'Tecnología / TIC',
    'TEC': 'Tecnología', 'TECN': 'Tecnología', 'LAT': 'Latín',
    'BIO': 'Biología y Geología', 'BGE': 'Biología y Geología', 'MUS': 'Música',
    'FRA': 'Francés', 'REL': 'Religión', 'VAE': 'Valores Éticos', 'ECO': 'Economía',
    'DTT': 'Dibujo Técnico', 'DT': 'Dibujo Técnico', 'FIL': 'Filosofía',
}

NOTA_S0 = 9.0
EQUIVALENCIAS_ESO = {'MAA': 'MAT', 'MAB': 'MAT', 'MAP': 'MAT', 'TYD': 'TEC'}
MATERIAS_TECNICAS = {'MAT', 'FIQ', 'BGE', 'TEC', 'TIC'}
MATERIAS_SOCIALES = {'LCL', 'GHI', 'LEI', 'LAT', 'ECO'}
OBJETIVO_TECNOLOGICO = {'MAT', 'FIQ', 'FIS', 'QUI', 'BIO', 'BGC', 'DIT', 'TEI'}
OBJETIVO_SOCIAL = {'MCS', 'ECO', 'HMC', 'GEG', 'LAT', 'GRI', 'EEM'}


# ==========================================
# UTILIDADES
# ==========================================
def codigo_materia(columna):
    """'E3_MAT' -> 'MAT'."""
    return columna.split("_", 1)[1].upper() if "_" in columna else columna.upper()


def obtener_nombre_humano(codigo):
    """Traduce 'E3_MAT' a 'Matemáticas (3º ESO)'."""
    codigo = str(codigo).strip()
    m = re.match(r"^([EB])([1-4])_(.+)$", codigo, flags=re.IGNORECASE)
    if not m:
        return codigo
    etapa = "ESO" if m.group(1).upper() == "E" else "Bachillerato"
    materia = DICCIONARIO_MATERIAS.get(m.group(3).upper(), m.group(3).upper())
    return f"{materia} ({m.group(2)}º {etapa})"


def _media_nan(X, idx):
    """Media por fila de las columnas `idx`, ignorando NaN (NaN si no hay ninguna nota)."""
    if not idx:
        return np.full(len(X), np.nan)
    sub = X[:, idx]
    m = ~np.isnan(sub)
    cnt = m.sum(axis=1)
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(cnt > 0, np.where(m, sub, 0.0).sum(axis=1) / cnt, np.nan)


def _media_ponderada(X, pesos):
    """Media ponderada por fila ignorando NaN."""
    m = ~np.isnan(X)
    num = np.where(m, X, 0.0) @ pesos
    den = m.astype(float) @ pesos
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(den > 0, num / den, np.nan)


def pesos_cursos(columnas):
    return np.array([int(c[1]) for c in columnas], dtype=float) ** POTENCIA_PESO_CURSO


def base_ponderada(X, columnas):
    """Nota de partida del alumno: su media ESO con más peso en los cursos altos."""
    b = _media_ponderada(np.asarray(X, dtype=float), pesos_cursos(columnas))
    return np.where(np.isnan(b), 6.0, b)


def nombres_features(columnas):
    """Nombres de las columnas que produce IngenieriaCaracteristicas, en orden."""
    cursos = sorted({int(c[1]) for c in columnas})
    return (list(columnas)
            + ['Perfil_Tecnico', 'Perfil_Social', 'Media_Global', 'Media_Ponderada']
            + [f'Media_E{c}' for c in cursos]
            + ['Tendencia', 'Pendiente', 'Ultimo_Curso', 'Desviacion', 'Minimo',
               'Sesgo_Cognitivo'])


class IngenieriaCaracteristicas(TransformerMixin, BaseEstimator):
    """Añade variables sintéticas a las notas de ESO (con NaN: solo se usan notas reales).

    El orden de las columnas de salida está en `nombres_features(columnas)`.
    Definida a nivel de módulo para que joblib pueda serializarla.
    """

    def __init__(self, columnas):
        self.columnas = columnas

    def fit(self, X, y=None):
        self.n_features_in_ = np.asarray(X).shape[1]
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        cols = list(self.columnas)
        cursos = np.array([int(c[1]) for c in cols], dtype=float)
        idx_tec = [i for i, c in enumerate(cols) if codigo_materia(c) in MATERIAS_TECNICAS]
        idx_soc = [i for i, c in enumerate(cols) if codigo_materia(c) in MATERIAS_SOCIALES]

        media_global = _media_nan(X, list(range(len(cols))))
        tec = np.where(np.isnan(t := _media_nan(X, idx_tec)), media_global, t)
        soc = np.where(np.isnan(s := _media_nan(X, idx_soc)), media_global, s)
        media_pond = _media_ponderada(X, pesos_cursos(cols))

        cursos_unicos = np.array(sorted(set(cursos)))
        MC = np.column_stack([_media_nan(X, [i for i, c in enumerate(cursos) if c == cu])
                              for cu in cursos_unicos])

        # Pendiente de la recta nota-curso usando solo los cursos con datos
        c = cursos_unicos[None, :]
        m = ~np.isnan(MC)
        n = m.sum(axis=1)
        with np.errstate(invalid='ignore', divide='ignore'):
            cm = np.where(n > 0, np.where(m, c, 0.0).sum(axis=1) / n, 0.0)
            ym = np.where(n > 0, np.where(m, MC, 0.0).sum(axis=1) / n, 0.0)
        dc = np.where(m, c - cm[:, None], 0.0)
        dy = np.where(m, MC - ym[:, None], 0.0)
        den = (dc ** 2).sum(axis=1)
        with np.errstate(invalid='ignore', divide='ignore'):
            pendiente = np.where(den > 0, (dc * dy).sum(axis=1) / den, 0.0)

        # Primer y último curso con datos
        primero = np.full(len(X), np.nan)
        ultimo = np.full(len(X), np.nan)
        for j in range(MC.shape[1]):
            primero = np.where(np.isnan(primero), MC[:, j], primero)
        for j in range(MC.shape[1] - 1, -1, -1):
            ultimo = np.where(np.isnan(ultimo), MC[:, j], ultimo)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            desviacion = np.nanstd(X, axis=1)
            minimo = np.nanmin(X, axis=1)

        return np.column_stack([
            X, tec, soc, media_global, media_pond, MC,
            ultimo - primero, pendiente, ultimo, desviacion, minimo, tec - soc,
        ])


def crear_pipeline(modelo, columnas, imputador=None):
    """Características (con NaN) -> imputación -> escalado -> modelo."""
    if imputador is None:
        imputador = KNNImputer(n_neighbors=5, weights='distance', keep_empty_features=True)
    return Pipeline([
        ('features', IngenieriaCaracteristicas(columnas)),
        ('imputer', imputador),
        ('scaler', StandardScaler()),
        ('modelo', modelo),
    ])


class PredictorConTendencia(RegressorMixin, BaseEstimator):
    """Predice la DIFERENCIA entre la nota de Bachillerato y la media ponderada de ESO."""

    def __init__(self, estimador, columnas):
        self.estimador = estimador
        self.columnas = columnas

    def fit(self, X, y):
        X = pd.DataFrame(X, columns=self.columnas)
        base = base_ponderada(X.values, self.columnas)
        self.estimador_ = clone(self.estimador).fit(X, np.asarray(y, dtype=float) - base)
        return self

    def predict(self, X):
        X = pd.DataFrame(X, columns=self.columnas)
        return base_ponderada(X.values, self.columnas) + self.estimador_.predict(X)

    def caracteristicas(self, X):
        """DataFrame con las variables derivadas (ya imputadas), accesibles por nombre."""
        arr = self.estimador_[:2].transform(pd.DataFrame(X, columns=self.columnas))
        return pd.DataFrame(arr, columns=nombres_features(self.columnas))


# ==========================================
# CARGA Y PREPARACIÓN DE DATOS
# ==========================================
def cargar_datos(directorio):
    archivos = sorted(Path(directorio).glob("*.csv"))
    if not archivos:
        raise FileNotFoundError(f"No se encontraron archivos CSV en {directorio}")
    frames = []
    for f in archivos:
        for enc in ("utf-8-sig", "latin-1"):
            try:
                frames.append(pd.read_csv(f, sep=None, engine="python", encoding=enc))
                break
            except UnicodeDecodeError:
                continue
    df = pd.concat(frames, ignore_index=True)
    df.columns = df.columns.astype(str).str.strip().str.replace('\ufeff', '', regex=False)
    return df


def _columna_id(df):
    col_id = next((c for c in ('Exp.', 'Exp') if c in df.columns), None)
    if col_id is None:
        raise ValueError("Falta la columna de expediente ('Exp.' o 'Exp') en los CSV.")
    return col_id


def limpiar_notas(serie):
    """Texto -> número 0-10 según el formato de tus CSV.

    'ME'/'MH' -> 10; 'S0' -> 9; letra + número ('N8', 'SB9') -> el número;
    códigos sin número (CV, NE, AF, RE...) o fuera de 0-10 -> NaN.
    """
    s = serie.astype(str).str.upper().str.replace(',', '.', regex=False).str.strip()
    num = pd.to_numeric(s.str.extract(r'(\d+(?:\.\d+)?)', expand=False), errors='coerce')
    num = num.where(~s.str.contains(r'ME|MH', na=False, regex=True), 10.0)
    num = num.where(s != 'S0', NOTA_S0)
    return num.where(num.between(0, 10))


def construir_tabla_alumnos(df):
    """Filas (alumno, materia, curso) -> una fila por alumno, una columna por curso_materia."""
    col_id = _columna_id(df)
    faltan = [c for c in ('Materia', 'Grupo', 'Nota') if c not in df.columns]
    if faltan:
        raise ValueError(f"Faltan columnas obligatorias en los CSV: {faltan}")

    df = df.copy()
    df['Nota'] = limpiar_notas(df['Nota'])
    if 'Nota Recuperación.' in df.columns:
        df['Nota Recuperación.'] = limpiar_notas(df['Nota Recuperación.'])
        df['Nota_Definitiva'] = df[['Nota', 'Nota Recuperación.']].max(axis=1)
    else:
        df['Nota_Definitiva'] = df['Nota']

    df['Nivel'] = df['Grupo'].astype(str).str.extract(r'^([EB][1-4])', expand=False)
    df['Materia_Limpia'] = (df['Materia'].astype(str).str.strip().str.upper()
                            .str.extract(r'^([A-ZÁÉÍÓÚÑ]+)', expand=False).fillna('VAR'))
    es_eso = df['Nivel'].astype(str).str.startswith('E')
    df.loc[es_eso, 'Materia_Limpia'] = df.loc[es_eso, 'Materia_Limpia'].replace(EQUIVALENCIAS_ESO)
    df = df.dropna(subset=['Nota_Definitiva', 'Nivel'])

    tabla = df.pivot_table(index=col_id, columns=['Nivel', 'Materia_Limpia'],
                           values='Nota_Definitiva', aggfunc='mean')
    tabla.columns = [f"{n}_{m}" for n, m in tabla.columns]
    return tabla[sorted(tabla.columns)]


def extraer_grupos(df, columna_grupo):
    """Promoción de cada alumno: valor más antiguo de `columna_grupo` en sus filas de BACHILLERATO.

    Se usa Bachillerato (y no la primera aparición en los datos) porque el primer año del
    dataset no es una promoción real: casi todos los alumnos "empezarían" ahí.
    """
    if not columna_grupo:
        return None
    if columna_grupo not in df.columns:
        raise ValueError(f"La columna de grupo '{columna_grupo}' no existe en los CSV. "
                         f"Columnas disponibles: {list(df.columns)}")
    es_bach = df['Grupo'].astype(str).str.match(r'^B[1-4]')
    return df[es_bach].groupby(_columna_id(df))[columna_grupo].min()
# ==========================================
# ENTRENAMIENTO
# ==========================================
def candidatos(columnas, privado=False):
    """Modelos a comparar, ordenados de MÁS SIMPLE a MÁS COMPLEJO (importa para la regla 1-SE).

    En modo privado se excluyen los que guardan datos de alumnos (KNNImputer, SVR).
    """
    cv_interno = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEMILLA)
    mediana = SimpleImputer(strategy='median', keep_empty_features=True)

    def p(modelo):
        imp = mediana if privado else None
        return PredictorConTendencia(crear_pipeline(modelo, columnas, imp), columnas)

    def con_busqueda(modelo, grid):
        grid = {f"estimador__{k}": v for k, v in grid.items()}
        return GridSearchCV(p(modelo), grid, cv=cv_interno,
                            scoring='neg_mean_absolute_error', n_jobs=1)

    if privado:
        imputadores = [clone(mediana)]
    else:
        imputadores = [KNNImputer(n_neighbors=k, weights='distance', keep_empty_features=True)
                       for k in (5, 10)] + [clone(mediana)]

    mods = {
        'Línea base 1 (media del grupo)': crear_pipeline(
            DummyRegressor(strategy='mean'), columnas, mediana if privado else None),
        'Línea base 2 (media ponderada)': p(DummyRegressor(strategy='mean')),
        'Ridge (ajustado)': con_busqueda(Ridge(), {
            'imputer': imputadores, 'modelo__alpha': [1, 10, 100]}),
        'Gradient Boosting': p(GradientBoostingRegressor(random_state=SEMILLA)),
        'Random Forest': p(RandomForestRegressor(n_estimators=200, random_state=SEMILLA)),
        'Red Neuronal (MLP)': p(MLPRegressor(hidden_layer_sizes=(32, 16), max_iter=2000,
                                             early_stopping=True, random_state=SEMILLA)),
    }
    if not privado:
        mods['SVM (SVR, ajustado)'] = con_busqueda(SVR(kernel='rbf'), {
            'imputer': imputadores, 'modelo__C': [0.3, 1, 3],
            'modelo__gamma': ['scale'], 'modelo__epsilon': [0.1]})
    miembros = [('ridge', Ridge(alpha=30)), ('gb', GradientBoostingRegressor(random_state=SEMILLA))]
    if not privado:
        miembros.insert(0, ('svr', SVR(kernel='rbf', C=3, epsilon=0.1)))
    mods['Ensamble (Voting)'] = p(VotingRegressor(miembros))
    return mods


def crear_validacion(grupos):
    """KFold aleatorio, o validación por grupos (promociones) si hay columna de grupo."""
    if grupos is None:
        return KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEMILLA), "KFold aleatorio"
    n_grupos = grupos.nunique()
    if n_grupos < 2:
        raise ValueError("La columna de grupo tiene un único valor; no sirve para validar.")
    if n_grupos <= 10:
        return LeaveOneGroupOut(), f"una promoción fuera cada vez ({n_grupos} grupos)"
    return GroupKFold(n_splits=N_FOLDS), f"GroupKFold ({n_grupos} grupos)"


def evaluar(est, X, y, cv, grupos):
    """CV manual: devuelve MAE por fold, R² medio y predicciones out-of-fold."""
    oof = np.full(len(y), np.nan)
    maes, r2s = [], []
    for tr, te in cv.split(X, y, groups=grupos):
        m = clone(est).fit(X.iloc[tr], y.iloc[tr])
        pred = m.predict(X.iloc[te])
        oof[te] = pred
        maes.append(mean_absolute_error(y.iloc[te], pred))
        r2s.append(r2_score(y.iloc[te], pred))
    return np.array(maes), float(np.mean(r2s)), oof


def elegir_por_un_error_estandar(reales, orden):
    """Entre los modelos a menos de 1 error estándar del mejor, el más simple (según `orden`)."""
    mejor = min(reales, key=lambda k: reales[k]['mae'])
    se = reales[mejor]['mae_std'] / np.sqrt(reales[mejor]['n_folds'])
    umbral = reales[mejor]['mae'] + se
    return next(k for k in orden if k in reales and reales[k]['mae'] <= umbral), mejor


def cuantiles_conformal(residuos_abs, niveles):
    """Cuantil conformal split con corrección de muestra finita: cubre >= nivel de los casos."""
    n = len(residuos_abs)
    out = {}
    for nivel in niveles:
        q = min(1.0, np.ceil((n + 1) * nivel) / n)
        out[nivel] = float(np.quantile(residuos_abs, q))
    return out


def ajustar_final(estimador, X, y):
    est = clone(estimador).fit(X, y)
    if isinstance(est, GridSearchCV):
        print(f"   Mejores parámetros: {est.best_params_}")
        return est.best_estimator_
    return est


def entrenar_modalidad(tabla, nombre, cols_eso, cols_objetivo, grupos_alumnos=None, privado=False):
    if not cols_objetivo:
        raise ValueError(f"No hay asignaturas de Bachillerato para la modalidad {nombre}.")

    sub = tabla.dropna(subset=cols_objetivo, how='all')
    y = sub[cols_objetivo].mean(axis=1)
    X = sub[cols_eso]
    X = X.dropna(axis=1, thresh=max(1, int(len(X) * FRACCION_MIN_COLUMNA)))
    mask = X.notna().any(axis=1)
    X, y = X[mask], y[mask]
    if len(X) < MIN_ALUMNOS:
        raise ValueError(f"Solo hay {len(X)} alumnos para {nombre} (mínimo {MIN_ALUMNOS}).")
    columnas = list(X.columns)

    grupos = None
    if grupos_alumnos is not None:
        grupos = grupos_alumnos.reindex(X.index)
        if grupos.isna().any():
            raise ValueError("Hay alumnos sin valor en la columna de grupo.")
        grupos = grupos.values
    cv, desc_cv = crear_validacion(None if grupos is None else pd.Series(grupos))

    print(f"\n{'=' * 60}\n EVALUACIÓN: {nombre.upper()}  ({len(X)} alumnos, {len(columnas)} asignaturas)")
    print(f" Validación: {desc_cv}{'  [modo privado]' if privado else ''}\n{'=' * 60}")

    mods = candidatos(columnas, privado)
    resultados, oofs = {}, {}
    for nombre_alg, est in mods.items():
        print(f"   evaluando {nombre_alg}...", flush=True)
        maes, r2, oof = evaluar(est, X, y, cv, grupos)
        resultados[nombre_alg] = {'mae': maes.mean(), 'mae_std': maes.std(),
                                  'n_folds': len(maes), 'r2': r2}
        oofs[nombre_alg] = oof
        print(f"-> {nombre_alg:32s} | MAE: {maes.mean():.3f} ± {maes.std():.3f} | R²: {r2:.3f}")

    reales = {k: v for k, v in resultados.items() if not k.startswith('Línea base')}
    mae_base = min(v['mae'] for k, v in resultados.items() if k.startswith('Línea base'))
    elegido, mejor_bruto = elegir_por_un_error_estandar(reales, list(mods))
    r = reales[elegido]
    print(f"\n🏆 Elegido: {elegido} (MAE CV {r['mae']:.3f} ± {r['mae_std']:.3f}; "
          f"mejor línea base {mae_base:.3f})")
    if elegido != mejor_bruto:
        print(f"   (el de menor MAE es {mejor_bruto}, pero la diferencia queda dentro de 1 error "
              "estándar: se prefiere el modelo más simple)")
    if r['mae'] >= mae_base:
        print("⚠️  El modelo elegido NO supera a la línea base: las predicciones no son fiables.")

    # Intervalos: residuos out-of-fold del modelo elegido (honestos, no ven su propio alumno).
    residuos = np.abs(y.values - oofs[elegido])
    q = cuantiles_conformal(residuos, NIVELES_INTERVALO)
    print("   Intervalos de predicción: " +
          ", ".join(f"{int(n * 100)}% → ±{v:.2f}" for n, v in q.items()))

    modelo = ajustar_final(mods[elegido], X, y)
    sesgo = modelo.caracteristicas(X)['Sesgo_Cognitivo'].values
    umbral_bajo, umbral_alto = np.percentile(sesgo, [25, 75])

    return {
        'version': VERSION_MODELO, 'pipeline': modelo, 'columnas': columnas,
        'modelo_nombre': elegido, 'mae': float(r['mae']), 'mae_std': float(r['mae_std']),
        'mae_base': float(mae_base), 'n_alumnos': int(len(X)),
        'intervalos': {int(n * 100): v for n, v in q.items()},
        'validacion': desc_cv, 'privado': bool(privado),
        'umbral_bajo': float(umbral_bajo), 'umbral_alto': float(umbral_alto),
    }


def entrenar_y_guardar_modelos(directorio, ruta_salida, columna_grupo=None, privado=False):
    print("⏳ Cargando datos y evaluando modelos (puede tardar)...")
    df = cargar_datos(directorio)
    tabla = construir_tabla_alumnos(df)
    grupos = extraer_grupos(df, columna_grupo)
    cols_eso = [c for c in tabla.columns if RE_ESO.match(c)]
    cols_bach = [c for c in tabla.columns if RE_BACH.match(c)]
    cols_tec = [c for c in cols_bach if codigo_materia(c) in OBJETIVO_TECNOLOGICO]
    cols_soc = [c for c in cols_bach if codigo_materia(c) in OBJETIVO_SOCIAL]
    print(f"Alumnos con ESO y Bachillerato a la vez: "
          f"{int((tabla[cols_eso].notna().any(axis=1) & tabla[cols_bach].notna().any(axis=1)).sum())}")

    paquete = {
        'tecnologico': entrenar_modalidad(tabla, "Modalidad Tecnológica", cols_eso, cols_tec,
                                          grupos, privado),
        'sociales': entrenar_modalidad(tabla, "Modalidad Ciencias Sociales", cols_eso, cols_soc,
                                       grupos, privado),
    }
    joblib.dump(paquete, ruta_salida)
    print("\n✅ Entrenamiento completado. Modelos guardados.")
    if not privado:
        print("ℹ️  Este .joblib puede contener notas reales de alumnos (KNNImputer/SVR). "
              "No lo compartas; usa --privado para evitarlo.\n")
    return paquete


# ==========================================
# INTERFAZ GRÁFICA
# ==========================================
class AplicacionPrediccion:
    def __init__(self, root, ruta_modelo, directorio, reentrenar=False,
                 columna_grupo=None, privado=False):
        self.root = root
        self.ruta_modelo = ruta_modelo
        self.directorio = directorio
        self.columna_grupo = columna_grupo
        self.privado = privado
        self.paquete = None
        self.entries = {}
        self._error_hilo = None

        root.title("Sistema Orientador Académico Basado en ML")
        root.geometry("620x820")
        root.resizable(False, False)
        ttk.Style().theme_use('clam')

        self.paquete = None if reentrenar else self._cargar_paquete_valido(ruta_modelo)
        if self.paquete is None:
            self._entrenar_en_hilo()
        else:
            self._construir_ui()

    @staticmethod
    def _cargar_paquete_valido(ruta):
        """Devuelve el modelo guardado, o None si no existe o es de una versión antigua."""
        if not Path(ruta).exists():
            return None
        try:
            paquete = joblib.load(ruta)
            claves = {'pipeline', 'columnas', 'mae', 'mae_std', 'mae_base', 'n_alumnos',
                      'umbral_bajo', 'umbral_alto', 'modelo_nombre', 'version', 'intervalos'}
            if all(k in paquete and claves <= set(paquete[k])
                   and paquete[k]['version'] == VERSION_MODELO
                   for k in ('tecnologico', 'sociales')):
                return paquete
        except Exception:
            pass
        print("⚠️  El modelo guardado es de una versión anterior; se reentrenará.")
        return None

    def _entrenar_en_hilo(self):
        self.frame_carga = ttk.Frame(self.root)
        self.frame_carga.pack(expand=True)
        ttk.Label(self.frame_carga, text="Entrenando modelos, puede tardar varios minutos...\n"
                                         "(el progreso se ve en la terminal)",
                  font=('Helvetica', 11), justify="center").pack(pady=10)
        barra = ttk.Progressbar(self.frame_carga, mode='indeterminate', length=300)
        barra.pack(pady=5)
        barra.start(15)

        def tarea():
            try:
                self.paquete = entrenar_y_guardar_modelos(
                    self.directorio, self.ruta_modelo, self.columna_grupo, self.privado)
            except Exception as e:
                self._error_hilo = e

        self.hilo = threading.Thread(target=tarea, daemon=True)
        self.hilo.start()
        self.root.after(300, self._comprobar_hilo)

    def _comprobar_hilo(self):
        if self.hilo.is_alive():
            self.root.after(300, self._comprobar_hilo)
            return
        self.frame_carga.destroy()
        if self._error_hilo:
            messagebox.showerror("Error de entrenamiento", str(self._error_hilo))
            self.root.destroy()
        else:
            self._construir_ui()

    def _construir_ui(self):
        root = self.root
        ttk.Label(root, text="Modalidad de Bachillerato a consultar:",
                  font=('Helvetica', 11, 'bold')).pack(pady=8)
        self.combo = ttk.Combobox(root, state="readonly", width=48, values=[
            "Bachillerato Tecnológico / Científico",
            "Bachillerato Ciencias Sociales / Humanidades"])
        self.combo.current(0)
        self.combo.pack(pady=2)
        self.combo.bind("<<ComboboxSelected>>", self.cargar_formulario)

        self.frame_inputs = ttk.LabelFrame(
            root, text=" Notas del expediente de ESO (deja en blanco las que no tengas) ")
        self.frame_inputs.pack(padx=15, pady=10, fill="both", expand=True)
        self.cargar_formulario()

        botones = ttk.Frame(root)
        botones.pack(pady=8)
        ttk.Button(botones, text="📊 EVALUAR", command=self.calcular_prediccion).pack(side="left", padx=5)
        ttk.Button(botones, text="Limpiar", command=self.limpiar).pack(side="left", padx=5)
        root.bind('<Return>', lambda e: self.calcular_prediccion())

        self.frame_resultados = ttk.LabelFrame(root, text=" Diagnóstico del algoritmo ")
        self.frame_resultados.pack(padx=15, pady=10, fill="x")
        self.lbl_resultado = ttk.Label(self.frame_resultados, text="Esperando datos de entrada...",
                                       font=('Helvetica', 10), justify="center", wraplength=540)
        self.lbl_resultado.pack(pady=8, padx=10)

    def _clave_modalidad(self):
        return 'tecnologico' if "Tecnológico" in self.combo.get() else 'sociales'

    def cargar_formulario(self, event=None):
        for w in self.frame_inputs.winfo_children():
            w.destroy()
        self.entries = {}

        canvas = tk.Canvas(self.frame_inputs, borderwidth=0, highlightthickness=0)
        scrollbar = ttk.Scrollbar(self.frame_inputs, orient="vertical", command=canvas.yview)
        scroll_frame = ttk.Frame(canvas)
        scroll_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True, padx=5, pady=5)
        scrollbar.pack(side="right", fill="y")

        def rueda(e):
            paso = -1 if (getattr(e, "delta", 0) > 0 or getattr(e, "num", 0) == 4) else 1
            canvas.yview_scroll(paso, "units")

        def activar(_):
            for ev in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
                canvas.bind_all(ev, rueda)

        def desactivar(_):
            for ev in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
                canvas.unbind_all(ev)

        canvas.bind("<Enter>", activar)
        canvas.bind("<Leave>", desactivar)

        for fila, col in enumerate(self.paquete[self._clave_modalidad()]['columnas']):
            ttk.Label(scroll_frame, text=f"{obtener_nombre_humano(col)}:", width=42,
                      anchor="w").grid(row=fila, column=0, padx=8, pady=3, sticky="w")
            ent = ttk.Entry(scroll_frame, width=8, justify="center")
            ent.grid(row=fila, column=1, padx=8, pady=3)
            self.entries[col] = ent

    def limpiar(self):
        for ent in self.entries.values():
            ent.delete(0, tk.END)
        self.lbl_resultado.config(text="Esperando datos de entrada...")

    def calcular_prediccion(self):
        cfg = self.paquete[self._clave_modalidad()]
        notas, invalidas, n_validas = {}, [], 0

        for col, ent in self.entries.items():
            txt = ent.get().strip().replace(',', '.')
            if not txt:
                notas[col] = np.nan
                continue
            try:
                val = float(txt)
            except ValueError:
                invalidas.append(obtener_nombre_humano(col))
                continue
            if 0 <= val <= 10:
                notas[col] = val
                n_validas += 1
            else:
                invalidas.append(obtener_nombre_humano(col))

        if invalidas:
            messagebox.showwarning("Datos inválidos",
                                   "Usa números entre 0 y 10 (ej. 7.5) en:\n- " + "\n- ".join(invalidas[:8]))
            return

        total = len(cfg['columnas'])
        minimo = max(3, int(np.ceil(total * FRACCION_MIN_NOTAS_GUI)))
        if n_validas < minimo:
            messagebox.showwarning("Faltan datos",
                                   f"Introduce al menos {minimo} notas (llevas {n_validas}) "
                                   "para que la estimación tenga sentido.")
            return

        df = pd.DataFrame([notas], columns=cfg['columnas'])
        modelo = cfg['pipeline']
        pred = float(np.clip(modelo.predict(df)[0], 0, 10))
        feats = modelo.caracteristicas(df).iloc[0]
        sesgo, pendiente = float(feats['Sesgo_Cognitivo']), float(feats['Pendiente'])

        nivel = 80 if 80 in cfg['intervalos'] else max(cfg['intervalos'])
        radio = cfg['intervalos'][nivel]
        lo, hi = max(0.0, pred - radio), min(10.0, pred + radio)

        if sesgo > cfg['umbral_alto']:
            perfil = "Sus notas de ESO son relativamente más fuertes en materias científico-técnicas."
        elif sesgo < cfg['umbral_bajo']:
            perfil = "Sus notas de ESO son relativamente más fuertes en materias sociales y de letras."
        else:
            perfil = "Sus notas de ESO están equilibradas entre materias técnicas y sociales."

        if pendiente > UMBRAL_TENDENCIA:
            tendencia = f"Tendencia ASCENDENTE (+{pendiente:.2f} puntos por curso)"
        elif pendiente < -UMBRAL_TENDENCIA:
            tendencia = f"Tendencia DESCENDENTE ({pendiente:.2f} puntos por curso)"
        else:
            tendencia = "Tendencia ESTABLE entre cursos"

        fiabilidad = "alta" if n_validas / total >= 0.6 else "media"
        aviso = ""
        if cfg['mae'] >= cfg['mae_base']:
            aviso = "\n⚠️ Este modelo no mejora a la predicción trivial; úsalo con mucha cautela."

        self.lbl_resultado.config(text=(
            f"🎯 Nota media estimada: {pred:.2f}\n"
            f"Intervalo del {nivel} %: entre {lo:.2f} y {hi:.2f}\n"
            f"Modelo: {cfg['modelo_nombre']} · {cfg['n_alumnos']} alumnos · error medio {cfg['mae']:.2f}\n"
            f"Fiabilidad según notas introducidas ({n_validas}/{total}): {fiabilidad}\n\n"
            f"📈 {tendencia}\n"
            f"{perfil}\n"
            "Describe el perfil de notas; no indica qué modalidad le conviene. "
            "Es una estimación estadística, no una decisión: consulta al orientador del centro."
            f"{aviso}"))


# ==========================================
# EJECUCIÓN
# ==========================================
def main():
    ap = argparse.ArgumentParser(description="Orientador académico ML")
    ap.add_argument("--datos", default=os.environ.get("DATOS_BACHILLERATO", DIRECTORIO_DATOS_DEFECTO),
                    help="Carpeta con los CSV (o variable de entorno DATOS_BACHILLERATO)")
    ap.add_argument("--modelo", default=ARCHIVO_MODELO_DEFECTO, help="Ruta del modelo .joblib")
    ap.add_argument("--reentrenar", action="store_true", help="Fuerza el reentrenamiento")
    ap.add_argument("--columna-grupo", default=None,
                    help="Columna del CSV con la promoción/curso académico, para validar "
                         "entrenando con unas promociones y evaluando en otras")
    ap.add_argument("--privado", action="store_true",
                    help="Entrena solo modelos que no guardan notas de alumnos")
    args = ap.parse_args()

    root = tk.Tk()
    AplicacionPrediccion(root, args.modelo, args.datos, args.reentrenar,
                         args.columna_grupo, args.privado)
    root.mainloop()


if __name__ == "__main__":
    main()