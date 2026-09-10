import streamlit as st
import joblib
import pandas as pd
import numpy as np

# ==========================================
# CONSTANTES (Las mismas que en tu código)
# ==========================================
DICCIONARIO_MATERIAS = {
    'EFI': 'Educación Física', 'EPV': 'Educación Plástica y Visual',
    'EPVA': 'Educación Plástica, Visual y Audiovisual', 'FIQ': 'Física y Química',
    'FQ': 'Física y Química', 'LCL': 'Lengua Castellana y Literatura',
    'MAT': 'Matemáticas', 'MAA': 'Matemáticas Académicas',
    'MAP': 'Matemáticas Aplicadas', 'MCS': 'Matemáticas Aplicadas a las CCSS',
    'GHI': 'Geografía e Historia', 'GEO': 'Geografía e Historia',
    'HIST': 'Historia', 'ING': 'Inglés', 'TIC': 'Tecnología / TIC',
    'TEC': 'Tecnología', 'LAT': 'Latín', 'BIO': 'Biología y Geología',
    'BGE': 'Biología y Geología', 'MUS': 'Música', 'FRA': 'Francés',
    'REL': 'Religión', 'VAE': 'Valores Éticos', 'ECO': 'Economía',
    'DTT': 'Dibujo Técnico', 'DT': 'Dibujo Técnico', 'FIL': 'Filosofía',
}

MATERIAS_TECNICAS = ['FIQ', 'MAA', 'MCS', 'MAT', 'TIC', 'BGE', 'BIO', 'TECN', 'FQ', 'DT', 'DTT']
MATERIAS_SOCIALES = ['LCL', 'GHI', 'ING', 'LAT', 'GEO', 'FRA', 'ECO', 'HIST', 'FIL']

def obtener_nombre_humano(codigo):
    if not codigo: return ""
    codigo_str = str(codigo).strip()
    if codigo_str.lower() in ['media_eso', 'mediaeso', 'promedio_eso']: return "Nota Media de la ESO"
    if "_" in codigo_str:
        partes = codigo_str.split("_", 1)
        curso = f"{partes[0][1]}º ESO" if partes[0].startswith("E") else f"{partes[0][1]}º Bachillerato" if partes[0].startswith("B") else partes[0]
        nombre_materia = DICCIONARIO_MATERIAS.get(partes[1].upper(), partes[1].upper())
        return f"{nombre_materia} ({curso})" if curso else nombre_materia
    cu = codigo_str.upper()
    if cu in DICCIONARIO_MATERIAS: return DICCIONARIO_MATERIAS[cu]
    if len(cu) > 2 and cu[0] in ['E', 'B'] and cu[1].isdigit():
        nombre_materia = DICCIONARIO_MATERIAS.get(cu[2:], cu[2:])
        return f"{nombre_materia} ({cu[1]}º {'ESO' if cu[0]=='E' else 'Bachillerato'})"
    return codigo_str

def aplicar_ingenieria_caracteristicas(df):
    df_nuevo = df.copy()
    cols_tec = [c for c in df_nuevo.columns if any(m in c for m in MATERIAS_TECNICAS)]
    cols_soc = [c for c in df_nuevo.columns if any(m in c for m in MATERIAS_SOCIALES)]
    df_nuevo['Perfil_Tecnico'] = df_nuevo[cols_tec].mean(axis=1) if cols_tec else df_nuevo.mean(axis=1)
    df_nuevo['Perfil_Social'] = df_nuevo[cols_soc].mean(axis=1) if cols_soc else df_nuevo.mean(axis=1)
    df_nuevo['Media_Global'] = df_nuevo.mean(axis=1)
    df_nuevo['Sesgo_Cognitivo'] = df_nuevo['Perfil_Tecnico'] - df_nuevo['Perfil_Social']
    return df_nuevo

# ==========================================
# CONFIGURACIÓN DE LA PÁGINA WEB
# ==========================================
st.set_page_config(page_title="Orientador Académico IA", page_icon="🎓", layout="centered")

st.title("🎓 Sistema Orientador Académico Basado en ML")
st.write("Introduce tus notas de la ESO para que la Inteligencia Artificial estime tu rendimiento en Bachillerato.")

# Cargar modelos en caché para que la web sea rápida
@st.cache_resource
def cargar_modelos():
    return joblib.load("modelos_bachillerato.joblib")

try:
    paquete = cargar_modelos()
except FileNotFoundError:
    st.error("Error: No se encuentra el archivo 'modelos_bachillerato.joblib'. Asegúrate de subirlo.")
    st.stop()

# Interfaz de usuario
modalidad = st.selectbox(
    "Modalidad de Bachillerato a consultar:",
    ["Bachillerato Tecnológico / Científico", "Bachillerato Ciencias Sociales / Humanidades"]
)

clave_config = 'tecnologico' if "Tecnológico" in modalidad else 'sociales'
config = paquete[clave_config]

st.subheader("Expediente Académico (ESO)")
st.info("Deja en 0.0 las asignaturas que no hayas cursado. El sistema calculará la equivalencia.")

notas_usuario = {}
cols = st.columns(2) # Dividir la pantalla en dos columnas para que quede bonito

# Generar casillas para introducir notas
for i, col_name in enumerate(config['columnas_base']):
    nombre_legible = obtener_nombre_humano(col_name)
    with cols[i % 2]: # Alternar entre la columna izquierda y derecha
        val = st.number_input(f"{nombre_legible}:", min_value=0.0, max_value=10.0, value=0.0, step=0.25)
        if val > 0:
            notas_usuario[col_name] = val
        else:
            notas_usuario[col_name] = np.nan # Si deja 0.0, lo tomamos como no cursado

if st.button("📊 EVALUAR Y OBTENER DIAGNÓSTICO", use_container_width=True):
    if all(pd.isna(v) for v in notas_usuario.values()):
        st.warning("⚠️ Debes introducir al menos una nota válida.")
    else:
        with st.spinner('Analizando perfil cognitivo...'):
            df_raw = pd.DataFrame([notas_usuario], columns=config['columnas_base'])
            df_imp = pd.DataFrame(config['imputer'].transform(df_raw), columns=config['columnas_base'])
            df_feat = aplicar_ingenieria_caracteristicas(df_imp)
            X_scl = config['scaler'].transform(df_feat)

            pred = np.clip(config['modelo'].predict(X_scl)[0], 0, 10)
            sesgo = df_feat['Sesgo_Cognitivo'].iloc[0]

            if sesgo > 0.6: 
                orientacion = "Perfil detectado: **MARCADO CIENTÍFICO-TECNOLÓGICO**.\n\nExiste una alta correlación positiva para Ciencias."
            elif sesgo < -0.6: 
                orientacion = "Perfil detectado: **MARCADO SOCIAL-HUMANÍSTICO**.\n\nEl rendimiento histórico sugiere inclinación hacia Humanidades."
            else: 
                orientacion = "Perfil detectado: **EQUILIBRADO / MULTIDISCIPLINAR**.\n\nAusencia de sesgos significativos. Flexibilidad para ambas ramas."

            st.success(f"### 🎯 Nota Media Estimada: {pred:.2f} sobre 10")
            st.markdown(orientacion)