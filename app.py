# app.py
# -*- coding: utf-8 -*-
import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
import json
import pickle
import io
import math
from typing import Dict, Any, List, Optional, Tuple
from sentence_transformers import SentenceTransformer
from bs4 import BeautifulSoup
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import groq
import faiss
import os
import warnings
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from dotenv import load_dotenv

# --- Cargar variables de entorno desde .env ---
load_dotenv()

# --- Constantes y Configuración ---
BASE_URL = "https://valencia.opendatasoft.com/api/explore/v2.1/"
INDEX_FILE = "faiss_opendata_valencia.idx"
METADATA_FILE = "faiss_metadata.pkl"
EMBEDDING_MODEL = 'all-mpnet-base-v2'
# LLM_MODEL = "llama3-70b-8192" # Groq model - No usado activamente en el planner/interpreter actual
GOOGLE_LLM_MODEL = "gemini-1.5-flash-latest"

# Obtener API Keys de variables de entorno
GROQ_API_KEY = os.getenv("API_KEY_GROQ")
API_KEY_GEMINI = os.getenv("API_KEY_GEMINI")

# --- Inicializar cliente Groq (opcional, si se usa)---
groq_client = None
if GROQ_API_KEY:
    try:
        groq_client = groq.Groq(api_key=GROQ_API_KEY)
        # st.sidebar.success("Cliente Groq inicializado.")
    except Exception as e:
        st.sidebar.error(f"Error inicializando Groq: {e}")
# else:
    # st.sidebar.warning("API_KEY_GROQ no encontrada. Funciones de LLM (Groq) pueden no estar disponibles.")

# Ignorar advertencias
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pandas")
warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")
warnings.filterwarnings("ignore", category=UserWarning, module="sentence_transformers")


# --- Configurar cliente de Google ---
gemini_model = None
if API_KEY_GEMINI:
    try:
        genai.configure(api_key=API_KEY_GEMINI)
        safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }
        gemini_model = genai.GenerativeModel(
            model_name=GOOGLE_LLM_MODEL,
            safety_settings=safety_settings
        )
        # st.sidebar.success("Cliente Gemini AI configurado.")
    except Exception as e:
        st.sidebar.error(f"Error configurando Google AI: {e}")
else:
    st.sidebar.warning("API_KEY_GEMINI no encontrada. Funciones de Gemini no disponibles.")


def sanitize_filename(filename: str) -> str:
    return re.sub(r'[<>:"/\\|?*]', '_', filename)

class FAISSIndex:
    def __init__(self, index_path: str = INDEX_FILE, metadata_path: str = METADATA_FILE):
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.index = None
        self.metadata = []
        self.load_index()

    def load_index(self):
        if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
            # st.info(f"Cargando índice FAISS desde {self.index_path}...") # Reduce verbosity
            self.index = faiss.read_index(self.index_path)
            # st.info(f"Cargando metadatos desde {self.metadata_path}...")
            with open(self.metadata_path, 'rb') as f:
                self.metadata = pickle.load(f)
            st.sidebar.success(f"Índice FAISS listo ({self.index.ntotal} vectores).")
        else:
            st.sidebar.warning("Archivos de índice no encontrados. Constrúyelo desde la barra lateral.")
            self.index = None

    def is_ready(self) -> bool:
        return self.index is not None and self.index.ntotal > 0

    def search(self, query_embedding: np.ndarray, top_k: int = 1) -> List[Dict[str, Any]]:
        if not self.is_ready():
            raise RuntimeError("El índice FAISS no está cargado o está vacío.")

        # Normalizar el embedding de la consulta aquí (vectores del índice ya están normalizados)
        # Asegurarse de que np.linalg.norm no sea cero para evitar división por cero
        norm = np.linalg.norm(query_embedding)
        if norm == 0: # Si el embedding es un vector nulo, no se puede normalizar
            # print("Warning: Query embedding has zero norm. Cannot normalize.") # Debug
            # Devuelve una lista vacía o maneja de otra manera, ya que la búsqueda fallará o será sin sentido
            return []

        query_embedding_norm = query_embedding / norm
        query_embedding_ready = query_embedding_norm.astype(np.float32).reshape(1, -1)

        distances, indices = self.index.search(query_embedding_ready, top_k)
        results = []
        if indices.size > 0:
            for i, idx in enumerate(indices[0]):
                if idx != -1:
                     results.append({
                         "metadata": self.metadata[idx],
                         # CRÍTICO: Asumimos que 'distances' es el producto escalar directo (mayor = mejor)
                         # Esto debe coincidir con cómo funcionaba en Colab.
                         "similarity": float(distances[0][i])
                     })
        return results

class APIQueryAgent:
    # Ajustar este umbral después de verificar que la similitud se calcula como en Colab
    SIMILARITY_THRESHOLD = 0.45 # Empezar aquí, Colab tenía 0.55. Ajustar según pruebas.

    def __init__(self, faiss_index: FAISSIndex):
        # Cachear el modelo para que no se recargue en cada instancia si es posible
        # (Streamlit podría re-ejecutar el script, así que @st.cache_resource podría ser mejor para el modelo)
        # Por ahora, lo dejamos así para simplicidad de la clase.
        self.model = SentenceTransformer(EMBEDDING_MODEL, device='cpu')
        self.faiss_index = faiss_index

    def get_embedding(self, text: str) -> np.ndarray:
        return self.model.encode(text, normalize_embeddings=False) # No normalizar aquí, se hace en FAISSIndex.search o al construir índice

    def search_dataset(self, query: str, top_k: int = 3) -> Optional[List[Dict[str, Any]]]:
        if not self.faiss_index.is_ready():
            st.error("Error: El índice FAISS no está listo para buscar.")
            return None

        # st.write(f"Vectorizando consulta: '{query}'") # UI
        query_embedding = self.get_embedding(query) # Obtiene el embedding (float32)

        # DEBUG en consola de Streamlit
        # print(f"DEBUG STREAMLIT Query Embedding (norm before search): {np.linalg.norm(query_embedding)}")
        # print(f"DEBUG STREAMLIT Query Embedding (dtype before search): {query_embedding.dtype}")
        # print(f"DEBUG STREAMLIT Query Embedding (first 5): {query_embedding[:5]}")


        try:
            # faiss_index.search se encargará de la normalización final y el formato
            results = self.faiss_index.search(query_embedding, top_k=top_k)
        except Exception as e:
             st.error(f"Error durante la búsqueda en FAISS: {e}")
             return None

        if not results:
            st.warning("No se encontró ningún dataset en el índice para la consulta.")
            return None

        # Ordenar por similitud descendente (mayor es mejor)
        results.sort(key=lambda x: x['similarity'], reverse=True)

        # st.info(f"Top {len(results)} candidatos encontrados (antes de validación):") # UI
        # for i, res in enumerate(results):
        #      title = res.get('metadata', {}).get('title', 'Sin Título')
        #      similarity = res.get('similarity', 0.0)
        #      st.write(f"  {i+1}. '{title}' (Similitud: {similarity:.4f})") # UI
        return results

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=6), retry=retry_if_exception_type(requests.exceptions.RequestException))
    def _fetch_api_data(self, endpoint: str, params: Dict = None) -> Optional[Dict]:
        try:
            response = requests.get(endpoint, params=params, timeout=20)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            # st.warning(f"⚠️ Error en la petición API a {endpoint}: {e}") # Evitar en funciones internas llamadas por build_index
            print(f"⚠️ Error en la petición API a {endpoint}: {e}") # Consola para build_index
            raise

    def get_all_datasets_metadata(self) -> List[Dict]:
        all_datasets_meta = []
        offset = 0
        limit = 100
        progress_bar = st.progress(0)
        status_text = st.empty()
        status_text.text("Obteniendo metadatos de todos los datasets desde la API...")
        total_datasets_for_progress = 1000 # Estimación inicial, se actualizará
        # datasets_processed_count = 0 # No es necesario si usamos offset y total_count

        while True:
            # status_text.text(f"Obteniendo datasets {offset} a {offset+limit-1}...")
            endpoint = f"{BASE_URL}catalog/datasets"
            params = {"limit": limit, "offset": offset} # Sin select, obtener todo
            data = self._fetch_api_data(endpoint, params)

            if not data or not data.get("results"):
                 # if not data: status_text.warning(f"Respuesta API vacía (offset {offset}).")
                 # elif not data.get("results"): status_text.info(f"Respuesta API sin 'results' (offset {offset}). Fin probable.")
                 break

            results = data["results"]
            for dataset in results:
                dataset_id = dataset.get("dataset_id")
                metas = dataset.get('metas', {})
                default_meta = metas.get('default', {})
                title = default_meta.get('title', 'Sin título')
                description_html = default_meta.get('description', '')
                if not dataset_id:
                    print(f"  Advertencia: Dataset sin ID (offset {offset}). Saltando.")
                    continue
                description_text = BeautifulSoup(description_html, "html.parser").get_text(separator=' ', strip=True) if description_html else ''
                all_datasets_meta.append({ "id": dataset_id, "title": title, "description": description_text })
                # datasets_processed_count += 1

            if data.get('total_count') and isinstance(data.get('total_count'), int):
                total_datasets_for_progress = data['total_count']
            
            current_progress = (offset + len(results)) / total_datasets_for_progress if total_datasets_for_progress > 0 else 0
            progress_bar.progress(min(1.0, current_progress))
            status_text.text(f"Obteniendo metadatos... {offset + len(results)}/{total_datasets_for_progress if total_datasets_for_progress else 'N/A'}")


            if len(results) < limit or (offset + limit) >= total_datasets_for_progress :
                 # status_text.info(f"Última página recibida (obtenidos: {len(results)}).")
                 break
            offset += limit

        progress_bar.empty()
        status_text.empty()
        st.success(f"Metadatos obtenidos para {len(all_datasets_meta)} datasets.")
        return all_datasets_meta

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10), retry=retry_if_exception_type(requests.exceptions.RequestException))
    def export_dataset(self, dataset_id: str) -> Optional[bytes]:
        # st.write(f"Descargando dataset '{dataset_id}' en formato CSV...") # UI
        endpoint = f"{BASE_URL}catalog/datasets/{dataset_id}/exports/csv"
        params = {"delimiter": ";"} # Probar con ; primero
        try:
            response = requests.get(endpoint, params=params, timeout=60)
            response.raise_for_status()
            if 'text/csv' in response.headers.get('Content-Type', ''):
                # st.success("Dataset descargado (con delimitador ';').") # UI
                return response.content
            else: # Si no es CSV con ';', probar sin especificar delimitador (la API usará coma por defecto)
                 # st.warning(f"Respuesta no CSV con ';'. Content-Type: {response.headers.get('Content-Type')}. Reintentando...") # UI
                 params = {}
                 response = requests.get(endpoint, params=params, timeout=60)
                 response.raise_for_status()
                 if 'text/csv' in response.headers.get('Content-Type', ''):
                     # st.success("Dataset descargado (con delimitador por defecto ',').") # UI
                     return response.content
                 else:
                     st.error(f"La respuesta sigue sin ser CSV. Content-Type: {response.headers.get('Content-Type')}")
                     return None
        except requests.exceptions.RequestException as e:
            st.error(f"Error descargando {dataset_id}: {e}")
            raise

# Cachear el modelo de sentence transformer para que no se recargue en cada ejecución de build_index
@st.cache_resource
def get_sentence_transformer_model(model_name):
    return SentenceTransformer(model_name, device='cpu')


def build_and_save_index(index_path: str = INDEX_FILE, metadata_path: str = METADATA_FILE):
    st.header("--- Creación/Actualización de Índice FAISS ---")
    
    # Crear una instancia temporal de FAISSIndex (aunque no cargará nada si los archivos no existen)
    # Esto es para que APIQueryAgent tenga un objeto faiss_index, aunque no se use para búsqueda aquí.
    # El índice real se construye y guarda al final de esta función.
    temp_faiss_placeholder = FAISSIndex(index_path, metadata_path) # No mostrará "listo" si no existe
    temp_faiss_placeholder.index = None # Asegurar que no se intente usar un índice viejo si load_index falló pero los archivos existen de una corrida anterior.

    # El agente usa el modelo cacheado
    agent_for_building = APIQueryAgent(temp_faiss_placeholder)
    # Sobrescribir el modelo del agente con el cacheado si es diferente
    agent_for_building.model = get_sentence_transformer_model(EMBEDDING_MODEL)


    datasets_metadata = agent_for_building.get_all_datasets_metadata()

    if not datasets_metadata:
        st.error("No se pudieron obtener metadatos. Abortando construcción de índice.")
        return

    st.write(f"Vectorizando {len(datasets_metadata)} datasets usando {EMBEDDING_MODEL}...")
    embeddings = []
    valid_metadata = []
    progress_bar_vector = st.progress(0)
    status_text_vector = st.empty()

    for i, meta in enumerate(datasets_metadata):
        text_to_embed = f"{meta['title']} {meta['description']}"
        # Usar el get_embedding del agent_for_building que tiene el modelo cacheado
        embedding = agent_for_building.get_embedding(text_to_embed)
        embeddings.append(embedding)
        valid_metadata.append(meta)
        if (i + 1) % 20 == 0 or (i + 1) == len(datasets_metadata):
            progress_percent = (i + 1) / len(datasets_metadata)
            progress_bar_vector.progress(progress_percent)
            status_text_vector.text(f"Vectorizados {i+1}/{len(datasets_metadata)}...")

    progress_bar_vector.empty()
    status_text_vector.empty()

    if not embeddings:
        st.error("No se generaron embeddings. Abortando construcción de índice.")
        return

    embeddings_np = np.array(embeddings).astype('float32')
    # Normalizar los embeddings ANTES de añadirlos al índice IP
    # Esto es crucial para que IndexFlatIP funcione como similitud coseno.
    faiss.normalize_L2(embeddings_np)

    # Obtener dimensión del modelo (el modelo ya está cargado en agent_for_building.model)
    embedding_dimension = agent_for_building.model.get_sentence_embedding_dimension()
    st.write(f"Creando índice FAISS (IndexFlatIP) con dimensión {embedding_dimension}...")
    index = faiss.IndexFlatIP(embedding_dimension) # Para producto interno (similitud coseno con vectores normalizados)
    index.add(embeddings_np)
    st.success(f"Índice creado con {index.ntotal} vectores.")

    st.write(f"Guardando índice en {index_path}...")
    faiss.write_index(index, index_path)
    st.write(f"Guardando metadatos en {metadata_path}...")
    with open(metadata_path, 'wb') as f:
        pickle.dump(valid_metadata, f)
    st.success("--- Proceso de Creación/Actualización de Índice Completado ---")
    st.balloons()
    # Forzar recarga del FAISSIndex global si existe o recargar la página para que FAISSIndex se actualice
    st.experimental_rerun()


class DatasetLoader:
     @staticmethod
     def load_dataset_from_bytes(dataset_bytes: bytes, dataset_title: str = "dataset") -> Optional[pd.DataFrame]:
        file_name_hint = sanitize_filename(dataset_title)
        df = None
        # st.write(f"Intentando cargar DataFrame '{file_name_hint}'...") # UI
        log_messages = []
        try:
            try:
                df = pd.read_csv(io.BytesIO(dataset_bytes), delimiter=';', encoding='utf-8', on_bad_lines='warn')
                if df.shape[1] <= 1 and len(df) > 0: # Si solo 1 col pero tiene filas, puede ser delimitador incorrecto
                    log_messages.append("Solo una columna con ';' y UTF-8. Podría ser delimitador incorrecto.")
                    raise ValueError("Posible delimitador incorrecto (';') o archivo no es CSV.")
                log_messages.append("Intentado con delimitador ';' y UTF-8.")
            except (pd.errors.ParserError, ValueError, UnicodeDecodeError) as e_utf8_semi:
                 log_messages.append(f"Fallo con ';' y UTF-8 ({e_utf8_semi}). Probando ',' y UTF-8...")
                 try:
                      df = pd.read_csv(io.BytesIO(dataset_bytes), delimiter=',', encoding='utf-8', on_bad_lines='warn')
                      log_messages.append("Intentado con delimitador ',' y UTF-8.")
                 except (pd.errors.ParserError, UnicodeDecodeError) as e_utf8_comma:
                      log_messages.append(f"Fallo con ',' y UTF-8 ({e_utf8_comma}). Probando ';' y latin1...")
                      try:
                           df = pd.read_csv(io.BytesIO(dataset_bytes), delimiter=';', encoding='latin1', on_bad_lines='warn')
                           if df.shape[1] <= 1 and len(df) > 0:
                               log_messages.append("Solo una columna con ';' y latin1. Podría ser delimitador incorrecto.")
                               raise ValueError("Posible delimitador incorrecto (';' latin1).")
                           log_messages.append("Intentado con delimitador ';' y latin1.")
                      except (pd.errors.ParserError, ValueError) as e_latin1_semi:
                           log_messages.append(f"Fallo con ';' y latin1 ({e_latin1_semi}). Probando ',' y latin1...")
                           try:
                                df = pd.read_csv(io.BytesIO(dataset_bytes), delimiter=',', encoding='latin1', on_bad_lines='warn')
                                log_messages.append("Intentado con delimitador ',' y latin1.")
                           except Exception as e_final:
                                st.error(f"Error final al parsear CSV '{file_name_hint}': {e_final}")
                                for msg in log_messages: st.caption(msg)
                                return None
            
            if df is None or df.empty:
                st.warning(f"DataFrame vacío o no parseado para '{file_name_hint}'.")
                for msg in log_messages: st.caption(msg)
                return None

            df.columns = [str(col).strip().replace(" ", "_").replace("(", "").replace(")", "").replace(":", "_").lower() for col in df.columns]
            # st.success(f"DataFrame '{file_name_hint}' cargado: {df.shape[0]} filas, {df.shape[1]} cols.") # UI
            # with st.expander("Log de Carga de CSV", expanded=False):
            #    for msg in log_messages: st.caption(msg)
            return df
        except pd.errors.EmptyDataError:
            st.error(f"Error al cargar '{file_name_hint}': Archivo CSV vacío.")
            return None
        except Exception as e:
            st.error(f"Error inesperado cargando '{file_name_hint}': {e}")
            return None


class DatasetAnalyzer:
    @staticmethod
    def analyze(df: pd.DataFrame) -> Dict[str, Any]:
        # st.write("Analizando estructura y estadísticas del DataFrame...") # UI
        analysis = { "numeric": [], "categorical": [], "temporal": [], "geospatial": [], "other": [],
                     "stats": None, "value_counts": {}, "temporal_range": {} }
        potential_temporal_keywords = ['fecha', 'date', 'año', 'ano', 'year', 'time', 'data', 'hora', 'dia', 'mes']
        potential_geo_keywords = ['geo', 'lat', 'lon', 'coord', 'wkt', 'point', 'shape', 'mapa', 'geopoint'] # Añadido geopoint
        df_copy = df.copy()

        conversion_log = []
        for col in df_copy.columns:
            original_dtype = df_copy[col].dtype
            # Intentar convertir a numérico si es objeto
            if df_copy[col].dtype == 'object':
                try:
                    # Reemplazar comas por puntos para decimales y luego convertir
                    df_copy[col] = pd.to_numeric(df_copy[col].str.replace(',', '.', regex=False).str.strip())
                    if original_dtype != df_copy[col].dtype: # Si el tipo cambió
                        conversion_log.append(f"Columna '{col}' (originalmente {original_dtype}) interpretada como numérica.")
                except (ValueError, TypeError, AttributeError):
                    pass # No es convertible a numérico o no es string para .str

            # Intentar convertir a datetime si es objeto o ya es datetime (para asegurar formato)
            is_already_datetime = pd.api.types.is_datetime64_any_dtype(df_copy[col].dtype)
            if df_copy[col].dtype == 'object' or is_already_datetime:
                 is_potential_temporal = any(key in col.lower() for key in potential_temporal_keywords)
                 if is_potential_temporal or is_already_datetime:
                     try:
                         original_non_nulls = df_copy[col].notna().sum()
                         if original_non_nulls == 0: continue

                         # Probar con dayfirst=True y luego False si la primera falla mucho
                         converted_col_dayfirst = pd.to_datetime(df_copy[col], errors='coerce', dayfirst=True, infer_datetime_format=False)
                         converted_col_standard = pd.to_datetime(df_copy[col], errors='coerce', dayfirst=False, infer_datetime_format=True) # infer_datetime_format es más rápido si funciona

                         # Elegir la conversión que resultó en más valores no nulos
                         success_rate_dayfirst = converted_col_dayfirst.notna().sum() / original_non_nulls if original_non_nulls > 0 else 0
                         success_rate_standard = converted_col_standard.notna().sum() / original_non_nulls if original_non_nulls > 0 else 0

                         if success_rate_dayfirst > 0.5 or success_rate_standard > 0.5:
                             if success_rate_dayfirst >= success_rate_standard:
                                 df_copy[col] = converted_col_dayfirst
                             else:
                                 df_copy[col] = converted_col_standard
                             
                             if not is_already_datetime and original_dtype != df_copy[col].dtype:
                                conversion_log.append(f"Columna '{col}' (originalmente {original_dtype}) convertida a datetime.")
                     except Exception: # Ignorar errores de conversión de datetime individuales
                         pass
        # if conversion_log:
        #     with st.expander("Log de Conversión de Tipos", expanded=False):
        #         for log_entry in conversion_log: st.caption(log_entry) # UI

        geo_extraction_log = []
        # Clasificar columnas después de conversiones
        for col in df_copy.columns:
            dtype = df_copy[col].dtype
            col_lower = col.lower()
            is_geo = False

            # Caso específico para 'geo_point_2d' u otros campos geoestructurados
            if any(key in col_lower for key in ['geo_point_2d', 'geopoint', 'geo_shape']): # OpenDataSoft y otros
                 analysis["geospatial"].append(col)
                 is_geo = True
                 # Intentar extraer lat/lon si no existen ya como columnas separadas y numéricas
                 # Esta modificación se hará en el DataFrame ORIGINAL `df` para que esté disponible para graficar
                 if not ('latitude' in df.columns and pd.api.types.is_numeric_dtype(df.get('latitude')) and
                         'longitude' in df.columns and pd.api.types.is_numeric_dtype(df.get('longitude'))):
                      try:
                           # Si es string "lat,lon"
                           if pd.api.types.is_string_dtype(df_copy[col]):
                                coords = df_copy[col].str.split(',', expand=True)
                                if coords.shape[1] == 2:
                                     df['latitude'] = pd.to_numeric(coords[0], errors='coerce')
                                     df['longitude'] = pd.to_numeric(coords[1], errors='coerce')
                                     geo_extraction_log.append(f"Extraídas 'latitude', 'longitude' desde string '{col}'.")
                           # Si es un objeto, podría ser un dict como {'lon': X, 'lat': Y} o {'latitude': Y, 'longitude': X}
                           elif pd.api.types.is_object_dtype(df_copy[col]) and df_copy[col].notna().any():
                                # Intentar convertir la primera entrada no nula para ver si es un diccionario
                                sample_val = df_copy[col].dropna().iloc[0]
                                if isinstance(sample_val, dict):
                                    lats = df_copy[col].apply(lambda x: x.get('lat') if isinstance(x, dict) else np.nan)
                                    lons = df_copy[col].apply(lambda x: x.get('lon') if isinstance(x, dict) else np.nan)
                                    if lats.notna().any() and lons.notna().any():
                                        df['latitude'] = pd.to_numeric(lats, errors='coerce')
                                        df['longitude'] = pd.to_numeric(lons, errors='coerce')
                                        geo_extraction_log.append(f"Extraídas 'latitude', 'longitude' desde dict en '{col}' (keys 'lat', 'lon').")
                                    else: # Probar con latitude/longitude
                                        lats_alt = df_copy[col].apply(lambda x: x.get('latitude') if isinstance(x, dict) else np.nan)
                                        lons_alt = df_copy[col].apply(lambda x: x.get('longitude') if isinstance(x, dict) else np.nan)
                                        if lats_alt.notna().any() and lons_alt.notna().any():
                                            df['latitude'] = pd.to_numeric(lats_alt, errors='coerce')
                                            df['longitude'] = pd.to_numeric(lons_alt, errors='coerce')
                                            geo_extraction_log.append(f"Extraídas 'latitude', 'longitude' desde dict en '{col}' (keys 'latitude', 'longitude').")

                           # Después de intentar extraer, verificar si se crearon y son numéricas
                           if 'latitude' in df.columns and pd.api.types.is_numeric_dtype(df['latitude']) and 'latitude' not in analysis["numeric"]:
                               analysis["numeric"].append('latitude') # Añadir a numéricas para el análisis
                           if 'longitude' in df.columns and pd.api.types.is_numeric_dtype(df['longitude']) and 'longitude' not in analysis["numeric"]:
                               analysis["numeric"].append('longitude')
                      except Exception as e:
                           geo_extraction_log.append(f"No se pudo extraer lat/lon de '{col}': {e}")
            # Buscar columnas que ya se llamen lat/lon
            elif any(key in col_lower for key in potential_geo_keywords):
                  if ('latitud' in col_lower or 'latitude' in col_lower) and pd.api.types.is_numeric_dtype(dtype):
                      analysis["geospatial"].append(col); is_geo = True
                  if ('longitud' in col_lower or 'longitude' in col_lower) and pd.api.types.is_numeric_dtype(dtype):
                       if col not in analysis["geospatial"]: analysis["geospatial"].append(col); is_geo = True
            if is_geo: continue # Si ya se clasificó como geo, no la añadas a otras categorías

            # Clasificación general
            if pd.api.types.is_numeric_dtype(dtype): analysis["numeric"].append(col)
            elif pd.api.types.is_datetime64_any_dtype(dtype):
                analysis["temporal"].append(col)
                try: analysis["temporal_range"][col] = (df_copy[col].min(), df_copy[col].max())
                except TypeError: analysis["temporal_range"][col] = (None, None) # Para NaT
            elif isinstance(dtype, pd.CategoricalDtype) or pd.api.types.is_object_dtype(dtype) or pd.api.types.is_string_dtype(dtype):
                analysis["categorical"].append(col)
                n_unique = df_copy[col].nunique()
                if n_unique < 100 and n_unique > 0:
                    try: analysis["value_counts"][col] = df_copy[col].value_counts().to_dict()
                    except TypeError: analysis["value_counts"][col] = {"error": "mixed types"}
            else: analysis["other"].append(col)

        # if geo_extraction_log:
        #     with st.expander("Log de Extracción Geoespacial", expanded=False):
        #         for log_entry in geo_extraction_log: st.caption(log_entry) # UI
        try:
            analysis["stats"] = df_copy.describe(include='all').to_dict()# datetime_is_numeric para incluir dtm en describe
            # st.success("Análisis de DataFrame completado.") # UI
        except Exception as e:
            st.warning(f"No se pudieron calcular estadísticas descriptivas: {e}")
            analysis["stats"] = {}

        analysis["stats"] = {
            k: {k2: (v2 if pd.notna(v2) else None) for k2, v2 in v.items()}
            for k, v in analysis["stats"].items()
        }
        analysis["temporal_range"] = {
            k: tuple((t.isoformat() if pd.notna(t) else None) for t in v)
            for k, v in analysis["temporal_range"].items()
        }
        return analysis


class LLMVisualizerPlanner:
    def suggest_visualizations(self, df_sample: pd.DataFrame, query: str, analysis: Dict) -> List[Dict]:
        if not gemini_model:
            st.error("Modelo Gemini no disponible para sugerir visualizaciones.")
            return []

        # st.write("Pidiendo sugerencias de visualización a Gemini...") # UI
        try: df_head_str = df_sample.head(3).to_markdown(index=False) if hasattr(df_sample.head(3), 'to_markdown') else df_sample.head(3).to_string()
        except Exception: df_head_str = df_sample.head(3).to_string()

        stats_summary = ""
        value_counts_summary = "Valores Comunes en Categóricas (Top 3 más frecuentes):\n"
        limited_value_counts = 0
        for col, counts in analysis.get("value_counts", {}).items():
             if limited_value_counts >= 5: break # Limitar cantidad de value_counts en el prompt
             if isinstance(counts, dict) and "error" not in counts:
                 top_items = list(counts.items())[:3]
                 value_counts_summary += f"- {col}: {', '.join([f'{k} ({v})' for k, v in top_items])}\n"
                 limited_value_counts +=1
        if not limited_value_counts: value_counts_summary = ""


        temporal_range_summary = ""
        has_temporal = False
        for col, (min_t, max_t) in analysis.get("temporal_range", {}).items():
             if min_t and max_t:
                 temporal_range_summary += f"- {col}: Desde {min_t} hasta {max_t}\n"
                 has_temporal = True
        if has_temporal:
            temporal_range_summary = f"Rangos Temporales Detectados:\n{temporal_range_summary}"
        else: temporal_range_summary = ""


        # Asegurar que las listas de columnas no sean demasiado largas para el prompt
        def shorten_list_for_prompt(col_list, max_items=15):
            if len(col_list) > max_items:
                return col_list[:max_items] + ["... (y otras)"]
            return col_list

        prompt = f"""
        Actúa como un analista de Business Intelligence experto. Tu objetivo es proponer las mejores visualizaciones de datos para responder a la consulta de un usuario, utilizando un dataset específico.

        Consulta del Usuario: "{query}"

        Dataset Resumido (primeras filas):
        Tiene {df_sample.shape[0]} filas (mostrando las primeras 3 de la muestra):
{df_head_str}

        Análisis de Columnas (USA ESTOS NOMBRES EXACTOS):
        - Numéricas: {shorten_list_for_prompt(analysis['numeric'])}
        - Categóricas: {shorten_list_for_prompt(analysis['categorical'])}
        - Temporales: {shorten_list_for_prompt(analysis['temporal'])}
        - Geoespaciales: {shorten_list_for_prompt(analysis['geospatial'])} (Si hay 'latitude' y 'longitude' en numéricas/geoespaciales, úsalas para mapas)
        - Otras: {shorten_list_for_prompt(analysis['other'])}

        {value_counts_summary}
        {temporal_range_summary}
        {stats_summary}

        Instrucciones:
        1.  Prioriza la **consulta del usuario**.
        2.  Usa **SOLAMENTE las columnas listadas** y sus tipos. Nombres EXACTOS. NO inventes columnas.
        3.  Si el dataset no parece contener datos directamente relevantes para la consulta, sugiere visualizaciones GENÉRICAS ÚTILES sobre los datos disponibles.
        4.  Sugiere entre 2 y 4 visualizaciones **útiles y variadas**.
        5.  Para cada visualización, proporciona:
            - "tipo_de_visualizacion": (String) ej: "histograma", "grafico de barras", "mapa de puntos", "grafico de lineas", "grafico circular", "diagrama de caja", "treemap", "mapa de calor".
            - "campos_involucrados": (Lista de strings) Nombres EXACTOS de columnas. Para mapas, incluye 'latitude' y 'longitude' si están disponibles. Para treemap, 'path' es una lista de columnas.
            - "titulo_de_la_visualizacion": (String) Título descriptivo.
            - "descripcion_utilidad": (String) Qué muestra y cómo ayuda.
            - "plotly_params": (Opcional, Dict) Parámetros para Plotly Express (ej: {{"color": "col_existente", "nbins": 20}}). Los valores DEBEN ser nombres de columnas existentes o valores numéricos/booleanos. Para mapas, "z" puede ser una columna numérica para mapas de calor/densidad.
        6.  Formato de Salida: **SOLAMENTE la lista JSON válida** ([{{...}}, {{...}}]). Sin texto fuera del JSON.

        Ejemplo de salida JSON:
        ```json
        [
          {{"tipo_de_visualizacion": "mapa de puntos", "campos_involucrados": ["latitude", "longitude", "nombre_evento"], "titulo_de_la_visualizacion": "Ubicación Geográfica de Eventos por Nombre", "descripcion_utilidad": "Visualiza la dispersión espacial de los eventos, coloreados por tipo.", "plotly_params": {{"color": "nombre_evento", "size": "magnitud_evento_si_existe"}}}},
          {{"tipo_de_visualizacion": "histograma", "campos_involucrados": ["columna_numerica_existente"], "titulo_de_la_visualizacion": "Distribución de Columna Numérica", "descripcion_utilidad": "Muestra la frecuencia de valores de la columna numérica.", "plotly_params": {{"nbins": 20}} }}
        ]
        ```
        Genera las sugerencias JSON:
        """
        raw_content = None; cleaned_json = None
        try:
            # DEBUG: Mostrar el prompt si es necesario
            # with st.expander("Ver Prompt enviado a Gemini (Planner)", expanded=False):
            #    st.text(prompt)

            generation_config = genai.types.GenerationConfig(
                response_mime_type="application/json", # Forzar JSON
                temperature=0.1, max_output_tokens=2048 # Baja temperatura para seguir instrucciones
            )
            response = gemini_model.generate_content(prompt, generation_config=generation_config)

            if not response.candidates:
                 st.warning(f"Respuesta de Gemini (Planner) fue bloqueada o no generó candidatos. Feedback: {response.prompt_feedback}")
                 return []

            # Acceso más robusto al texto de la respuesta
            if hasattr(response, 'text') and isinstance(response.text, str) and response.text.strip():
                raw_content = response.text
            elif response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
                raw_content = "".join(part.text for part in response.candidates[0].content.parts if hasattr(part, 'text'))
            else:
                st.warning("Respuesta de Gemini (Planner) no contiene texto extraíble.")
                # print(f"DEBUG GEMINI PLANNER RESPONSE: {response}") # Para consola
                return []

            # DEBUG: Mostrar respuesta cruda si es necesario
            # with st.expander("Ver Respuesta CRUDA de Gemini (Planner)", expanded=False):
            #    st.text(raw_content)

            if not raw_content or not raw_content.strip():
                st.warning("Respuesta de Gemini (Planner) vacía.")
                return []

            # Limpieza de JSON
            match_block = re.search(r'```json\s*([\s\S]*?)\s*```', raw_content, re.IGNORECASE)
            if match_block:
                cleaned_json = match_block.group(1).strip()
            else: # Si no hay bloque ```json, intentar parsear directamente si empieza con [ o {
                raw_content_strip = raw_content.strip()
                if raw_content_strip.startswith(('[', '{')):
                    cleaned_json = raw_content_strip
                else:
                    st.error("Respuesta de Gemini (Planner) no está en formato JSON esperado.")
                    st.text_area("Respuesta problemática (Planner):", raw_content, height=200)
                    return [] # No se puede parsear

            if not cleaned_json:
                 st.error("No se pudo extraer JSON de la respuesta (Planner).")
                 return []

            visualizations = json.loads(cleaned_json)
            if isinstance(visualizations, dict): # Si devuelve un dict con una clave que es la lista
                 keys = list(visualizations.keys())
                 if len(keys) == 1 and isinstance(visualizations[keys[0]], list):
                     visualizations = visualizations[keys[0]]
                 else: # Si es un diccionario pero no un wrapper, convertirlo en una lista de un solo elemento
                     visualizations = [visualizations]

            if not isinstance(visualizations, list):
                st.warning(f"Respuesta parseada de Gemini (Planner) no es una lista. Tipo: {type(visualizations)}")
                return []

            # st.success(f"Gemini sugirió {len(visualizations)} visualizaciones.") # UI
            return visualizations

        except json.JSONDecodeError as e:
          st.error(f"Error decodificando JSON de Gemini (Planner): {e}")
          if cleaned_json: st.text_area("JSON que falló al parsear (Planner):", cleaned_json, height=150)
          elif raw_content: st.text_area("Respuesta cruda original (Planner):", raw_content, height=150)
          return []
        except Exception as e:
          st.error(f"Error inesperado procesando sugerencias de Gemini (Planner): {e}")
          if raw_content: st.text_area("Respuesta cruda en error inesperado (Planner):", raw_content, height=150)
          # import traceback; print(traceback.format_exc()) # Para consola
          return []


class DatasetVisualizer:
    PLOT_FUNCTIONS = {
        "histograma": px.histogram, "grafico de barras": px.bar, "barras": px.bar,
        "grafico de lineas": px.line, "linea": px.line,
        "grafico de dispersion": px.scatter, "dispersion": px.scatter,
        "box plot": px.box, "boxplot": px.box, "diagrama de caja": px.box,
        "mapa de puntos": px.scatter_mapbox, "mapa de dispersion": px.scatter_mapbox,
        "grafico circular": px.pie, "circular": px.pie, "tarta": px.pie,
        "treemap": px.treemap,
        "mapa de calor": px.density_mapbox, "mapa de densidad": px.density_mapbox
    }
    AGGREGATION_THRESHOLD = 2000 # Umbral de filas para auto-agregar

    @staticmethod
    def normalize_chart_type(chart_type_raw: str) -> str:
        import unicodedata
        if not isinstance(chart_type_raw, str): return ""
        chart_type_lower = chart_type_raw.lower().strip()
        nfkd_form = unicodedata.normalize('NFKD', chart_type_lower)
        normalized = "".join([c for c in nfkd_form if not unicodedata.combining(c)])
        normalized = normalized.replace('grafico', 'grafico') # Estandarizar común
        return normalized

    @staticmethod
    def plot(df: pd.DataFrame, config: Dict) -> Optional[go.Figure]:
        chart_type_raw = config.get("tipo_de_visualizacion", "")
        chart_type = DatasetVisualizer.normalize_chart_type(chart_type_raw)
        campos_orig = config.get("campos_involucrados", [])
        # Filtrar campos para que solo se usen los que existen en el DataFrame
        campos = [c for c in campos_orig if c in df.columns]
        if not campos and len(campos_orig)>0: # Si había campos pero ninguno existe
            st.warning(f"Ninguno de los campos sugeridos ({campos_orig}) para '{chart_type_raw}' existe en el DataFrame. Columnas disponibles: {list(df.columns)}")
            return None
        if not campos and len(campos_orig)==0: # Si no se sugirió ningún campo
             st.warning(f"No se especificaron campos para '{chart_type_raw}'.")
             return None


        title = config.get("titulo_de_la_visualizacion", chart_type_raw)
        plotly_params_orig = config.get("plotly_params", {}) or {}

        plot_func = DatasetVisualizer.PLOT_FUNCTIONS.get(chart_type)
        if not plot_func:
             plot_func = DatasetVisualizer.PLOT_FUNCTIONS.get(chart_type_raw.lower().strip()) # Fallback
             if not plot_func:
                  st.warning(f"Tipo de visualización no soportado: '{chart_type_raw}' (Normalizado: '{chart_type}')")
                  return None

        df_processed = df.copy()
        plot_args = {"data_frame": df_processed, "title": title}
        axis_mapping = {}

        try:
            # --- Lógica de mapeo de ejes y validación de campos ---
            if chart_type == "histograma":
                 if campos and campos[0] in df_processed.columns: axis_mapping['x'] = campos[0]
                 else: raise ValueError(f"Campo X '{campos[0] if campos else 'N/A'}' no encontrado para {chart_type}")
            elif chart_type in ["grafico circular", "circular", "tarta"]:
                 if campos and campos[0] in df_processed.columns: axis_mapping['names'] = campos[0]
                 else: raise ValueError(f"Campo 'names' '{campos[0] if campos else 'N/A'}' no encontrado para {chart_type}")
                 if len(campos) > 1 and campos[1] in df_processed.columns: axis_mapping['values'] = campos[1]
            elif chart_type in ["grafico de barras", "barras", "grafico de lineas", "linea", "grafico de dispersion", "dispersion", "box plot", "boxplot", "diagrama de caja"]:
                if not campos: raise ValueError(f"No se proporcionaron campos para {chart_type}")
                if campos[0] not in df_processed.columns: raise ValueError(f"Campo X '{campos[0]}' no encontrado para {chart_type}")
                axis_mapping['x'] = campos[0]

                if len(campos) >= 2:
                    if campos[1] not in df_processed.columns: raise ValueError(f"Campo Y '{campos[1]}' no encontrado para {chart_type}")
                    axis_mapping['y'] = campos[1]
                elif chart_type in ['box plot', 'boxplot', 'diagrama de caja']: # Boxplot puede funcionar solo con Y (el primer campo)
                    axis_mapping['y'] = axis_mapping.pop('x')
                else: # Bar, line, scatter necesitan Y si X está presente
                    raise ValueError(f"Se necesitan 2 campos (X, Y) para {chart_type}, solo se proporcionó X: {campos[0]}")
            elif chart_type == "treemap":
                 path_cols = [c for c in campos if c in df_processed.columns] # Usar solo los campos que existen
                 if not path_cols: raise ValueError(f"Ninguno de los campos para 'path' ({campos}) encontrados en treemap")
                 axis_mapping['path'] = path_cols
            elif chart_type in ["mapa de puntos", "mapa de dispersion", "mapa de calor", "mapa de densidad"]:
                lat_col, lon_col = None, None
                if 'latitude' in df_processed.columns and 'longitude' in df_processed.columns and \
                   pd.api.types.is_numeric_dtype(df_processed['latitude']) and pd.api.types.is_numeric_dtype(df_processed['longitude']):
                    lat_col, lon_col = 'latitude', 'longitude'
                else: # Búsqueda más genérica si no existen 'latitude'/'longitude' exactas o no son numéricas
                    for c in df_processed.columns:
                        cl = c.lower()
                        if not lat_col and ('lat' in cl or 'ycoord' in cl) and pd.api.types.is_numeric_dtype(df_processed[c]): lat_col = c
                        if not lon_col and ('lon' in cl or 'xcoord' in cl) and pd.api.types.is_numeric_dtype(df_processed[c]): lon_col = c
                
                if lat_col and lon_col:
                    axis_mapping['lat'], axis_mapping['lon'] = lat_col, lon_col
                    plot_args['zoom'] = 10
                    plot_args['mapbox_style'] = "open-street-map"
                    if chart_type in ["mapa de calor", "mapa de densidad"]:
                        z_col = None
                        # Intentar obtener z de plotly_params si se especifica y existe
                        if 'z' in plotly_params_orig and plotly_params_orig['z'] in df_processed.columns and pd.api.types.is_numeric_dtype(df_processed[plotly_params_orig['z']]):
                            z_col = plotly_params_orig['z']
                        elif len(campos) >= 3 and campos[2] in df_processed.columns and pd.api.types.is_numeric_dtype(df_processed[campos[2]]):
                            z_col = campos[2] # Tercer campo involucrado
                        # Fallbacks
                        elif 'numplazas' in df_processed.columns and pd.api.types.is_numeric_dtype(df_processed['numplazas']): z_col = 'numplazas'
                        
                        if z_col and z_col in df_processed.columns:
                            axis_mapping['z'] = z_col
                            plot_args['radius'] = 8
                        # else: st.caption(f"Mapa de calor/densidad '{title}' sin columna 'z' explícita.") # UI
                else:
                    st.warning(f"No se encontraron columnas lat/lon numéricas adecuadas para mapa '{title}'.")
                    return None
            # --- Auto-agregación ---
            x_col_agg = axis_mapping.get('x')
            y_col_agg = axis_mapping.get('y')

            if chart_type in ["grafico de barras", "barras", "grafico de lineas", "linea"] and x_col_agg and y_col_agg:
                if x_col_agg in df_processed.columns and y_col_agg in df_processed.columns:
                    is_x_cat_or_time = (pd.api.types.is_string_dtype(df_processed[x_col_agg]) or
                                        pd.api.types.is_object_dtype(df_processed[x_col_agg]) or # Cat puede ser obj
                                        pd.api.types.is_datetime64_any_dtype(df_processed[x_col_agg]))
                    is_y_numeric = pd.api.types.is_numeric_dtype(df_processed[y_col_agg])

                    # Agregación si X es categórico/temporal, Y es numérico, y hay muchos puntos o X tiene alta cardinalidad
                    if is_x_cat_or_time and is_y_numeric and \
                       (len(df_processed) > DatasetVisualizer.AGGREGATION_THRESHOLD or df_processed[x_col_agg].nunique() > 75):
                        agg_func = 'sum' if chart_type in ["grafico de barras", "barras"] else 'mean'
                        # st.caption(f"Auto-agregando para '{title}': {len(df_processed)} filas. Agrupando '{x_col_agg}', '{agg_func}' de '{y_col_agg}'.") # UI
                        try:
                            df_agg = df_processed.groupby(x_col_agg, dropna=False)[y_col_agg].agg(agg_func).reset_index()
                            plot_args['data_frame'] = df_agg
                        except Exception as agg_e:
                             st.warning(f"Falló auto-agregación para '{title}': {agg_e}. Usando datos crudos.")
                             # plot_args['data_frame'] ya es df_processed (copia de df)
            
            # Procesar y validar plotly_params
            current_df_for_plot = plot_args['data_frame'] # DF que se usará (original o agregado)
            valid_plotly_params = {}
            param_aliases = {'nbinsx': 'nbins', 'nbinsy': 'nbins'}
            for key, value in plotly_params_orig.items():
                param_key = param_aliases.get(key.lower(), key)
                
                # Validar si el valor es una columna existente
                is_valid_col = isinstance(value, str) and value in current_df_for_plot.columns
                is_valid_col_list = isinstance(value, list) and all(isinstance(c, str) and c in current_df_for_plot.columns for c in value)
                is_non_col_param = not (isinstance(value, str) or isinstance(value,list)) # ej. nbins=20

                if (param_key == 'path' and is_valid_col_list) or \
                   (param_key != 'path' and is_valid_col) or \
                   is_non_col_param:
                    valid_plotly_params[param_key] = value
                # else: # Debug:
                #    print(f"Parámetro plotly_params '{key}:{value}' descartado para '{title}'.")

            plot_args.update(axis_mapping)
            plot_args.update(valid_plotly_params)

            # Validar que todas las columnas usadas finalmente existan en el DF que se va a graficar
            final_df_check = plot_args['data_frame']
            cols_to_verify = []
            for k_ax, v_ax in axis_mapping.items():
                if isinstance(v_ax, list): cols_to_verify.extend(v_ax)
                else: cols_to_verify.append(v_ax)
            for k_param, v_param in valid_plotly_params.items():
                 if isinstance(v_param, str) and v_param in final_df_check.columns: cols_to_verify.append(v_param)
                 elif isinstance(v_param, list): # ej path=['colA', 'colB'] or hover_data=['colC']
                     for item_in_list in v_param:
                         if isinstance(item_in_list, str) and item_in_list in final_df_check.columns:
                             cols_to_verify.append(item_in_list)
            
            for col_final in set(cols_to_verify): # Usar set para evitar duplicados
                if col_final not in final_df_check.columns:
                    raise ValueError(f"Columna '{col_final}' (de axis_mapping o plotly_params) no existe en el DataFrame final. Disp: {list(final_df_check.columns)}")
            
            # print(f"DEBUG Plot Args para '{title}': {plot_args}") # Consola
            fig = plot_func(**plot_args)
            return fig

        except ValueError as ve:
            st.warning(f"Error de configuración para gráfico '{title}' ({chart_type_raw}): {ve}")
            return None
        except Exception as e:
            st.error(f"Error inesperado generando gráfico '{title}' ({chart_type_raw}): {e}")
            # import traceback; st.text(traceback.format_exc()) # UI Debug
            return None


class LLMInterpreter:
     def generate_insights(self, query: str, analysis: Dict, viz_configs_generated: List[Dict], df_sample: pd.DataFrame, dataset_title: str) -> str:
        if not gemini_model:
            st.error("Modelo Gemini no disponible para generar insights.")
            return "Error: Modelo de lenguaje no disponible."

        # st.write("Generando insights basados en las visualizaciones con Gemini...") # UI
        if not viz_configs_generated: # Usar las que SÍ se generaron
            return "No se generaron visualizaciones válidas, por lo que no se pueden extraer insights."

        viz_summary = "Se generaron (o intentaron generar) las siguientes visualizaciones sobre este dataset:\n"
        for i, config in enumerate(viz_configs_generated):
               viz_summary += f"{i+1}. **{config.get('titulo_de_la_visualizacion', 'Sin título')}** (Tipo: {config.get('tipo_de_visualizacion', 'desconocido')}):\n"
               viz_summary += f"   - Campos usados: {config.get('campos_involucrados', [])}\n"
               viz_summary += f"   - Utilidad descrita por IA: {config.get('descripcion_utilidad', 'N/A')}\n"

        cols_limit = 10
        df_cols_list = list(df_sample.columns[:cols_limit])
        if len(df_sample.columns) > cols_limit: df_cols_list.append("...")
        data_summary = f"El dataset '{dataset_title}' (primeras filas de muestra) tiene columnas como: {df_cols_list}."

        def shorten_list_for_prompt(col_list, max_items=5): # Aún más corto para insights
            if len(col_list) > max_items:
                return col_list[:max_items] + ["..."]
            return col_list

        analysis_summary = (f"Análisis estructural: Numéricas: {shorten_list_for_prompt(analysis['numeric'])}, "
                            f"Categóricas: {shorten_list_for_prompt(analysis['categorical'])}, "
                            f"Temporales: {shorten_list_for_prompt(analysis['temporal'])}.")


        prompt = f"""
        Actúa como un analista de datos conciso.

        Contexto:
        - Consulta original del usuario: "{query}"
        - Dataset analizado: '{dataset_title}'. {data_summary}
        - {analysis_summary}
        - Visualizaciones generadas (o intentadas) y su propósito según la IA que las sugirió:
        {viz_summary}

        Tu Tarea:
        Basándote **EXCLUSIVAMENTE** en la consulta original y la descripción de las visualizaciones generadas (que provienen del dataset '{dataset_title}'), redacta un resumen muy breve (1-2 párrafos, MÁXIMO 100 palabras) con los **principales insights o conclusiones** que se pueden extraer **SOBRE LOS DATOS PRESENTADOS en relación a la consulta**.

        Instrucciones Críticas:
        - **NO inventes información** no respaldada por la descripción de las visualizaciones o la consulta.
        - **Céntrate en responder la consulta original del usuario**.
        - **Sintetiza**. No describas cada gráfico.
        - Si las visualizaciones no son suficientes para responder, menciónalo brevemente.
        - Resultado: **SOLO el texto del resumen en lenguaje natural**. Sin saludos, sin preámbulos, sin markdown.

        Genera el resumen de insights:
        """
        raw_content = None
        try:
            # with st.expander("Ver Prompt enviado a Gemini (Interpreter)", expanded=False):
            #    st.text(prompt) # UI

            generation_config = genai.types.GenerationConfig(temperature=0.4, max_output_tokens=350) # Un poco más creativo para insights
            response = gemini_model.generate_content(prompt, generation_config=generation_config)

            if not response.candidates:
                 st.warning(f"Respuesta de Gemini (Insights) fue bloqueada. Feedback: {response.prompt_feedback}")
                 return "No se pudieron generar insights debido a restricciones."

            if hasattr(response, 'text') and isinstance(response.text, str) and response.text.strip():
                raw_content = response.text
            elif response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
                raw_content = "".join(part.text for part in response.candidates[0].content.parts if hasattr(part, 'text'))
            else:
                st.warning("No se pudo extraer texto de la respuesta de Gemini (Insights).")
                return "No se generaron insights (respuesta sin texto)."


            if not raw_content or not raw_content.strip():
                st.warning("Respuesta de Gemini (Insights) vacía.")
                return "No se generaron insights."

            insights = re.sub(r"^(Aquí tienes|Claro,|Basado en|Finalmente|En resumen|De acuerdo|Según los datos)[,.:]?\s*", "", raw_content, flags=re.IGNORECASE).strip()
            return insights
        except Exception as e:
             st.error(f"Error inesperado procesando Gemini (Interpreter): {e}")
             if raw_content: st.text_area("Respuesta recibida de Gemini (Interpreter):", raw_content, height=100)
             return "Error al generar el resumen de insights."


def validate_dataset_relevance(query: str, dataset_title: str, dataset_description: str) -> bool:
    if not gemini_model:
        st.warning("Modelo Gemini no disponible para validación de relevancia. Asumiendo relevancia.")
        return True

    prompt = f"""
    Consulta del usuario: "{query}"
    Dataset encontrado: Título: "{dataset_title}"
    Descripción del dataset (primeros 500 caracteres): "{dataset_description[:500]}"

    Pregunta concisa: ¿Es este dataset ALTAMENTE relevante y MUY PROBABLEMENTE útil para responder directamente a la consulta del usuario?
    Considera palabras clave y tema. La descripción es más importante que el título si difieren.
    Responde únicamente con una sola palabra: 'Sí' o 'No'.
    """
    raw_response = None
    try:
        # with st.expander(f"Prompt Validación Relevancia para '{dataset_title}'", expanded=False): st.text(prompt)

        generation_config = genai.types.GenerationConfig(temperature=0.0, max_output_tokens=10)
        response = gemini_model.generate_content(prompt, generation_config=generation_config)

        if not response.candidates:
            st.warning(f"Respuesta Validación (Gemini) bloqueada. Feedback: {response.prompt_feedback}. Asumiendo relevancia.")
            return True
        
        if hasattr(response, 'text') and isinstance(response.text, str) and response.text.strip():
            raw_response = response.text
        elif response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
            raw_response = "".join(part.text for part in response.candidates[0].content.parts if hasattr(part, 'text'))
        else:
            st.warning("No se pudo extraer texto de Validación (Gemini). Asumiendo relevancia.")
            return True

        if not raw_response or not raw_response.strip():
            st.warning("Respuesta Validación (Gemini) vacía. Asumiendo relevancia.")
            return True

        decision = raw_response.strip().lower().replace('.', '').replace(',', '')
        # st.caption(f"Respuesta Validación LLM para '{dataset_title}': '{decision}'") # UI
        return decision == 'sí' or decision == 'si'

    except Exception as e:
        st.error(f"Error durante validación LLM: {e}. Asumiendo relevancia.")
        if raw_response: st.text_area("Respuesta Validación (Error):", raw_response, height=80)
        return True

# --- Global FAISSIndex instance (singleton-like for Streamlit session) ---
# Esto ayuda a que el índice se cargue una vez por sesión y esté disponible
@st.cache_resource
def get_faiss_index_instance():
    print("Attempting to load FAISSIndex instance...") # Debug console
    return FAISSIndex()


def run_full_dashboard_pipeline_st(user_query: str, faiss_index_instance: FAISSIndex):
    if not gemini_model:
        st.error("Modelo Gemini (Google AI) no está configurado o disponible. El pipeline no puede continuar.")
        st.info("Por favor, asegúrate de que la variable de entorno API_KEY_GEMINI está correctamente configurada en tu archivo .env y reinicia la aplicación.")
        return

    st.subheader(f"Procesando consulta: \"{user_query}\"")
    
    with st.spinner("Buscando y validando datasets..."):
        # faiss_index = FAISSIndex() # Carga el índice si existe (o usa el cacheado)
        faiss_index = faiss_index_instance # Usar la instancia cacheada

        if not faiss_index.is_ready():
            st.error("El índice FAISS no está listo. Por favor, constrúyelo primero usando el botón en la barra lateral.")
            return

        # APIQueryAgent necesita el modelo de embeddings, que también podemos cachear
        # Sin embargo, la instancia de APIQueryAgent se crea aquí, así que cargará su propio modelo ST
        # Para optimizar, el modelo ST también podría ser un @st.cache_resource pasado a APIQueryAgent.
        # Por ahora, se crea uno nuevo en cada pipeline run.
        api_agent = APIQueryAgent(faiss_index)
        search_results = api_agent.search_dataset(user_query, top_k=3) # Buscar los 3 mejores

        selected_dataset_info = None
        if search_results:
            with st.expander("Candidatos iniciales y validación", expanded=False):
                for result_idx, result in enumerate(search_results):
                    dataset_id = result["metadata"]["id"]
                    dataset_title = result["metadata"]["title"]
                    dataset_description = result["metadata"].get("description", "")
                    similarity_score = result['similarity']

                    st.markdown(f"--- Candidato {result_idx + 1}/{len(search_results)} ---")
                    st.markdown(f"**Evaluando:** `{dataset_title}` (Similitud: {similarity_score:.4f})")

                    if similarity_score < api_agent.SIMILARITY_THRESHOLD:
                        st.warning(f"  Similitud ({similarity_score:.4f}) < umbral ({api_agent.SIMILARITY_THRESHOLD}). Descartando.")
                        continue

                    if not validate_dataset_relevance(user_query, dataset_title, dataset_description):
                        st.warning(f"  Dataset '{dataset_title}' considerado NO relevante por LLM. Descartando.")
                        continue

                    st.success(f"Dataset '{dataset_title}' validado como relevante.")
                    selected_dataset_info = result
                    break # Encontramos uno bueno
        
    if not selected_dataset_info:
        st.error("No se encontró un dataset suficientemente relevante o validado para tu consulta.")
        if search_results:
            st.markdown("Los candidatos más cercanos fueron (pero no pasaron filtros):")
            for res in search_results:
                 st.caption(f"- '{res['metadata']['title']}' (Similitud: {res['similarity']:.4f})")
        st.markdown("""**Sugerencias:** *Intenta reformular tu consulta, o prueba una más general.*""")
        return

    dataset_id = selected_dataset_info["metadata"]["id"]
    dataset_title = selected_dataset_info["metadata"]["title"]
    st.success(f"🎯 Dataset Seleccionado: **{dataset_title}** (ID: {dataset_id})")

    df = None # Inicializar df
    with st.spinner(f"Descargando y cargando '{dataset_title}'..."):
        dataset_bytes = api_agent.export_dataset(dataset_id)
        if not dataset_bytes:
            st.error(f"No se pudo descargar '{dataset_title}'.")
            return
        df = DatasetLoader.load_dataset_from_bytes(dataset_bytes, dataset_title)
        if df is None or df.empty:
            st.error(f"Dataset '{dataset_title}' vacío o no pudo ser cargado como DataFrame.")
            return
        # Mostrar muestra del DF
        with st.expander(f"Primeras filas del dataset '{dataset_title}'", expanded=False):
            st.dataframe(df.head())

    analysis = None
    with st.spinner("Analizando el dataset..."):
        analyzer = DatasetAnalyzer()
        # El df se pasa por referencia y puede ser modificado por DatasetAnalyzer si extrae lat/lon
        analysis = analyzer.analyze(df)
        with st.expander("Ver Análisis Detallado del Dataset", expanded=False):
            # Convertir sets a listas para serialización JSON si es necesario, pero st.json puede manejarlo
            st.json(analysis, expanded=False)

    viz_configs_suggested = None
    with st.spinner("Generando sugerencias de visualización con IA..."):
        planner = LLMVisualizerPlanner()
        # Pasar una muestra más grande del df al planner, pero no todo si es masivo
        df_sample_for_planner = df.head(20) if len(df) > 20 else df
        viz_configs_suggested = planner.suggest_visualizations(df_sample_for_planner, user_query, analysis)

    figures = []
    valid_viz_configs_generated = []

    if viz_configs_suggested:
        st.subheader("📊 Visualizaciones Sugeridas y Generadas")
        visualizer = DatasetVisualizer()
        
        # Preparar para mostrar en columnas si son pocas gráficas
        # max_cols = 2 # Por ejemplo, máximo 2 columnas
        # num_viz_to_show = len(viz_configs_suggested)
        # st_cols = st.columns(min(num_viz_to_show, max_cols)) if num_viz_to_show > 0 else []

        for idx, config in enumerate(viz_configs_suggested):
            # current_col = st_cols[idx % len(st_cols)] if st_cols else st # Para distribuir en columnas
            # with current_col:
            with st.container(): # Usar container para agrupar mejor
                title_viz = config.get('titulo_de_la_visualizacion', f'Visualización Sugerida {idx+1}')
                st.markdown(f"**{idx+1}. {title_viz}**")
                st.caption(f"Tipo: {config.get('tipo_de_visualizacion', 'N/A')}. Campos: {config.get('campos_involucrados', 'N/A')}")
                st.caption(f"Descripción IA: {config.get('descripcion_utilidad', 'N/A')}")

                with st.spinner(f"Generando gráfico: {title_viz}..."):
                    fig = visualizer.plot(df, config) # Usar el df completo para graficar
                if fig:
                    try:
                        st.plotly_chart(fig, use_container_width=True)
                        figures.append(fig)
                        valid_viz_configs_generated.append(config)
                    except Exception as e_plot:
                        st.error(f"Error al mostrar el gráfico '{title_viz}': {e_plot}")
                else:
                    st.warning(f"No se pudo generar el gráfico: {title_viz}")
    else:
        st.info("La IA no sugirió ninguna visualización o hubo un error en el proceso.")

    if not figures: # Si después de intentar, no hay figuras
        st.warning("No se generaron visualizaciones válidas para mostrar.")

    with st.spinner("Generando insights con IA..."):
        st.subheader("💡 Insights del Analista Virtual")
        interpreter = LLMInterpreter()
        # Usar df.head() para la muestra en el prompt de insights
        insights_text = interpreter.generate_insights(user_query, analysis, valid_viz_configs_generated, df.head(), dataset_title)
        if "Error" in insights_text or not insights_text.strip():
            st.warning(insights_text if insights_text.strip() else "No se pudieron generar insights o la respuesta fue vacía.")
        else:
            st.markdown(insights_text)

    st.success("--- ✅ Pipeline Completado ---")
    st.balloons()


# --- Interfaz de Streamlit ---
st.set_page_config(layout="wide", page_title="Analista Datos Valencia")

# Cargar la instancia del índice FAISS una vez
faiss_index_global = get_faiss_index_instance()


st.title("BI Assistant - Open Data València")
st.markdown("""
Bienvenido al asistente para explorar los [Datos Abiertos del Ayuntamiento de Valencia](https://valencia.opendatasoft.com/pages/home/).
Escribe tu pregunta y el sistema intentará encontrar un dataset relevante, visualizarlo y ofrecerte insights.
""")

st.sidebar.header("Acciones del Índice")
if st.sidebar.button("Construir/Actualizar Índice FAISS", help="Descarga metadatos y crea/actualiza el índice para búsqueda semántica. Puede tardar varios minutos. La página se recargará al finalizar."):
    # No se puede usar st.spinner directamente aquí si build_and_save_index usa st.progress y otros
    build_and_save_index()
    # build_and_save_index llama a st.experimental_rerun() al final


# --- Área Principal ---
user_query = st.text_input("¿Qué datos te gustaría analizar o qué pregunta tienes sobre Valencia?",
                           placeholder="Ej: ¿Dónde hay estaciones de Valenbisi cerca del centro? o ¿Cómo ha evolucionado el tráfico en la Av. Blasco Ibáñez?")

if st.button("Analizar Consulta", type="primary"):
    if user_query:
        # Verificar API keys (Gemini es crucial para el pipeline actual)
        if not API_KEY_GEMINI or not gemini_model:
            st.error("API Key de Gemini (Google AI) no configurada o el modelo no se inicializó. Por favor, añádela al archivo .env, asegúrate que es válida y reinicia la aplicación.")
            st.stop()
        
        run_full_dashboard_pipeline_st(user_query, faiss_index_global)
    else:
        st.warning("Por favor, introduce una consulta.")

st.markdown("---")
st.caption("Desarrollado como demostración. Los resultados pueden variar.")