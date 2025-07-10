# app.py
# -*- coding: utf-8 -*-
import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
# from plotly.subplots import make_subplots # No se usa directamente ahora
import re
import json
import pickle
import io
# import math # No se usa directamente ahora
from typing import Dict, Any, List, Optional # Eliminado Tuple si no se usa
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
GOOGLE_LLM_MODEL = "gemini-1.5-flash-latest"

# Obtener API Keys de variables de entorno
GROQ_API_KEY = os.getenv("API_KEY_GROQ")
API_KEY_GEMINI = os.getenv("API_KEY_GEMINI")

# --- Inicializar cliente Groq (opcional)---
groq_client = None
if GROQ_API_KEY:
    try:
        groq_client = groq.Groq(api_key=GROQ_API_KEY)
    except Exception as e:
        st.sidebar.error(f"Error inicializando Groq: {e}")

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
    except Exception as e:
        st.sidebar.error(f"Error configurando Google AI: {e}. Verifica tu API Key y la configuración del proyecto.")
# else: 
    # st.sidebar.warning("API_KEY_GEMINI no encontrada. Funciones de Gemini no disponibles.")


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
        # self.index_path y self.metadata_path ya están configurados por __init__
        if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
            self.index = faiss.read_index(self.index_path)
            with open(self.metadata_path, 'rb') as f:
                self.metadata = pickle.load(f)
            # st.sidebar.success(...) # Comenta o quita los st.sidebar de aquí si dan problemas en el script
        else:
            # st.sidebar.warning(...) # Comenta o quita los st.sidebar de aquí
            self.index = None

    def is_ready(self) -> bool:
        return self.index is not None and self.index.ntotal > 0

    def search(self, query_embedding: np.ndarray, top_k: int = 1) -> List[Dict[str, Any]]:
        if not self.is_ready():
            raise RuntimeError("El índice FAISS no está cargado o está vacío.")
        norm = np.linalg.norm(query_embedding)
        if norm == 0: return []
        query_embedding_norm = query_embedding / norm
        query_embedding_ready = query_embedding_norm.astype(np.float32).reshape(1, -1)
        distances, indices = self.index.search(query_embedding_ready, top_k)
        results = []
        if indices.size > 0:
            for i, idx_val in enumerate(indices[0]): # Renombrado idx a idx_val para evitar conflicto
                if idx_val != -1:
                     results.append({
                         "metadata": self.metadata[idx_val],
                         "similarity": float(distances[0][i])
                     })
        return results

class APIQueryAgent:
    SIMILARITY_THRESHOLD = 0.45

    def __init__(self, faiss_index: FAISSIndex, sentence_model): # Recibe el modelo
        self.model = sentence_model # Usa el modelo cacheado
        self.faiss_index = faiss_index

    def get_embedding(self, text: str) -> np.ndarray:
        return self.model.encode(text, normalize_embeddings=False)

    def search_dataset(self, query: str, top_k: int = 3) -> Optional[List[Dict[str, Any]]]:
        if not self.faiss_index.is_ready():
            st.error("Error: El índice FAISS no está listo para buscar.")
            return None
        query_embedding = self.get_embedding(query)
        try:
            results = self.faiss_index.search(query_embedding, top_k=top_k)
        except Exception as e:
             st.error(f"Error durante la búsqueda en FAISS: {e}")
             return None
        if not results:
            st.warning("No se encontró ningún dataset en el índice para la consulta.")
            return None
        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=6), retry=retry_if_exception_type(requests.exceptions.RequestException))
    def _fetch_api_data(self, endpoint: str, params: Dict = None) -> Optional[Dict]:
        try:
            response = requests.get(endpoint, params=params, timeout=20)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"⚠️ Error en la petición API a {endpoint}: {e}")
            raise

    def get_all_datasets_metadata(self) -> List[Dict]:
        all_datasets_meta = []
        offset = 0
        limit = 100
        progress_bar = st.progress(0)
        status_text = st.empty()
        status_text.text("Obteniendo metadatos de todos los datasets desde la API...")
        total_datasets_for_progress = 0 # Se actualizará con la primera respuesta
        first_call = True

        while True:
            endpoint = f"{BASE_URL}catalog/datasets"
            params = {"limit": limit, "offset": offset}
            data = self._fetch_api_data(endpoint, params)

            if not data: break # Error en fetch o respuesta vacía
            if first_call:
                total_datasets_for_progress = data.get('total_count', 0)
                if not isinstance(total_datasets_for_progress, int): total_datasets_for_progress = 0
                first_call = False

            results = data.get("results", [])
            if not results and offset > 0 : # Si no hay resultados y no es la primera llamada, fin.
                 break

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

            current_fetched_count = offset + len(results)
            if total_datasets_for_progress > 0:
                progress_percent = min(1.0, current_fetched_count / total_datasets_for_progress)
                progress_bar.progress(progress_percent)
            status_text.text(f"Obteniendo metadatos... {current_fetched_count}/{total_datasets_for_progress if total_datasets_for_progress > 0 else 'N/A'}")

            if len(results) < limit or (total_datasets_for_progress > 0 and current_fetched_count >= total_datasets_for_progress) :
                 break
            offset += limit

        progress_bar.empty()
        status_text.empty()
        st.success(f"Metadatos obtenidos para {len(all_datasets_meta)} datasets.")
        return all_datasets_meta

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10), retry=retry_if_exception_type(requests.exceptions.RequestException))
    def export_dataset(self, dataset_id: str) -> Optional[bytes]:
        endpoint = f"{BASE_URL}catalog/datasets/{dataset_id}/exports/csv"
        params = {"delimiter": ";"}
        try:
            response = requests.get(endpoint, params=params, timeout=60)
            response.raise_for_status()
            if 'text/csv' in response.headers.get('Content-Type', ''):
                return response.content
            else:
                 params = {} # Probar sin delimitador (usará coma por defecto)
                 response = requests.get(endpoint, params=params, timeout=60)
                 response.raise_for_status()
                 if 'text/csv' in response.headers.get('Content-Type', ''):
                     return response.content
                 else:
                     st.error(f"La respuesta sigue sin ser CSV. Content-Type: {response.headers.get('Content-Type')}")
                     return None
        except requests.exceptions.RequestException as e:
            st.error(f"Error descargando {dataset_id}: {e}")
            raise

@st.cache_resource
def get_sentence_transformer_model(model_name):
    # print(f"Loading SentenceTransformer model: {model_name}") # Debug
    return SentenceTransformer(model_name, device='cpu')


def build_and_save_index(index_path: str = INDEX_FILE, metadata_path: str = METADATA_FILE):
    st.header("--- Creación/Actualización de Índice FAISS ---")
    
    # Usar el modelo cacheado para la construcción del índice
    sentence_model_instance = get_sentence_transformer_model(EMBEDDING_MODEL)
    # Pasar un FAISSIndex placeholder, no se usará para búsqueda aquí.
    agent_for_building = APIQueryAgent(FAISSIndex(index_path, metadata_path), sentence_model_instance)

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
    faiss.normalize_L2(embeddings_np) # Normalizar para IndexFlatIP (producto escalar)

    embedding_dimension = agent_for_building.model.get_sentence_embedding_dimension()
    st.write(f"Creando índice FAISS (IndexFlatIP) con dimensión {embedding_dimension}...")
    index = faiss.IndexFlatIP(embedding_dimension)
    index.add(embeddings_np)
    st.success(f"Índice creado con {index.ntotal} vectores.")

    st.write(f"Guardando índice en {index_path}...")
    faiss.write_index(index, index_path)
    st.write(f"Guardando metadatos en {metadata_path}...")
    with open(metadata_path, 'wb') as f:
        pickle.dump(valid_metadata, f)
    st.success("--- Proceso de Creación/Actualización de Índice Completado ---")
    st.balloons()
    st.info("Índice actualizado. La aplicación usará el nuevo índice en la próxima consulta o si refrescas la página.")
    # No st.rerun() aquí para evitar problemas con callbacks de la sidebar.
    # Se confía en que get_faiss_index_instance() recargará al cambiar los archivos.


class DatasetLoader:
     @staticmethod
     def load_dataset_from_bytes(dataset_bytes: bytes, dataset_title: str = "dataset") -> Optional[pd.DataFrame]:
        file_name_hint = sanitize_filename(dataset_title)
        df = None
        log_messages = []
        try:
            try:
                df = pd.read_csv(io.BytesIO(dataset_bytes), delimiter=';', encoding='utf-8', on_bad_lines='warn')
                log_messages.append("Intentado con delimitador ';' y UTF-8.")
                if df.shape[1] <= 1 and len(df) > 0:
                    raise ValueError("Posible delimitador incorrecto (';') o archivo no es CSV.")
            except (pd.errors.ParserError, ValueError, UnicodeDecodeError):
                 log_messages.append("Fallo con ';' y UTF-8. Probando ',' y UTF-8...")
                 try:
                      df = pd.read_csv(io.BytesIO(dataset_bytes), delimiter=',', encoding='utf-8', on_bad_lines='warn')
                      log_messages.append("Intentado con delimitador ',' y UTF-8.")
                      if df.shape[1] <= 1 and len(df) > 0: # Chequear también para coma
                          raise ValueError("Posible delimitador incorrecto (',') o archivo no es CSV.")
                 except (pd.errors.ParserError, ValueError, UnicodeDecodeError):
                      log_messages.append("Fallo con ',' y UTF-8. Probando ';' y latin1...")
                      try:
                           df = pd.read_csv(io.BytesIO(dataset_bytes), delimiter=';', encoding='latin1', on_bad_lines='warn')
                           log_messages.append("Intentado con delimitador ';' y latin1.")
                           if df.shape[1] <= 1 and len(df) > 0:
                               raise ValueError("Posible delimitador incorrecto (';' latin1).")
                      except (pd.errors.ParserError, ValueError, UnicodeDecodeError):
                           log_messages.append("Fallo con ';' y latin1. Probando ',' y latin1...")
                           try:
                                df = pd.read_csv(io.BytesIO(dataset_bytes), delimiter=',', encoding='latin1', on_bad_lines='warn')
                                log_messages.append("Intentado con delimitador ',' y latin1.")
                                if df.shape[1] <= 1 and len(df) > 0:
                                    raise ValueError("Posible delimitador incorrecto (',' latin1).")
                           except Exception as e_final:
                                st.error(f"Error final al parsear CSV '{file_name_hint}': {e_final}")
                                for msg in log_messages: st.caption(msg)
                                return None
            
            if df is None or df.empty:
                st.warning(f"DataFrame vacío o no parseado para '{file_name_hint}'.")
                for msg in log_messages: st.caption(msg)
                return None

            df.columns = [str(col).strip().replace(" ", "_").replace("(", "").replace(")", "").replace(":", "_").lower() for col in df.columns]
            return df
        except pd.errors.EmptyDataError:
            st.error(f"Error al cargar '{file_name_hint}': Archivo CSV vacío.")
            return None
        except Exception as e:
            st.error(f"Error inesperado cargando '{file_name_hint}': {e}")
            return None


class DatasetAnalyzer:
    @staticmethod
    def analyze(df: pd.DataFrame) -> Dict[str, Any]: # df se modifica por referencia si se añaden lat/lon
        analysis = { "numeric": [], "categorical": [], "temporal": [], "geospatial": [], "other": [],
                     "stats": None, "value_counts": {}, "temporal_range": {} }
        potential_temporal_keywords = ['fecha', 'date', 'año', 'ano', 'year', 'time', 'data', 'hora', 'dia', 'mes']
        potential_geo_keywords = ['geo', 'lat', 'lon', 'coord', 'wkt', 'point', 'shape', 'mapa', 'geopoint']
        df_copy = df.copy() # Trabajar con copia para análisis, pero modificar df original para lat/lon

        # Conversión de tipos
        for col in df_copy.columns:
            original_dtype = df_copy[col].dtype
            if df_copy[col].dtype == 'object':
                try:
                    df_copy[col] = pd.to_numeric(df_copy[col].str.replace(',', '.', regex=False).str.strip())
                except (ValueError, TypeError, AttributeError): pass
            is_already_datetime = pd.api.types.is_datetime64_any_dtype(df_copy[col].dtype)
            if df_copy[col].dtype == 'object' or is_already_datetime:
                 if any(key in col.lower() for key in potential_temporal_keywords) or is_already_datetime:
                     try:
                         original_non_nulls = df_copy[col].notna().sum(); prev_dtype = df_copy[col].dtype
                         if original_non_nulls == 0: continue
                         converted_col_dayfirst = pd.to_datetime(df_copy[col], errors='coerce', dayfirst=True, infer_datetime_format=False)
                         converted_col_standard = pd.to_datetime(df_copy[col], errors='coerce', dayfirst=False, infer_datetime_format=True)
                         success_rate_dayfirst = converted_col_dayfirst.notna().sum() / original_non_nulls if original_non_nulls > 0 else 0
                         success_rate_standard = converted_col_standard.notna().sum() / original_non_nulls if original_non_nulls > 0 else 0
                         if success_rate_dayfirst > 0.5 or success_rate_standard > 0.5:
                             df_copy[col] = converted_col_dayfirst if success_rate_dayfirst >= success_rate_standard else converted_col_standard
                     except Exception: pass
        
        # Clasificación de columnas y extracción geo
        # Definir lat_col_name y lon_col_name para consistencia
        lat_col_name, lon_col_name = 'latitude', 'longitude'

        for col in df_copy.columns: # Iterar sobre la copia para análisis
            dtype = df_copy[col].dtype; col_lower = col.lower(); is_geo_classified = False
            
            # Si ya existen 'latitude' y 'longitude' numéricas en el df original, marcarlas
            if col == lat_col_name and pd.api.types.is_numeric_dtype(df.get(lat_col_name)):
                if col not in analysis["numeric"]: analysis["numeric"].append(col)
                if col not in analysis["geospatial"]: analysis["geospatial"].append(col)
                is_geo_classified = True
            if col == lon_col_name and pd.api.types.is_numeric_dtype(df.get(lon_col_name)):
                if col not in analysis["numeric"]: analysis["numeric"].append(col)
                if col not in analysis["geospatial"]: analysis["geospatial"].append(col)
                is_geo_classified = True
            
            if is_geo_classified: continue

            # Extracción de geo_point_2d u otros campos geoestructurados
            if any(key in col_lower for key in ['geo_point_2d', 'geopoint', 'geo_shape']):
                 analysis["geospatial"].append(col) # Añadir la columna original a geoespaciales
                 is_geo_classified = True
                 # Intentar extraer y AÑADIR/SOBREESCRIBIR 'latitude' y 'longitude' en el DataFrame ORIGINAL `df`
                 if not (lat_col_name in df.columns and pd.api.types.is_numeric_dtype(df[lat_col_name]) and \
                         lon_col_name in df.columns and pd.api.types.is_numeric_dtype(df[lon_col_name])):
                      try:
                           if pd.api.types.is_string_dtype(df_copy[col]): # Usar df_copy para leer
                                coords = df_copy[col].str.split(',', expand=True)
                                if coords.shape[1] == 2:
                                     df[lat_col_name] = pd.to_numeric(coords[0], errors='coerce') # Modificar df original
                                     df[lon_col_name] = pd.to_numeric(coords[1], errors='coerce')
                           elif pd.api.types.is_object_dtype(df_copy[col]) and df_copy[col].notna().any():
                                sample_val = df_copy[col].dropna().iloc[0]
                                if isinstance(sample_val, dict):
                                    lats = df_copy[col].apply(lambda x: x.get('lat', x.get('latitude')) if isinstance(x, dict) else np.nan)
                                    lons = df_copy[col].apply(lambda x: x.get('lon', x.get('longitude')) if isinstance(x, dict) else np.nan)
                                    if lats.notna().any() and lons.notna().any():
                                        df[lat_col_name] = pd.to_numeric(lats, errors='coerce')
                                        df[lon_col_name] = pd.to_numeric(lons, errors='coerce')
                           # Después de intentar extraer, si se crearon y son numéricas, añadirlas al análisis
                           if lat_col_name in df.columns and pd.api.types.is_numeric_dtype(df[lat_col_name]):
                               if lat_col_name not in analysis["numeric"]: analysis["numeric"].append(lat_col_name)
                               if lat_col_name not in analysis["geospatial"]: analysis["geospatial"].append(lat_col_name)
                           if lon_col_name in df.columns and pd.api.types.is_numeric_dtype(df[lon_col_name]):
                               if lon_col_name not in analysis["numeric"]: analysis["numeric"].append(lon_col_name)
                               if lon_col_name not in analysis["geospatial"]: analysis["geospatial"].append(lon_col_name)
                      except Exception: pass # Silenciar errores de extracción
            elif any(key in col_lower for key in potential_geo_keywords): # Columnas ya nombradas como lat/lon
                  if ('latitud' in col_lower or 'latitude' in col_lower) and pd.api.types.is_numeric_dtype(dtype):
                      if col not in analysis["geospatial"]: analysis["geospatial"].append(col); is_geo_classified = True
                  if ('longitud' in col_lower or 'longitude' in col_lower) and pd.api.types.is_numeric_dtype(dtype):
                       if col not in analysis["geospatial"]: analysis["geospatial"].append(col); is_geo_classified = True
            
            if is_geo_classified: continue

            if pd.api.types.is_numeric_dtype(dtype): analysis["numeric"].append(col)
            elif pd.api.types.is_datetime64_any_dtype(dtype):
                analysis["temporal"].append(col)
                try: analysis["temporal_range"][col] = (df_copy[col].min(), df_copy[col].max())
                except TypeError: analysis["temporal_range"][col] = (None, None)
            elif isinstance(dtype, pd.CategoricalDtype) or pd.api.types.is_object_dtype(dtype) or pd.api.types.is_string_dtype(dtype):
                analysis["categorical"].append(col)
                n_unique = df_copy[col].nunique()
                if n_unique < 100 and n_unique > 0:
                    try: analysis["value_counts"][col] = df_copy[col].value_counts().to_dict()
                    except TypeError: analysis["value_counts"][col] = {"error": "mixed types"}
            else: analysis["other"].append(col)
        try:
            analysis["stats"] = df_copy.describe(include='all').to_dict() # Quitado datetime_is_numeric
        except Exception as e:
            st.warning(f"No se pudieron calcular estadísticas descriptivas: {e}")
            analysis["stats"] = {}
        analysis["stats"] = { k: {k2: (v2 if pd.notna(v2) else None) for k2, v2 in v.items()} for k, v in analysis["stats"].items() }
        analysis["temporal_range"] = { k: tuple((t.isoformat() if pd.notna(t) else None) for t in v) for k, v in analysis["temporal_range"].items() }
        return analysis


class LLMVisualizerPlanner:
    def suggest_visualizations(self, df_sample: pd.DataFrame, query: str, analysis: Dict) -> List[Dict]:
        if not gemini_model:
            st.error("Modelo Gemini no disponible para sugerir visualizaciones.")
            return []
        try: df_head_str = df_sample.head(3).to_markdown(index=False) if hasattr(df_sample.head(3), 'to_markdown') else df_sample.head(3).to_string()
        except Exception: df_head_str = df_sample.head(3).to_string()
        stats_summary = ""
        value_counts_summary = "Valores Comunes en Categóricas (Top 3 más frecuentes):\n"
        limited_value_counts = 0
        for col, counts in analysis.get("value_counts", {}).items():
             if limited_value_counts >= 5: break
             if isinstance(counts, dict) and "error" not in counts:
                 top_items = list(counts.items())[:3]
                 value_counts_summary += f"- {col}: {', '.join([f'{k} ({v})' for k, v in top_items])}\n"; limited_value_counts +=1
        if not limited_value_counts: value_counts_summary = ""
        temporal_range_summary = ""
        has_temporal = False
        for col, (min_t, max_t) in analysis.get("temporal_range", {}).items():
             if min_t and max_t: temporal_range_summary += f"- {col}: Desde {min_t} hasta {max_t}\n"; has_temporal = True
        if not has_temporal: temporal_range_summary = ""
        def shorten_list_for_prompt(col_list, max_items=15):
            return col_list[:max_items] + ["... (y otras)"] if len(col_list) > max_items else col_list

        # Nuevo aviso para el LLM sobre lat/lon
        lat_lon_instructions = ""
        if 'latitude' in analysis.get('numeric', []) and 'longitude' in analysis.get('numeric', []):
            lat_lon_instructions = "NOTA IMPORTANTE PARA MAPAS: Se han detectado o creado columnas 'latitude' y 'longitude' numéricas. DEBES usarlas para los parámetros `lat` y `lon` de cualquier mapa (ej., en `campos_involucrados` o como `plotly_params: {{'lat': 'latitude', 'lon': 'longitude'}}`). No uses la columna original 'geo_point_2d' para los ejes del mapa si 'latitude' y 'longitude' están disponibles."
        elif 'latitude' in analysis.get('geospatial', []) and 'longitude' in analysis.get('geospatial', []): # Por si están en geo y no en num
             lat_lon_instructions = "NOTA IMPORTANTE PARA MAPAS: Columnas 'latitude' y 'longitude' están presentes en geoespaciales. DEBES usarlas para los parámetros `lat` y `lon` de cualquier mapa. No uses la columna original 'geo_point_2d' para los ejes del mapa si 'latitude' y 'longitude' están disponibles."


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
        - Geoespaciales: {shorten_list_for_prompt(analysis['geospatial'])}
        - Otras: {shorten_list_for_prompt(analysis['other'])}
        {lat_lon_instructions}

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
            - "campos_involucrados": (Lista de strings) Nombres EXACTOS de columnas. Para mapas, si 'latitude' y 'longitude' están disponibles, úsalas aquí o en plotly_params.
            - "titulo_de_la_visualizacion": (String) Título descriptivo.
            - "descripcion_utilidad": (String) Qué muestra y cómo ayuda.
            - "plotly_params": (Opcional, Dict) Parámetros para Plotly Express. Los valores DEBEN ser nombres de columnas existentes o valores numéricos/booleanos. Para mapas, "z" puede ser una columna numérica para mapas de calor/densidad. Si 'latitude' y 'longitude' no están en campos_involucrados para un mapa, puedes ponerlas aquí: {{"lat": "latitude", "lon": "longitude"}}.
        6.  Formato de Salida: **SOLAMENTE la lista JSON válida** ([{{...}}, {{...}}]). Sin texto fuera del JSON.
        Genera las sugerencias JSON:
        """
        raw_content = None; cleaned_json = None
        try:
            generation_config = genai.types.GenerationConfig(response_mime_type="application/json", temperature=0.1, max_output_tokens=2048)
            response = gemini_model.generate_content(prompt, generation_config=generation_config)
            if not response.candidates: st.warning(f"Planner bloqueado. Feedback: {response.prompt_feedback}"); return []
            if hasattr(response, 'text') and isinstance(response.text, str) and response.text.strip(): raw_content = response.text
            elif response.candidates[0].content.parts: raw_content = "".join(part.text for part in response.candidates[0].content.parts if hasattr(part, 'text'))
            else: st.warning("Planner: Respuesta sin texto."); return []
            if not raw_content or not raw_content.strip(): st.warning("Planner: Respuesta vacía."); return []
            match_block = re.search(r'```json\s*([\s\S]*?)\s*```', raw_content, re.IGNORECASE)
            if match_block: cleaned_json = match_block.group(1).strip()
            else:
                raw_content_strip = raw_content.strip()
                if raw_content_strip.startswith(('[', '{')): cleaned_json = raw_content_strip
                else: st.error("Planner: Respuesta no JSON."); st.text_area("Respuesta:", raw_content, height=200); return []
            if not cleaned_json: st.error("Planner: No se extrajo JSON."); return []
            visualizations = json.loads(cleaned_json)
            if isinstance(visualizations, dict):
                 keys = list(visualizations.keys())
                 if len(keys) == 1 and isinstance(visualizations[keys[0]], list): visualizations = visualizations[keys[0]]
                 else: visualizations = [visualizations]
            if not isinstance(visualizations, list): st.warning(f"Planner: Respuesta no lista. Tipo: {type(visualizations)}"); return []
            return visualizations
        except json.JSONDecodeError as e: st.error(f"Planner: Error JSON: {e}"); st.text_area("JSON:", cleaned_json or raw_content, height=150); return []
        except Exception as e: st.error(f"Planner: Error: {e}"); st.text_area("Respuesta:", raw_content, height=150); return []


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
    AGGREGATION_THRESHOLD = 2000

    @staticmethod
    def normalize_chart_type(chart_type_raw: str) -> str:
        import unicodedata
        if not isinstance(chart_type_raw, str): return ""
        chart_type_lower = chart_type_raw.lower().strip()
        nfkd_form = unicodedata.normalize('NFKD', chart_type_lower)
        normalized = "".join([c for c in nfkd_form if not unicodedata.combining(c)])
        return normalized.replace('grafico', 'grafico')

    @staticmethod
    def plot(df: pd.DataFrame, config: Dict) -> Optional[go.Figure]:
        chart_type_raw = config.get("tipo_de_visualizacion", "")
        chart_type = DatasetVisualizer.normalize_chart_type(chart_type_raw)
        campos_orig = config.get("campos_involucrados", [])
        campos = [c for c in campos_orig if c in df.columns] # Usar solo campos existentes
        if not campos and campos_orig: st.warning(f"Ninguno de los campos sugeridos ({campos_orig}) para '{chart_type_raw}' existe. Cols: {list(df.columns)}"); return None
        if not campos and not campos_orig: st.warning(f"No se especificaron campos para '{chart_type_raw}'."); return None
        title = config.get("titulo_de_la_visualizacion", chart_type_raw)
        plotly_params_orig = config.get("plotly_params", {}) or {}
        plot_func = DatasetVisualizer.PLOT_FUNCTIONS.get(chart_type) or DatasetVisualizer.PLOT_FUNCTIONS.get(chart_type_raw.lower().strip())
        if not plot_func: st.warning(f"Tipo visualización no soportado: '{chart_type_raw}'"); return None
        df_processed = df.copy(); plot_args = {"data_frame": df_processed, "title": title}; axis_mapping = {}
        lat_col_name, lon_col_name = 'latitude', 'longitude' # Nombres estándar

        try:
            if chart_type == "histograma":
                 if campos and campos[0] in df_processed.columns: axis_mapping['x'] = campos[0]
                 else: raise ValueError(f"Campo X '{campos[0] if campos else 'N/A'}' no encontrado")
            elif chart_type in ["grafico circular", "circular", "tarta"]:
                 if campos and campos[0] in df_processed.columns: axis_mapping['names'] = campos[0]
                 else: raise ValueError(f"Campo 'names' '{campos[0] if campos else 'N/A'}' no encontrado")
                 if len(campos) > 1 and campos[1] in df_processed.columns: axis_mapping['values'] = campos[1]
            elif chart_type in ["grafico de barras", "barras", "grafico de lineas", "linea", "grafico de dispersion", "dispersion", "box plot", "boxplot", "diagrama de caja"]:
                if not campos: raise ValueError(f"No campos para {chart_type}")
                if campos[0] not in df_processed.columns: raise ValueError(f"Campo X '{campos[0]}' no encontrado")
                axis_mapping['x'] = campos[0]
                if len(campos) >= 2:
                    if campos[1] not in df_processed.columns: raise ValueError(f"Campo Y '{campos[1]}' no encontrado")
                    axis_mapping['y'] = campos[1]
                elif chart_type in ['box plot', 'boxplot', 'diagrama de caja']: axis_mapping['y'] = axis_mapping.pop('x')
                else: raise ValueError(f"Se necesitan 2 campos (X, Y) para {chart_type}, solo X: {campos[0]}")
            elif chart_type == "treemap":
                 path_cols = [c for c in campos if c in df_processed.columns]
                 if not path_cols: raise ValueError(f"Ningún campo para 'path' ({campos}) encontrado")
                 axis_mapping['path'] = path_cols
            elif chart_type in ["mapa de puntos", "mapa de dispersion", "mapa de calor", "mapa de densidad"]:
                # Intentar usar lat/lon de plotly_params si el LLM los puso ahí
                param_lat = plotly_params_orig.get('lat')
                param_lon = plotly_params_orig.get('lon')

                if param_lat and param_lon and param_lat in df_processed.columns and param_lon in df_processed.columns and \
                   pd.api.types.is_numeric_dtype(df_processed[param_lat]) and pd.api.types.is_numeric_dtype(df_processed[param_lon]):
                    axis_mapping['lat'], axis_mapping['lon'] = param_lat, param_lon
                # Si no, buscar las columnas estándar 'latitude'/'longitude' (que DatasetAnalyzer debería haber creado)
                elif lat_col_name in df_processed.columns and lon_col_name in df_processed.columns and \
                     pd.api.types.is_numeric_dtype(df_processed[lat_col_name]) and pd.api.types.is_numeric_dtype(df_processed[lon_col_name]):
                    axis_mapping['lat'], axis_mapping['lon'] = lat_col_name, lon_col_name
                else: # Fallback a buscar cualquier columna que parezca lat/lon
                    found_lat, found_lon = None, None
                    for c in df_processed.columns:
                        cl = c.lower()
                        if not found_lat and ('lat' in cl or 'ycoord' in cl) and pd.api.types.is_numeric_dtype(df_processed[c]): found_lat = c
                        if not found_lon and ('lon' in cl or 'xcoord' in cl) and pd.api.types.is_numeric_dtype(df_processed[c]): found_lon = c
                    if found_lat and found_lon: axis_mapping['lat'], axis_mapping['lon'] = found_lat, found_lon
                    else: raise ValueError(f"No se encontraron lat/lon numéricas adecuadas. Columnas: {list(df_processed.columns)}")
                
                plot_args.update({'zoom': 10, 'mapbox_style': "open-street-map"})
                
                if chart_type in ["mapa de calor", "mapa de densidad"]:
                    z_col = None
                    if 'z' in plotly_params_orig and plotly_params_orig['z'] in df_processed.columns and pd.api.types.is_numeric_dtype(df_processed[plotly_params_orig['z']]):
                        z_col = plotly_params_orig['z']
                    else: # Buscar 'z' en campos_involucrados que no sean lat/lon
                        potential_z = [f for f in campos if f not in [axis_mapping.get('lat'), axis_mapping.get('lon')] and f in df_processed.columns and pd.api.types.is_numeric_dtype(df_processed[f])]
                        if potential_z: z_col = potential_z[0]
                        elif 'numplazas' in df_processed.columns and pd.api.types.is_numeric_dtype(df_processed['numplazas']): z_col = 'numplazas' # Fallback común
                    if z_col: axis_mapping['z'] = z_col; plot_args['radius'] = 10
            
            x_col_agg, y_col_agg = axis_mapping.get('x'), axis_mapping.get('y')
            if chart_type in ["grafico de barras", "barras", "grafico de lineas", "linea"] and x_col_agg and y_col_agg:
                if x_col_agg in df_processed.columns and y_col_agg in df_processed.columns:
                    is_x_cat_or_time = (pd.api.types.is_string_dtype(df_processed[x_col_agg]) or pd.api.types.is_object_dtype(df_processed[x_col_agg]) or pd.api.types.is_datetime64_any_dtype(df_processed[x_col_agg]))
                    is_y_numeric = pd.api.types.is_numeric_dtype(df_processed[y_col_agg])
                    if is_x_cat_or_time and is_y_numeric and (len(df_processed) > DatasetVisualizer.AGGREGATION_THRESHOLD or df_processed[x_col_agg].nunique() > 75):
                        agg_func = 'sum' if chart_type in ["grafico de barras", "barras"] else 'mean'
                        try: plot_args['data_frame'] = df_processed.groupby(x_col_agg, dropna=False)[y_col_agg].agg(agg_func).reset_index()
                        except Exception as agg_e: st.warning(f"Falló auto-agregación para '{title}': {agg_e}")
            
            current_df_for_plot = plot_args['data_frame']
            valid_plotly_params = {}; param_aliases = {'nbinsx': 'nbins', 'nbinsy': 'nbins'}
            for key, value in plotly_params_orig.items():
                param_key = param_aliases.get(key.lower(), key)
                is_valid_col = isinstance(value, str) and value in current_df_for_plot.columns
                is_valid_col_list = isinstance(value, list) and all(isinstance(c, str) and c in current_df_for_plot.columns for c in value)
                is_non_col_param = not (isinstance(value, str) or isinstance(value,list))
                if (param_key == 'path' and is_valid_col_list) or (param_key != 'path' and is_valid_col) or is_non_col_param:
                    valid_plotly_params[param_key] = value
            plot_args.update(axis_mapping); plot_args.update(valid_plotly_params)
            final_df_check = plot_args['data_frame']; cols_to_verify = []
            for v_ax in axis_mapping.values(): cols_to_verify.extend(v_ax if isinstance(v_ax, list) else [v_ax])
            for v_param in valid_plotly_params.values():
                 if isinstance(v_param, str) and v_param in final_df_check.columns: cols_to_verify.append(v_param)
                 elif isinstance(v_param, list) and all(isinstance(i, str) and i in final_df_check.columns for i in v_param): cols_to_verify.extend(v_param)
            for col_final in set(cols_to_verify):
                if col_final not in final_df_check.columns: raise ValueError(f"Columna '{col_final}' no existe. Disp: {list(final_df_check.columns)}")
            return plot_func(**plot_args)
        except ValueError as ve: st.warning(f"Error config gráfico '{title}': {ve}"); return None
        except Exception as e: st.error(f"Error inesperado gráfico '{title}': {e}"); return None


class LLMInterpreter:
     def generate_insights(self, query: str, analysis: Dict, viz_configs_generated: List[Dict], df_sample: pd.DataFrame, dataset_title: str) -> str:
        if not gemini_model: st.error("Modelo Gemini no disponible para insights."); return "Error: Modelo no disponible."
        if not viz_configs_generated: return "No se generaron visualizaciones válidas, no hay insights."
        viz_summary = "Visualizaciones generadas (o intentadas) y su propósito:\n"
        for i, config in enumerate(viz_configs_generated):
               viz_summary += f"{i+1}. **{config.get('titulo_de_la_visualizacion', 'Sin título')}** ({config.get('tipo_de_visualizacion', 'N/A')}):\n"
               viz_summary += f"   - Campos: {config.get('campos_involucrados', [])}. Utilidad IA: {config.get('descripcion_utilidad', 'N/A')}\n"
        df_cols_list = list(df_sample.columns[:10]); df_cols_list.append("..." if len(df_sample.columns) > 10 else "")
        data_summary = f"Dataset '{dataset_title}' (muestra) cols: {df_cols_list}."
        def shorten_list_for_prompt(col_list, max_items=5): return col_list[:max_items] + ["..."] if len(col_list) > max_items else col_list
        analysis_summary = (f"Análisis: Num: {shorten_list_for_prompt(analysis['numeric'])}, "
                            f"Cat: {shorten_list_for_prompt(analysis['categorical'])}, "
                            f"Temp: {shorten_list_for_prompt(analysis['temporal'])}.")
        prompt = f"""
        Actúa como analista de datos conciso.
        Contexto:
        - Consulta: "{query}"
        - Dataset: '{dataset_title}'. {data_summary} {analysis_summary}
        - Visualizaciones generadas: {viz_summary}
        Tarea: Basado EXCLUSIVAMENTE en la consulta y descripción de visualizaciones, redacta un resumen (1-2 párrafos, MÁXIMO 100 palabras) con insights sobre los datos en relación a la consulta.
        Instrucciones: NO inventes información. Céntrate en responder la consulta. Sintetiza. Si las visualizaciones no son suficientes, menciónalo. Resultado: SOLO el resumen en lenguaje natural.
        Genera el resumen:
        """
        raw_content = None
        try:
            generation_config = genai.types.GenerationConfig(temperature=0.4, max_output_tokens=350)
            response = gemini_model.generate_content(prompt, generation_config=generation_config)
            if not response.candidates: st.warning(f"Insights bloqueado. Feedback: {response.prompt_feedback}"); return "Restricciones impidieron generar insights."
            if hasattr(response, 'text') and isinstance(response.text, str) and response.text.strip(): raw_content = response.text
            elif response.candidates[0].content.parts: raw_content = "".join(part.text for part in response.candidates[0].content.parts if hasattr(part, 'text'))
            else: st.warning("Insights: Respuesta sin texto."); return "No insights (respuesta sin texto)."
            if not raw_content or not raw_content.strip(): st.warning("Insights: Respuesta vacía."); return "No insights."
            return re.sub(r"^(Aquí tienes|Claro,|Basado en|Finalmente|En resumen|De acuerdo|Según los datos)[,.:]?\s*", "", raw_content, flags=re.IGNORECASE).strip()
        except Exception as e: st.error(f"Insights: Error: {e}"); return "Error generando insights."


def validate_dataset_relevance(query: str, dataset_title: str, dataset_description: str) -> bool:
    if not gemini_model: st.warning("Modelo Gemini no disponible para validación. Asumiendo relevancia."); return True
    prompt = f"""
    Consulta: "{query}"
    Dataset: Título: "{dataset_title}", Desc: "{dataset_description[:500]}"
    Pregunta: ¿Es este dataset ALTAMENTE relevante para responder la consulta? Responde SÓLO 'Sí' o 'No'.
    """
    raw_response = None
    try:
        generation_config = genai.types.GenerationConfig(temperature=0.0, max_output_tokens=10)
        response = gemini_model.generate_content(prompt, generation_config=generation_config)
        if not response.candidates: st.warning(f"Validación bloqueada. Feedback: {response.prompt_feedback}. Asumiendo relevancia."); return True
        if hasattr(response, 'text') and isinstance(response.text, str) and response.text.strip(): raw_response = response.text
        elif response.candidates[0].content.parts: raw_response = "".join(part.text for part in response.candidates[0].content.parts if hasattr(part, 'text'))
        else: st.warning("Validación: Respuesta sin texto. Asumiendo relevancia."); return True
        if not raw_response or not raw_response.strip(): st.warning("Validación: Respuesta vacía. Asumiendo relevancia."); return True
        decision = raw_response.strip().lower().replace('.', '').replace(',', '')
        return decision == 'sí' or decision == 'si'
    except Exception as e: st.error(f"Validación LLM: Error: {e}. Asumiendo relevancia."); return True

@st.cache_resource
def get_faiss_index_instance():
    # print("Attempting to load FAISSIndex instance...") # Debug
    instance = FAISSIndex()
    if instance.is_ready():
        st.sidebar.success(f"Índice FAISS listo ({instance.index.ntotal} vectores).")
    else:
        st.sidebar.warning("Archivos de índice no encontrados o vacíos. Constrúyelo desde la barra lateral.")
    return instance


def run_full_dashboard_pipeline_st(user_query: str, faiss_index_instance: FAISSIndex, sentence_model_instance):
    if not gemini_model:
        st.error("Modelo Gemini (Google AI) no está configurado o disponible. El pipeline no puede continuar.")
        return

    st.subheader(f"Procesando consulta: \"{user_query}\"")
    with st.spinner("Buscando y validando datasets..."):
        if not faiss_index_instance.is_ready():
            st.error("El índice FAISS no está listo. Constrúyelo primero.")
            return
        api_agent = APIQueryAgent(faiss_index_instance, sentence_model_instance) # Pasa el modelo cacheado
        search_results = api_agent.search_dataset(user_query, top_k=3)
        selected_dataset_info = None
        if search_results:
            with st.expander("Candidatos iniciales y validación", expanded=False):
                for result_idx, result in enumerate(search_results):
                    dataset_title = result["metadata"]["title"]
                    similarity_score = result['similarity']
                    st.markdown(f"--- Candidato {result_idx + 1}/{len(search_results)}: `{dataset_title}` (Sim: {similarity_score:.4f}) ---")
                    if similarity_score < api_agent.SIMILARITY_THRESHOLD:
                        st.warning(f"  Similitud baja. Descartando.")
                        continue
                    if not validate_dataset_relevance(user_query, dataset_title, result["metadata"].get("description", "")):
                        st.warning(f"  No relevante por LLM. Descartando.")
                        continue
                    st.success(f"Dataset '{dataset_title}' validado.")
                    selected_dataset_info = result; break
    if not selected_dataset_info:
        st.error("No se encontró un dataset relevante para tu consulta."); return
    dataset_id = selected_dataset_info["metadata"]["id"]; dataset_title = selected_dataset_info["metadata"]["title"]
    st.success(f"🎯 Dataset Seleccionado: **{dataset_title}** (ID: {dataset_id})")
    df = None
    with st.spinner(f"Descargando y cargando '{dataset_title}'..."):
        dataset_bytes = api_agent.export_dataset(dataset_id)
        if not dataset_bytes: st.error(f"No se pudo descargar '{dataset_title}'."); return
        df = DatasetLoader.load_dataset_from_bytes(dataset_bytes, dataset_title)
        if df is None or df.empty: st.error(f"Dataset '{dataset_title}' vacío o no cargado."); return
        with st.expander(f"Primeras filas de '{dataset_title}'", expanded=False): st.dataframe(df.head())
    analysis = None
    with st.spinner("Analizando el dataset..."):
        analyzer = DatasetAnalyzer()
        analysis = analyzer.analyze(df) # df puede ser modificado aquí (lat/lon)
        with st.expander("Análisis Detallado del Dataset", expanded=False): st.json(analysis, expanded=False)
    viz_configs_suggested = None
    with st.spinner("Generando sugerencias de visualización con IA..."):
        planner = LLMVisualizerPlanner()
        df_sample_for_planner = df.head(20) if len(df) > 20 else df.copy() # Usar copia para no modificar df
        viz_configs_suggested = planner.suggest_visualizations(df_sample_for_planner, user_query, analysis)
    figures = []; valid_viz_configs_generated = []
    if viz_configs_suggested:
        st.subheader("📊 Visualizaciones Sugeridas y Generadas")
        visualizer = DatasetVisualizer()
        for idx, config in enumerate(viz_configs_suggested):
            with st.container():
                title_viz = config.get('titulo_de_la_visualizacion', f'Visualización {idx+1}')
                st.markdown(f"**{idx+1}. {title_viz}**")
                st.caption(f"Tipo: {config.get('tipo_de_visualizacion', 'N/A')}. Campos: {config.get('campos_involucrados', 'N/A')}. IA Desc: {config.get('descripcion_utilidad', 'N/A')}")
                with st.spinner(f"Generando: {title_viz}..."): fig = visualizer.plot(df, config) # df modificado por Analyzer
                if fig:
                    try: st.plotly_chart(fig, use_container_width=True); figures.append(fig); valid_viz_configs_generated.append(config)
                    except Exception as e_plot: st.error(f"Error mostrando '{title_viz}': {e_plot}")
                else: st.warning(f"No se pudo generar: {title_viz}")
    else: st.info("IA no sugirió visualizaciones o hubo un error.")
    if not figures: st.warning("No se generaron visualizaciones válidas.")
    with st.spinner("Generando insights con IA..."):
        st.subheader("💡 Insights del Analista Virtual")
        interpreter = LLMInterpreter()
        insights_text = interpreter.generate_insights(user_query, analysis, valid_viz_configs_generated, df.head(), dataset_title)
        if "Error" in insights_text or not insights_text.strip(): st.warning(insights_text or "No se generaron insights.")
        else: st.markdown(insights_text)
    st.success("--- ✅ Pipeline Completado ---"); st.balloons()

# --- Interfaz de Streamlit ---
st.set_page_config(layout="wide", page_title="Analista Datos Valencia")
faiss_index_global = get_faiss_index_instance() # Carga o usa la instancia cacheada
sentence_model_global = get_sentence_transformer_model(EMBEDDING_MODEL) # Carga o usa el modelo cacheado

st.title("BI Assistant - Open Data València")
st.markdown('Bienvenido al asistente para explorar [Datos Abiertos del Ayuntamiento de Valencia](https://valencia.opendatasoft.com/pages/home/?flg=es-es).')
st.sidebar.header("Acciones del Índice")
if st.sidebar.button("Construir/Actualizar Índice FAISS", help="Descarga metadatos y crea/actualiza el índice. Puede tardar varios minutos."):
    build_and_save_index()
user_query = st.text_input("¿Qué datos te gustaría analizar o qué pregunta tienes sobre Valencia?",
                           placeholder="Ej: ¿Dónde hay estaciones Valenbisi? o Tráfico en Av. Blasco Ibáñez")
if st.button("Analizar Consulta", type="primary"):
    if user_query:
        if not API_KEY_GEMINI or not gemini_model:
            st.error("API Key de Gemini (Google AI) no configurada o modelo no inicializado. Verifica .env y reinicia."); st.stop()
        if not faiss_index_global.is_ready(): # Chequeo adicional por si acaso
            st.error("Índice FAISS no está listo. Por favor, constrúyelo primero desde la barra lateral."); st.stop()
        run_full_dashboard_pipeline_st(user_query, faiss_index_global, sentence_model_global)
    else: st.warning("Por favor, introduce una consulta.")
st.markdown("---"); st.caption("Desarrollado como demostración. Los resultados pueden variar.")