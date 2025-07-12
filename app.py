# -*- coding: utf-8 -*-
import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import re
import json
import io
from typing import Dict, Any, List, Optional
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
import unicodedata
import logging
from pathlib import Path 

# --- Configuración del Logging ---
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuración inicial ---
load_dotenv()

SCRIPT_DIR = Path(__file__).resolve().parent
INDEX_FILE = str(SCRIPT_DIR / "faiss_opendata_valencia.idx")
METADATA_FILE = str(SCRIPT_DIR / "faiss_metadata.json")

BASE_URL = "https://valencia.opendatasoft.com/api/explore/v2.1/"
CATALOG_LIST_URL = "https://valencia.opendatasoft.com/api/v2/catalog/datasets"
EMBEDDING_MODEL = 'paraphrase-MiniLM-L6-v2'
GOOGLE_LLM_MODEL = "gemini-1.5-flash-latest"
LLAMA3_70B_MODEL_NAME_GROQ = "llama3-70b-8192"
GROQ_API_KEY = os.getenv("API_KEY_GROQ")
API_KEY_GEMINI = os.getenv("API_KEY_GEMINI")

groq_client = None
if GROQ_API_KEY:
    try:
        groq_client = groq.Groq(api_key=GROQ_API_KEY)
    except Exception as e:
        logging.error(f"Error inicializando Groq: {e}")

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
        gemini_model = genai.GenerativeModel(model_name=GOOGLE_LLM_MODEL, safety_settings=safety_settings)
    except Exception as e:
        logging.error(f"Error configurando Google AI: {e}.")


# --- Funciones de ayuda y clases ---
def make_llm_call(prompt_text: str, is_json_output: bool = False):
    active_llm = st.session_state.get("current_llm_provider", "gemini")
    if active_llm == "gemini":
        if not gemini_model: return "Error: Modelo Gemini no configurado."
        try:
            generation_config_params = {"temperature": 0.1, "max_output_tokens": 2048}
            if is_json_output:
                generation_config_params["response_mime_type"] = "application/json"
            else:
                generation_config_params["temperature"] = 0.4
                generation_config_params["max_output_tokens"] = 450
            generation_config_obj = genai.types.GenerationConfig(**generation_config_params)
            response = gemini_model.generate_content(prompt_text, generation_config=generation_config_obj)
            if not response.candidates: return f"Error Gemini: No candidates. Feedback: {response.prompt_feedback}"
            content_parts_text = "".join(p.text for p in response.candidates[0].content.parts if hasattr(p, 'text')).strip()
            final_text = response.text if hasattr(response, 'text') and response.text and response.text.strip() else content_parts_text
            if not final_text: return "Error Gemini: Respuesta vacía."
            return final_text
        except Exception as e: return f"Error Gemini: {e}"
    elif active_llm == "llama3":
        if not groq_client: return "Error: Cliente Groq para Llama3 no configurado."
        try:
            messages = [{"role": "user", "content": prompt_text}]
            response_format_arg = {"type": "json_object"} if is_json_output else None
            chat_completion = groq_client.chat.completions.create(
                messages=messages, model=LLAMA3_70B_MODEL_NAME_GROQ,
                temperature=0.1 if is_json_output else 0.4,
                max_tokens=2048 if is_json_output else 450,
                response_format=response_format_arg)
            response_content = chat_completion.choices[0].message.content.strip()
            if is_json_output and response_content.startswith("```json"):
                match = re.search(r'```json\s*([\s\S]*?)\s*```', response_content, re.IGNORECASE)
                if match: response_content = match.group(1).strip()
            return response_content
        except Exception as e: return f"Error Llama3 (Groq): {e}"
    else: return f"Error: Proveedor de LLM '{active_llm}' no reconocido."

def sanitize_filename(filename: str) -> str: return re.sub(r'[<>:"/\\|?*]', '_', filename)

class FAISSIndex:
    def __init__(self, index_path: str = INDEX_FILE, metadata_path: str = METADATA_FILE):
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.index = None; self.metadata = []; self.load_index()

    def load_index(self):
        if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
            try:
                self.index = faiss.read_index(self.index_path)
                with open(self.metadata_path, 'r', encoding='utf-8') as f:
                    self.metadata = json.load(f)
                logging.info(f"Índice FAISS y metadatos JSON cargados correctamente. {self.index.ntotal} vectores.")
            except Exception as e:
                logging.error(f"Error al cargar índice o metadatos: {e}")
                self.index = None
                self.metadata = []
        else:
            logging.warning(f"No se encontraron los ficheros del índice en {self.index_path} o {self.metadata_path}")
            self.index = None
    
    def is_ready(self) -> bool: return self.index is not None and self.index.ntotal > 0
    
    def search(self, query_embedding: np.ndarray, top_k: int = 1) -> List[Dict[str, Any]]:
        if not self.is_ready(): return []
        norm = np.linalg.norm(query_embedding)
        if norm == 0: return []
        query_embedding_norm = (query_embedding / norm).astype(np.float32).reshape(1, -1)
        distances, indices = self.index.search(query_embedding_norm, top_k)
        
        results = []
        if indices.size > 0:
            for i, idx_val in enumerate(indices[0]):
                if idx_val != -1 and idx_val < len(self.metadata):
                     similarity_score = 1 - (distances[0][i]**2) / 2
                     results.append({"metadata": self.metadata[idx_val], "similarity": float(similarity_score)})
        return results

class APIQueryAgent:
    SIMILARITY_THRESHOLD = 0.45
    def __init__(self, faiss_index: FAISSIndex, sentence_model): self.model = sentence_model; self.faiss_index = faiss_index
    def get_embedding(self, text: str) -> np.ndarray: return self.model.encode(text, normalize_embeddings=True)
    def search_dataset(self, query: str, top_k: int = 3) -> Optional[List[Dict[str, Any]]]:
        if not self.faiss_index.is_ready(): return None
        query_embedding = self.get_embedding(query)
        try: return self.faiss_index.search(query_embedding, top_k=top_k)
        except Exception as e: st.error(f"APIQueryAgent: Error FAISS search: {e}"); return None
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=6), retry=retry_if_exception_type(requests.exceptions.RequestException))
    def _fetch_api_data(self, endpoint: str, params: Dict = None) -> Optional[Dict]:
        try: response = requests.get(endpoint, params=params, timeout=20); response.raise_for_status(); return response.json()
        except requests.exceptions.RequestException as e: print(f"API Fetch Error: {e}"); st.error(f"API Error: {e}"); raise
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10), retry=retry_if_exception_type(requests.exceptions.RequestException))
    def export_dataset(self, dataset_id: str) -> Optional[bytes]:
        endpoint = f"{BASE_URL}catalog/datasets/{dataset_id}/exports/csv"; params = {"delimiter": ";"}
        try:
            response = requests.get(endpoint, params=params, timeout=60); response.raise_for_status()
            if 'text/csv' in response.headers.get('Content-Type', ''): return response.content
            else:
                 params = {}; response = requests.get(endpoint, params=params, timeout=60); response.raise_for_status()
                 if 'text/csv' in response.headers.get('Content-Type', ''): return response.content
                 else: return None
        except requests.exceptions.RequestException as e: st.error(f"Error descargando {dataset_id}: {e}"); raise

@st.cache_resource
def get_sentence_transformer_model(model_name): return SentenceTransformer(model_name, device='cpu')

def build_and_save_index(target_model_name: str = EMBEDDING_MODEL, index_path_to_save: str = INDEX_FILE, metadata_path_to_save: str = METADATA_FILE):
    st.header(f"Construyendo Índice FAISS para: {target_model_name}")
    
    try:
        sentence_model_instance = get_sentence_transformer_model(target_model_name)
    except Exception as e:
        st.error(f"Error al cargar el modelo de embeddings: {e}")
        return

    all_metadata = []
    all_embeddings_list = []
    
    limit = 100
    start = 0
    total_datasets = None

    with st.spinner("Obteniendo catálogo de datasets de OpenData Valencia..."):
        try:
            initial_response = requests.get(CATALOG_LIST_URL, params={"limit": 1, "offset": 0}, timeout=20)
            initial_response.raise_for_status()
            total_datasets = initial_response.json().get('total_count', 0)
            if total_datasets == 0:
                st.warning("No se encontraron datasets en el catálogo.")
                return
            st.info(f"Se encontraron {total_datasets} datasets. Procediendo a generar embeddings...")
        except requests.exceptions.RequestException as e:
            st.error(f"Error crítico al conectar con la API de OpenData Valencia: {e}")
            return

    progress_bar = st.progress(0.0, "Iniciando proceso...")
    status_area = st.empty()
    
    while start < total_datasets:
        try:
            status_area.write(f"Obteniendo página de datasets... offset={start}, limit={limit}")
            params = {"limit": limit, "offset": start}
            response = requests.get(CATALOG_LIST_URL, params=params, timeout=30)
            
            if response.status_code != 200:
                status_area.warning(f"Respuesta de la API para offset={start}: Código de estado {response.status_code}. Saltando página.")
                start += limit
                continue
            
            data = response.json()
            datasets_page = data.get('datasets', [])
            
            if not datasets_page:
                status_area.warning(f"La página con offset={start} no devolvió datasets. Finalizando bucle.")
                break

            texts_for_page = []
            metadata_for_page = []
            
            for dataset_info in datasets_page:
                dataset = dataset_info.get('dataset', {})
                dataset_id = dataset.get('dataset_id', '')
                meta = dataset.get('metas', {}).get('default', {})
                title = meta.get('title', 'Sin título')
                description_html = meta.get('description', '')
                description = BeautifulSoup(description_html, "html.parser").get_text().strip() if description_html else ""

                texts_for_page.append(f"título: {title}; descripción: {description}")
                metadata_for_page.append({"id": dataset_id, "title": title, "description": description})

            if texts_for_page:
                page_embeddings = sentence_model_instance.encode(texts_for_page, normalize_embeddings=True, show_progress_bar=False)
                all_embeddings_list.extend(page_embeddings)
                all_metadata.extend(metadata_for_page)
            
            start += len(datasets_page)
            progress_bar.progress(min(start / total_datasets, 1.0), text=f"Procesados {start}/{total_datasets} datasets")

        except requests.exceptions.RequestException as e:
            st.error(f"Error de red en offset {start}. Deteniendo... Error: {e}")
            break
        except Exception as e:
            st.error(f"Ocurrió un error inesperado en offset {start}: {e}")
            break
            
    if not all_embeddings_list:
        st.error("No se pudieron generar embeddings. El proceso ha fallado. Revisa los mensajes de estado de la API.")
        return

    progress_bar.empty()
    status_area.empty()

    with st.spinner(f"Construyendo y guardando el índice FAISS en {index_path_to_save}..."):
        embeddings_np = np.array(all_embeddings_list).astype('float32')
        d = embeddings_np.shape[1]
        index = faiss.IndexFlatL2(d)
        index.add(embeddings_np)
        
        faiss.write_index(index, index_path_to_save)
        
        with open(metadata_path_to_save, 'w', encoding='utf-8') as f:
            json.dump(all_metadata, f, ensure_ascii=False, indent=2)

    st.success(f"¡Índice FAISS con {index.ntotal} vectores construido y guardado con éxito!")
    st.balloons()
    st.info("La página se recargará para usar el nuevo índice.")
    st.rerun()

class DatasetLoader:
    @staticmethod
    def load_dataset_from_bytes(dataset_bytes: bytes, dataset_title: str = "dataset") -> Optional[pd.DataFrame]:
        file_name_hint = sanitize_filename(dataset_title); df = None
        try: df = pd.read_csv(io.BytesIO(dataset_bytes), delimiter=';', encoding='utf-8', on_bad_lines='warn'); assert df.shape[1] > 1 or len(df) == 0
        except:
            try: df = pd.read_csv(io.BytesIO(dataset_bytes), delimiter=',', encoding='utf-8', on_bad_lines='warn'); assert df.shape[1] > 1 or len(df) == 0
            except:
                try: df = pd.read_csv(io.BytesIO(dataset_bytes), delimiter=';', encoding='latin1', on_bad_lines='warn'); assert df.shape[1] > 1 or len(df) == 0
                except:
                    try: df = pd.read_csv(io.BytesIO(dataset_bytes), delimiter=',', encoding='latin1', on_bad_lines='warn'); assert df.shape[1] > 1 or len(df) == 0
                    except Exception as e_final: st.error(f"Error parseando CSV '{file_name_hint}': {e_final}"); return None
        if df is None or df.empty: return None
        df.columns = [str(col).strip().replace(" ", "_").lower() for col in df.columns]; return df

class DatasetAnalyzer:
    @staticmethod
    def analyze(df: pd.DataFrame) -> Dict[str, Any]:
        analysis = {"numeric": [], "categorical": [], "temporal": [], "geospatial": [], "other": [], "stats": None, "value_counts": {}, "temporal_range": {}}
        keywords = {'temp': ['fecha', 'date', 'año', 'ano', 'year', 'time'], 'geo': ['geo', 'lat', 'lon', 'coord', 'wkt', 'point', 'shape']}
        df_copy = df.copy(); lat_col, lon_col = 'latitude', 'longitude'
    
        # Pre-procesamiento
        for col in df_copy.columns:
            if df_copy[col].dtype == 'object':
                try: df_copy[col] = pd.to_numeric(df_copy[col].str.replace(',', '.', regex=False).str.strip())
                except (ValueError, AttributeError): pass
            if any(k in col.lower() for k in keywords['temp']) or pd.api.types.is_datetime64_any_dtype(df_copy[col].dtype):
                try:
                    orig_nn = df_copy[col].notna().sum()
                    if orig_nn == 0: continue
                    conv_df = pd.to_datetime(df_copy[col], errors='coerce', dayfirst=True)
                    if (conv_df.notna().sum() / orig_nn) > 0.5: df_copy[col] = conv_df
                except Exception: pass
    
        # Análisis de columnas
        for col in df_copy.columns:
            dtype = df_copy[col].dtype; cl = col.lower(); is_geo = False
            
            # <<< LÓGICA GEO REFORZADA >>>
            if any(k in cl for k in ['geo_point_2d', 'geopoint', 'geo_shape']):
                analysis["geospatial"].append(col); is_geo = True
                try:
                    # Asegurarse de que es una columna de strings antes de usar .str
                    if pd.api.types.is_string_dtype(df_copy[col]):
                        coords = df_copy[col].str.split(',', expand=True)
                        if coords.shape[1] >= 2: # Comprobar que la división produjo al menos 2 columnas
                            df[lat_col] = pd.to_numeric(coords[0], errors='coerce')
                            df[lon_col] = pd.to_numeric(coords[1], errors='coerce')
                            # Añadir las nuevas columnas al análisis si no están ya
                            if lat_col not in analysis["numeric"]: analysis["numeric"].append(lat_col)
                            if lon_col not in analysis["numeric"]: analysis["numeric"].append(lon_col)
                            if lat_col not in analysis["geospatial"]: analysis["geospatial"].append(lat_col)
                            if lon_col not in analysis["geospatial"]: analysis["geospatial"].append(lon_col)
                except Exception as e:
                    logging.warning(f"No se pudieron extraer coordenadas de la columna '{col}': {e}")
                    pass
                
            elif any(k in cl for k in keywords['geo']):
                if ('latitud' in cl or 'latitude' in cl): analysis["geospatial"].append(col); is_geo=True
                if ('longitud' in cl or 'longitude' in cl): analysis["geospatial"].append(col); is_geo=True
            
            if is_geo: continue
    
            if pd.api.types.is_numeric_dtype(dtype): analysis["numeric"].append(col)
            elif pd.api.types.is_datetime64_any_dtype(dtype): analysis["temporal"].append(col); analysis["temporal_range"][col] = (df_copy[col].min(), df_copy[col].max())
            elif pd.api.types.is_object_dtype(dtype) or pd.api.types.is_string_dtype(dtype):
                analysis["categorical"].append(col)
                if 0 < df_copy[col].nunique() < 100:
                    try: analysis["value_counts"][col] = df_copy[col].value_counts().to_dict()
                    except: pass
            else: analysis["other"].append(col)
                
        try:
            analysis["stats"] = df_copy.describe(include='all').to_dict()
        except Exception as e:
            logging.error(f"Error al generar describe(): {e}")
            analysis["stats"] = {}
            
        return analysis

class LLMVisualizerPlanner:
    def suggest_visualizations(self, df_sample: pd.DataFrame, query: str, analysis: Dict) -> List[Dict]:
        try: df_head_str = df_sample.head(3).to_markdown(index=False)
        except: df_head_str = df_sample.head(3).to_string()
        value_counts_summary = "\nValores Comunes en Categóricas (Top 3):\n"
        for col, counts in list(analysis.get("value_counts", {}).items())[:5]:
            top_items = list(counts.items())[:3]
            value_counts_summary += f"- {col}: {', '.join([f'{k} ({v})' for k, v in top_items])}\n"
        
        prompt = f"""Actúa como un analista de BI experto. Tu objetivo es proponer las mejores visualizaciones para responder a la consulta de un usuario, usando un dataset.
Consulta del Usuario: "{query}"
Dataset Resumido (primeras filas de {df_sample.shape[0]}):
{df_head_str}
Análisis de Columnas (USA ESTOS NOMBRES EXACTOS):
- Numéricas: {analysis['numeric']}
- Categóricas: {analysis['categorical']}
- Temporales: {analysis['temporal']}
- Geoespaciales: {analysis['geospatial']}
{value_counts_summary if len(analysis.get("value_counts", {})) > 0 else ""}
Instrucciones:
1.  Prioriza la consulta del usuario.
2.  Usa **SOLAMENTE las columnas listadas**. Nombres EXACTOS.
3.  Sugiere entre 2 y 4 visualizaciones variadas.
4.  Formato de Salida: **SOLAMENTE la lista JSON válida** ([{{...}}, {{...}}]).
5.  Para cada visualización, proporciona:
    - "tipo_de_visualizacion": (String) Elige ESTRICTAMENTE de esta lista: ["histograma", "grafico de barras", "mapa de puntos", "grafico de lineas", "grafico circular", "diagrama de caja", "treemap", "mapa de calor"].
    - "campos_involucrados": (Lista de strings) Nombres EXACTOS de columnas.
    - "titulo_de_la_visualizacion": (String) Título descriptivo.
    - "descripcion_utilidad": (String) Qué muestra y cómo ayuda.
    - "plotly_params": (Opcional, Dict) Parámetros para Plotly Express.
Genera el JSON:"""
        raw_content = make_llm_call(prompt, is_json_output=True)
        if raw_content.startswith("Error"): st.error(f"Planner Error: {raw_content}"); return []
        try:
            match_block = re.search(r'```json\s*([\s\S]*?)\s*```', raw_content, re.IGNORECASE)
            cleaned_json = match_block.group(1).strip() if match_block else raw_content.strip()
            visualizations = json.loads(cleaned_json)
            if isinstance(visualizations, dict):
                 keys = list(visualizations.keys())
                 if len(keys) == 1 and isinstance(visualizations[keys[0]], list): visualizations = visualizations[keys[0]]
                 else: visualizations = [visualizations]
            return visualizations if isinstance(visualizations, list) else []
        except Exception as e: st.error(f"Planner JSON Error: {e}"); st.text_area("Respuesta:", raw_content, height=150); return []

class DatasetVisualizer:
    PLOT_FUNCTIONS = {
        "histograma": px.histogram, "barras": px.bar, "lineas": px.line,
        "dispersion": px.scatter, "caja": px.box, "puntos": px.scatter_map,
        "circular": px.pie, "treemap": px.treemap, "calor": px.density_mapbox
    }

    @staticmethod
    def _normalize_chart_type(chart_type: str) -> str:
        if not isinstance(chart_type, str): return ""
        nfkd_form = unicodedata.normalize('NFKD', chart_type.lower().strip())
        normalized = "".join([c for c in nfkd_form if not unicodedata.combining(c)])
        if "barras" in normalized or "bar" in normalized: return "barras"
        if "lineas" in normalized or "line" in normalized: return "lineas"
        if "mapa de puntos" in normalized or "scatter map" in normalized: return "puntos"
        if "mapa de calor" in normalized or "density map" in normalized: return "calor"
        if "dispersion" in normalized or "scatter" in normalized: return "dispersion"
        if "diagrama de caja" in normalized or "box" in normalized: return "caja"
        if "circular" in normalized or "tarta" in normalized or "pie" in normalized: return "circular"
        return normalized

    @staticmethod
    def plot(df: pd.DataFrame, config: Dict) -> Optional[go.Figure]:
        chart_type_norm = DatasetVisualizer._normalize_chart_type(config.get("tipo_de_visualizacion", ""))
        plot_func = DatasetVisualizer.PLOT_FUNCTIONS.get(chart_type_norm)
        if not plot_func:
            st.warning(f"Tipo visualización no soportado: '{config.get('tipo_de_visualizacion', '')}'")
            return None
        campos_orig = config.get("campos_involucrados", [])
        campos = [c for c in campos_orig if c in df.columns]
        if not campos and campos_orig: st.warning(f"Campos no existen ({campos_orig})."); return None
        if not campos: st.warning(f"No hay campos para '{chart_type_norm}'."); return None
        
        title = config.get("titulo_de_la_visualizacion", chart_type_norm)
        
        df_plot = df.copy()

        plot_args = {"data_frame": df_plot, "title": title, **(config.get("plotly_params", {}) or {})}
        plot_args.pop('type', None)

        try:
            if chart_type_norm == "histograma": plot_args['x'] = campos[0]
            elif chart_type_norm == "circular": plot_args.update({'names': campos[0], 'values': campos[1] if len(campos) > 1 else None})
            elif chart_type_norm in ["barras", "lineas", "dispersion", "caja"]: plot_args.update({'x': campos[0], 'y': campos[1] if len(campos) > 1 else None})
            elif chart_type_norm == "treemap": plot_args['path'] = [c for c in campos if c in df.columns]
            elif chart_type_norm in ["puntos", "calor"]:
                lat_col, lon_col = None, None
                p_lat, p_lon = plot_args.get('lat'), plot_args.get('lon')
                if p_lat in df.columns and p_lon in df.columns: lat_col, lon_col = p_lat, p_lon
                elif 'latitude' in df.columns and 'longitude' in df.columns: lat_col, lon_col = 'latitude', 'longitude'
                else:
                    for c in campos:
                        cl = c.lower()
                        if 'latit' in cl or cl == 'y': lat_col = c
                        if 'longit' in cl or cl == 'x': lon_col = c
                        if lat_col and lon_col: break
                if not (lat_col and lon_col): raise ValueError("No se pudieron encontrar columnas de latitud/longitud.")
                
                df_plot[lat_col] = pd.to_numeric(df_plot[lat_col], errors='coerce')
                df_plot[lon_col] = pd.to_numeric(df_plot[lon_col], errors='coerce')
                df_plot.dropna(subset=[lat_col, lon_col], inplace=True)
                if df_plot.empty:
                    st.warning("No hay datos geoespaciales válidos para mostrar después de la limpieza.")
                    return None
                
                plot_args.update({'lat': lat_col, 'lon': lon_col, 'zoom': plot_args.get('zoom', 10)})
                
                if chart_type_norm == "calor":
                    plot_args['mapbox_style'] = "open-street-map" 
                    z_cands = [f for f in campos if f not in [lat_col, lon_col] and pd.api.types.is_numeric_dtype(df[f])]
                    if z_cands:
                        plot_args['z'] = z_cands[0]
                        df_plot[z_cands[0]] = pd.to_numeric(df_plot[z_cands[0]], errors='coerce')
                        df_plot.dropna(subset=[z_cands[0]], inplace=True)

            return plot_func(**plot_args)
        except Exception as e:
            st.error(f"Error generando el gráfico '{title}': {e}")
            return None

class LLMInterpreter:
    def generate_insights(self, query: str, analysis: Dict, viz_configs_generated: List[Dict], df_sample: pd.DataFrame, dataset_title: str) -> str:
        if not viz_configs_generated: return "No se generaron visualizaciones válidas."
        viz_summary = "\n".join([f"- **{c.get('titulo_de_la_visualizacion', 'N/A')}** ({c.get('tipo_de_visualizacion', 'N/A')}): {c.get('descripcion_utilidad', 'N/A')}" for c in viz_configs_generated])
        prompt = f"""Actúa como analista de datos conciso.
Contexto:
- Consulta: "{query}"
- Dataset: '{dataset_title}' ({df_sample.shape[0]} filas en muestra).
- Visualizaciones Generadas: {viz_summary}
Tarea: Redacta un resumen breve (1-2 párrafos, máx 120 palabras) con los insights más relevantes. Céntrate en responder la consulta. No inventes información.
Genera el resumen:"""
        raw_content = make_llm_call(prompt, is_json_output=False)
        return raw_content if not raw_content.startswith("Error") else "No se generaron insights."

def validate_dataset_relevance(query: str, dataset_title: str, dataset_description: str) -> bool:
    prompt = f"""Evalúa la relevancia. Consulta: "{query}". Dataset: Título="{dataset_title}", Desc="{dataset_description[:300]}". ¿Es este dataset ALTAMENTE relevante para la consulta? Responde solo con 'Sí' o 'No'."""
    raw_response = make_llm_call(prompt, is_json_output=False)
    return 'sí' in raw_response.lower() or 'si' in raw_response.lower()

@st.cache_resource
def get_faiss_index_instance():
    instance = FAISSIndex()
    # Mover el mensaje de éxito/error a la función main para mejor control del flujo
    return instance

def run_visualization_pipeline(user_query: str, df: pd.DataFrame, analysis: Dict, dataset_title: str):
    active_llm_provider = st.session_state.get("current_llm_provider", "gemini")
    st.subheader(f"Analizando consulta (LLM: {active_llm_provider.upper()}): \"{user_query}\"")
    with st.spinner(f"Generando visualizaciones con {active_llm_provider.upper()}..."):
        planner = LLMVisualizerPlanner()
        df_sample_viz = df.head(20) if len(df) > 20 else df.copy()
        viz_configs_suggested = planner.suggest_visualizations(df_sample_viz, user_query, analysis)
        if viz_configs_suggested:
            with st.expander("JSON Sugerencias Visualización", expanded=False): st.json(viz_configs_suggested)
    
    valid_viz_configs_generated = []
    if viz_configs_suggested:
        st.subheader("Visualizaciones sugeridas")
        visualizer = DatasetVisualizer()
        for idx, config in enumerate(viz_configs_suggested):
            title_viz = config.get('titulo_de_la_visualizacion', f'Visualización {idx+1}')
            st.markdown(f"**{idx+1}. {title_viz}**")
            fig = visualizer.plot(df, config)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
                valid_viz_configs_generated.append(config)
            else:
                st.warning(f"No se pudo generar: {title_viz}")
    else:
        st.info(f"IA ({active_llm_provider.upper()}) no sugirió visualizaciones.")
    
    with st.spinner(f"Generando insights con {active_llm_provider.upper()}..."):
        st.subheader("💡 Insights del Analista Virtual")
        interpreter = LLMInterpreter()
        df_sample_ins = df.head(5) if len(df) > 5 else df.copy()
        insights_text = interpreter.generate_insights(user_query, analysis, valid_viz_configs_generated, df_sample_ins, dataset_title)
        st.markdown(insights_text)
    
    st.success("--- ✅ Análisis completado ---")
    st.balloons()


def main():
    st.set_page_config(layout="wide", page_title="Analista Datos Valencia")

    if "current_llm_provider" not in st.session_state: st.session_state.current_llm_provider = "gemini"
    if 'active_df' not in st.session_state: st.session_state.active_df = None
    if 'active_analysis' not in st.session_state: st.session_state.active_analysis = None
    if 'active_dataset_title' not in st.session_state: st.session_state.active_dataset_title = None
    if 'last_query' not in st.session_state: st.session_state.last_query = ""
    if 'run_initial_analysis' not in st.session_state: st.session_state.run_initial_analysis = False

    faiss_index_global = get_faiss_index_instance()
    sentence_model_global = get_sentence_transformer_model(EMBEDDING_MODEL)

    col1, col2 = st.columns([3, 1])
    col1.title("Data València Agent")
    with col2:
        available_llms = [llm for llm, client in [("gemini", gemini_model), ("llama3", groq_client)] if client]
        if available_llms:
            st.session_state.current_llm_provider = st.radio("Selecciona LLM:", options=available_llms, horizontal=True, key="llm_selector")
        else:
            st.error("Ningún LLM configurado. Verifica API Keys.")
            st.stop()
    
    st.sidebar.header("Acciones del Índice")
    if faiss_index_global.is_ready():
        st.sidebar.success(f"Índice FAISS listo ({faiss_index_global.index.ntotal} vectores).")
    
    if st.sidebar.button("Construir/actualizar Índice FAISS"):
        build_and_save_index()

    if st.session_state.active_df is None:
        display_initial_view(faiss_index_global, sentence_model_global)
    else:
        display_conversation_view()

    st.markdown("---")
    st.caption(f"Desarrollado con {EMBEDDING_MODEL} y {st.session_state.get('current_llm_provider','N/A').upper()}.")


def display_initial_view(faiss_index, sentence_model):
    st.markdown('Bienvenido al asistente para explorar [Datos Abiertos del Ayuntamiento de Valencia](https://valencia.opendatasoft.com/pages/home/?flg=es-es).')
    
    if not faiss_index.is_ready():
        st.warning("El índice de búsqueda no está listo. Por favor, constrúyelo desde el menú de la izquierda para poder analizar consultas.")
        return

    st.markdown("##### ¿No sabes qué preguntar? Prueba con esto:")
    examples = [
        "Aparcamientos para bicis", 
        "Intensidad del tráfico en Valencia", 
        "Calidad del aire en la ciudad",
        "Centros educativos"
    ]
    
    if 'user_query_main' not in st.session_state: st.session_state.user_query_main = ""
    
    cols = st.columns(len(examples))
    for i, example in enumerate(examples):
        if cols[i].button(example): 
            st.session_state.user_query_main = example
            st.rerun()

    user_query_input = st.text_input("¿Qué datos te gustaría analizar?", key="user_query_main")

    if st.button("Analizar Consulta", type="primary"):
        if user_query_input:
            api_agent = APIQueryAgent(faiss_index, sentence_model)
            with st.spinner("Buscando y validando datasets..."):
                search_results = api_agent.search_dataset(user_query_input, top_k=5)
                valid_candidates = []
                if search_results:
                    for result in search_results:
                        if result['similarity'] > api_agent.SIMILARITY_THRESHOLD and validate_dataset_relevance(user_query_input, result["metadata"]["title"], result["metadata"]["description"]):
                            valid_candidates.append(result)
            
            if not valid_candidates:
                st.error("No se encontró ningún dataset relevante para tu consulta. Intenta ser más específico o prueba con otra pregunta.")
                if search_results:
                    with st.expander("Resultados de búsqueda con baja relevancia (para depuración):"):
                        st.json([{"title": r['metadata']['title'], "similarity": r['similarity']} for r in search_results])
                return

            selected_dataset_info = sorted(valid_candidates, key=lambda x: x['similarity'], reverse=True)[0]
            dataset_id = selected_dataset_info["metadata"]["id"]
            dataset_title = selected_dataset_info["metadata"]["title"]

            with st.spinner(f"Descargando y analizando '{dataset_title}'..."):
                dataset_bytes = api_agent.export_dataset(dataset_id)
                if not dataset_bytes: st.error(f"Fallo en la descarga del dataset '{dataset_title}'."); return
                df = DatasetLoader.load_dataset_from_bytes(dataset_bytes, dataset_title)
                if df is None or df.empty: st.error(f"El dataset '{dataset_title}' está vacío o no se pudo cargar."); return
                analysis = DatasetAnalyzer().analyze(df)
            
            st.session_state.active_df = df
            st.session_state.active_analysis = analysis
            st.session_state.active_dataset_title = dataset_title
            st.session_state.last_query = user_query_input
            st.session_state.run_initial_analysis = True 
            st.rerun()
        else:
            st.warning("Por favor, introduce una consulta.")


def display_conversation_view():
    st.success(f"Dataset activo: **{st.session_state.active_dataset_title}**")
    csv_data = st.session_state.active_df.to_csv(index=False, sep=';').encode('utf-8')
    st.download_button(label="📥 Descargar Dataset (CSV)", data=csv_data, file_name=f"{sanitize_filename(st.session_state.active_dataset_title)}.csv")
    
    if st.session_state.run_initial_analysis:
        run_visualization_pipeline(st.session_state.last_query, st.session_state.active_df, st.session_state.active_analysis, st.session_state.active_dataset_title)
        st.session_state.run_initial_analysis = False

    st.markdown("---")
    follow_up_query = st.text_input("Haz una pregunta de seguimiento sobre este dataset:", key="follow_up_query")
    
    col_run, col_reset = st.columns([3, 1])
    if col_run.button("Analizar Seguimiento", type="primary"):
        if follow_up_query:
            run_visualization_pipeline(follow_up_query, st.session_state.active_df, st.session_state.active_analysis, st.session_state.active_dataset_title)
        else:
            st.warning("Introduce una consulta de seguimiento.")
    
    if col_reset.button("Finalizar y empezar de nuevo"):
        keys_to_delete = [k for k in st.session_state.keys() if k not in ['current_llm_provider']]
        for key in keys_to_delete:
            del st.session_state[key]
        st.rerun()


if __name__ == "__main__":
    main()