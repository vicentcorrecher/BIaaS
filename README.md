# Data València Agent  (Analista de Datos Abiertos de Valencia)


**Data València Agent** es una aplicación web interactiva desarrollada con Streamlit y potenciada por Modelos de Lenguaje Grandes (LLMs) como Gemini y Llama 3. Su objetivo es permitir a cualquier usuario explorar el [catálogo de Datos Abiertos del Ayuntamiento de Valencia](https://valencia.opendatasoft.com/pages/home/?flg=es-es) utilizando lenguaje natural.

La aplicación encuentra el dataset más relevante para la consulta del usuario, lo descarga, lo analiza y genera visualizaciones y resúmenes de forma automática, actuando como un analista de datos virtual.

---

##  Características Principales

-   **Búsqueda Semántica**: Utiliza embeddings de sentencias (`sentence-transformers`) y un índice vectorial (FAISS) para encontrar el dataset más relevante para una consulta en lenguaje natural.
-   **Multi-LLM**: Permite cambiar entre diferentes proveedores de LLM (Google Gemini, Llama 3 a través de Groq) para la planificación y generación de insights.
-   **Análisis Automático de Datos**: Identifica automáticamente tipos de columnas (numéricas, categóricas, geoespaciales, temporales).
-   **Generación de Visualizaciones**: El LLM planifica y sugiere los gráficos más adecuados (mapas, barras, líneas, etc.) para responder a la consulta del usuario.
-   **Creación de Insights**: Un agente LLM interpreta los gráficos y los datos para generar un resumen ejecutivo en texto.
-   **Interfaz Interactiva**: Construida con Streamlit para una experiencia de usuario fluida y conversacional.

---

## 🛠️ Arquitectura y Tecnologías

El proyecto sigue una arquitectura modular basada en agentes, donde cada componente tiene una responsabilidad clara:

-   **Frontend**: `Streamlit`
-   **Búsqueda y RAG (Retrieval-Augmented Generation)**:
    -   **Embeddings**: `sentence-transformers` (modelo `paraphrase-MiniLM-L6-v2`)
    -   **Índice Vectorial**: `FAISS`
-   **Modelos de Lenguaje (LLMs)**:
    -   `Google Gemini` (a través de `google-generativeai`)
    -   `Llama 3` (a través de `groq`)
-   **Análisis y Manipulación de Datos**: `Pandas`, `NumPy`
-   **Visualización**: `Plotly Express`
-   **Gestión de Dependencias**: `pip` y `requirements.txt`
-   **Gestión de Secrets**: `python-dotenv`

---

## ⚙️ Instalación y Ejecución Local

Sigue estos pasos para ejecutar el proyecto en tu máquina local.

### Prerrequisitos

-   Python 3.8+
-   Git

### 1. Clonar el Repositorio

```bash
git clone https://github.com/TU_USUARIO/TU_REPOSITORIO.git
cd TU_REPOSITORIO
```


### 2. Crear y activar un entorno virtual
Es una buena práctica aislar las dependencias del proyecto para evitar conflictos.
```bash
# Crear el entorno virtual
python -m venv venv

# Activar el entorno (en Windows)
.\venv\Scripts\activate

# Activar el entorno (en macOS o Linux)
source venv/bin/activate
```
Sabrás que está activado porque tu terminal mostrará (venv) al principio de la línea.

### 3. Instalar las dependencias
Con el entorno activado, instala todas las librerías necesarias con un solo comando.
``` bash
pip install -r requirements.txt
```
### 4. Configurar las Claves de API
El proyecto necesita claves para acceder a los servicios de LLM (Google Gemini y Groq).
En la raíz del proyecto, crea un fichero llamado .env.
Abre el fichero .env y añade tus claves con el siguiente formato, reemplazando los valores de ejemplo:
```bash
ini
API_KEY_GEMINI="TU_API_KEY_DE_GOOGLE_AI_STUDIO"
API_KEY_GROQ="TU_API_KEY_DE_GROQ"
```

Este fichero está incluido en el .gitignore, por lo que tus claves secretas nunca se subirán al repositorio.

### 5. Ejecutar la aplicación
¡Ya está todo listo! Inicia la aplicación Streamlit con este comando:
``` bash
streamlit run app.py
```
La aplicación se abrirá automáticamente en una nueva pestaña de tu navegador.
### 6. Construir el Índice FAISS (Solo la primera vez)
Para que la búsqueda funcione, necesitas crear el índice vectorial localmente.
Cuando la aplicación se inicie, verás un menú en la barra lateral izquierda.
Haz clic en el botón "Construir/actualizar Índice FAISS".
El proceso comenzará y puede tardar varios minutos. Descargará la información de más de 280 datasets y generará sus embeddings.
Una vez que veas el mensaje de éxito y los globos, el índice estará creado y la aplicación estará 100% funcional.
💡 Uso
Escribe una consulta en lenguaje natural en el campo de texto principal (ej: "¿Dónde hay aparcamientos para bicis?").
Haz clic en "Analizar Consulta".
El agente buscará el dataset más relevante, lo analizará y te presentará visualizaciones e insights.
Puedes realizar preguntas de seguimiento sobre el dataset activo.

📈 Posibles mejoras futuras
Implementar un sistema de caché más avanzado para los resultados de la API.
Permitir al usuario seleccionar manualmente un dataset si la búsqueda semántica no es precisa.
Añadir soporte para más tipos de visualizaciones.
Mejorar la gestión de memoria para datasets muy grandes.


Agradecimientos
A OpenData València por proporcionar los datos.
A las comunidades de Streamlit, Hugging Face y FAISS.


Una vez que tengas este fichero `README.md` guardado en tu carpeta de proyecto, solo tienes que subirlo a GitHub con estos comandos:


# Añade el nuevo README.md (y cualquier otro cambio)
git add README.md

# Guarda los cambios con un mensaje descriptivo
git commit -m "docs: Add professional README file"

# Sube los cambios a tu repositorio en GitHub
git push origin main
