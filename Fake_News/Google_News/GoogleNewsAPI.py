import requests
from langdetect import detect

API_KEY = "AIzaSyDE5zhx3j_fRsIpp7lkzXNrq4-b2pibw3U"  # Reemplaza con tu API key
CX = "763973daa34c3425d"         # Reemplaza con tu ID del motor de búsqueda
QUERY = "Honduras Gana en el Azteca"

# Realizamos la consulta
params = {
    "key": API_KEY,
    "cx": CX,
    "q": QUERY,
    "num": 10,            # Máximo 10 resultados por solicitud
    "lr": "lang_es",      # Restringir idioma a español
    "start": 3,           # Primera página
    # "sortBy": "date",     # Ordenar por fecha
}

response = requests.get("https://www.googleapis.com/customsearch/v1", params=params)

if response.status_code == 200:
    data = response.json()
    noticias = data.get("items", [])

    # Filtrar noticias en español y excluir resultados de YouTube
    noticias_filtradas = [
        noticia for noticia in noticias
        if detect(noticia.get('title', '')) == 'es'  # Filtrar noticias en español
        and 'youtube' not in noticia['link'].lower()  # Excluir resultados de YouTube
    ]

    print("Noticias en español (sin YouTube):")
    for noticia in noticias_filtradas:
        print(f"Título: {noticia['title']}")
        print(f"URL: {noticia['link']}")
        print(f"Descripción: {noticia.get('snippet', 'Sin descripción')}\n")
else:
    print(f"Error al obtener noticias: {response.status_code}")
    print(f"Mensaje de error: {response.json().get('error', {}).get('message', 'Sin mensaje de error')}")
