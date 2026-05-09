# Entregable 2 de PCD: Sistema de Recomendación Musical

## ¿Qué es este proyecto?

Este proyecto es un **sistema de recomendación musical** hecho en Python. Es parte del curso de Programación Científica de Datos (PCD). El sistema te permite crear un catálogo de canciones, escuchar música y recibir recomendaciones personalizadas basadas en tus gustos.

Imagina que es como Spotify, pero más simple y enfocado en aprender patrones de diseño de software.

## Características Principales

### 🎵 Lo que puedes hacer:
- **Crear un catálogo**: Genera automáticamente canciones, artistas y playlists con datos aleatorios.
- **Escuchar canciones**: Selecciona una canción por su ID y el sistema aprende de tus gustos.
- **Obtener recomendaciones**: Recibe sugerencias de canciones, artistas o playlists que te podrían gustar.
- **Cambiar estrategias**: Elige cómo buscar elementos (por orden alfabético, fecha o al azar).
- **Configurar recomendaciones**: Decide si quieres recomendaciones de canciones, artistas o playlists.
- **Buscar canciones**: Encuentra una canción específica usando su ID.
- **Ver historial**: Revisa lo que has hecho en la sesión actual.
- **Ver estadísticas**: Mira números sobre tus gustos musicales (como el ritmo promedio que te gusta).

### 🏗️ Cómo está hecho (Patrones de Diseño):
- **Strategy**: Para cambiar cómo buscar cosas.
- **Chain of Responsibility**: Para calcular estadísticas paso a paso.
- **Decorator**: Para construir recomendaciones de forma flexible.
- **Singleton**: Para que cada usuario tenga su propio sistema.

## Instalación

### ¿Qué necesitas?
- Python 3.8 o más nuevo (viene con Windows si tienes Anaconda).
- Un editor de código como VS Code.

### Pasos para instalar:
1. Descarga o clona este proyecto en tu computadora.
2. Abre una terminal (PowerShell en Windows) y ve a la carpeta del proyecto:
   ```
   cd ruta/al/proyecto/pcd_entregable2_2526_adrian_roberto
   ```

### Ejecutar el programa:
```
python main.py
```

### Menú principal:
Cuando ejecutes el programa, verás un menú como este:

```
Bienvenido al Sistema de Recomendación Musical

1. Generar catálogo
2. Escuchar canción
3. Obtener recomendaciones
4. Cambiar estrategia de búsqueda
5. Configurar tipos de recomendación
6. Buscar canción por ID
7. Ver historial de sesión
8. Ver estadísticas de sesión
9. Salir

Selecciona una opción:
```

## Estructura del Proyecto

```
pcd_entregable2_2526_adrian_roberto/
├── src/                    # Código fuente
│   ├── catalogo.py         # Clases para canciones, artistas, playlists
│   ├── estadisticas.py     # Cálculo de estadísticas musicales
│   ├── estrategias.py      # Diferentes formas de buscar
│   ├── recomendador.py     # Sistema de recomendaciones
├── tests/                  # Pruebas del código
│   ├── test_catalogo.py
│   ├── test_estadisticas.py
│   ├── test_estrategias.py
│   └── test_recomendador.py
├── diagramas/              # Diagramas UML y de secuencia
│   ├── diagramaUML.uxf     # Diagrama de clases
│   ├── diagrama_casos_uso.uxf
│   └── diagrama_secuencia2.uxf
├── main.py                 # Programa principal
└── README.md               # Este archivo
```

### Ejecutar pruebas:
```
python -m pytest tests/
```

### Diagramas UML (en UMLet):
- **diagramaUML.uxf**: Muestra todas las clases y cómo se relacionan.
- **diagrama_casos_uso.uxf**: Qué puede hacer el usuario.
- **diagrama_secuencia2.uxf**: Cómo fluye el programa paso a paso.

## Autores

Adrian Gonzales Muñoz
Roberto Bermudez Sevilla

