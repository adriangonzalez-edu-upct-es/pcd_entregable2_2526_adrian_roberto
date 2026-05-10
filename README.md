# Entregable 2 de PCD: Sistema de Recomendación Musical

## ¿Qué es este proyecto?

Este proyecto implementa un sistema básico de gestión y recomendación de música en Python.

## Características Principales

### 🎵 Lo que puedes hacer:
- **Crear un catálogo**: Genera automáticamente canciones, artistas y playlists con datos aleatorios.
- **Escuchar canciones**: Selecciona una canción por su ID y el sistema aprende de tus gustos.
- **Obtener recomendaciones**: Recibe sugerencias de canciones, artistas o playlists.
- **Cambiar estrategias**: Elige cómo buscar elementos (por orden alfabético, fecha o al azar).
- **Configurar recomendaciones**: Decide si quieres recomendaciones de canciones, artistas o playlists.
- **Buscar canciones**: Encuentra una canción específica usando su ID.
- **Ver historial**: Revisa lo que has hecho en la sesión actual.
- **Ver estadísticas**: Mira números sobre tus gustos musicales.

### 🏗️ Cómo está hecho (Patrones de Diseño):
- **Strategy**: Para cambiar cómo buscar cosas.
- **Chain of Responsibility**: Para calcular estadísticas paso a paso.
- **Decorator**: Para construir recomendaciones de forma flexible.
- **Singleton**: Para que cada usuario tenga su propio sistema.

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
│   ├── diagramaUML.uxf     
│   ├── diagrama_casos_uso.uxf
│   └── diagrama_secuencia2.uxf
├── main.py                 # Programa principal
└── README.md               
```
### Diagramas UML (en UMLet):
- **diagramaUML.uxf**: Muestra todas las clases y cómo se relacionan.
- **diagrama_casos_uso.uxf**: Qué puede hacer el usuario.
- **diagrama_secuencia2.uxf**: Cómo fluye el programa paso a paso.

## Autores

Adrian Gonzales Muñoz
Roberto Bermudez Sevilla

