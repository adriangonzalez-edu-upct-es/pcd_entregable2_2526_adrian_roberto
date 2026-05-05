from abc import ABC
from datetime import datetime
from functools import reduce
from typing import List, Dict, Optional
import string
from numpy import random, mean, timedelta

"""
Módulo para gestionar un catálogo de elementos musicales.

Este módulo define clases para representar elementos musicales, incluyendo
una clase abstracta ItemMusical y una implementación concreta Cancion.
También incluye excepciones personalizadas para manejar errores.
"""

class CatalogoError(Exception):
    """Excepción base para errores relacionados con el catálogo."""
    pass


class ItemMusicalError(CatalogoError):
    """Excepción para errores específicos de elementos musicales."""
    pass


class ItemMusical(ABC):
    """
    Clase abstracta que representa un elemento musical genérico.

    Proporciona métodos para validar y acceder a atributos básicos como nombre,
    fecha de creación y atributos sonoros y sentimentales.
    """

    def __init__(
        self,
        nombre: str,
        fecha_creacion: datetime,
        atributos_sonoros: dict[str, float],
        atributos_sentimentales: dict[str, float],
    ):
        """
        Inicializa un ItemMusical.
        Raises:
            ItemMusicalError: Si los parámetros no son válidos.
        """
        self._validar_nombre(nombre)
        self._validar_fecha(fecha_creacion)
        self._validar_atributos(atributos_sonoros, "sonoros")
        self._validar_atributos(atributos_sentimentales, "sentimentales")

        self.nombre = nombre
        self.fecha_creacion = fecha_creacion
        self.atributos_sonoros = atributos_sonoros.copy()
        self.atributos_sentimentales = atributos_sentimentales.copy()

    def _validar_nombre(self, nombre: str) -> None:
        """
        Valida que el nombre sea una cadena no vacía.
        Raises:
            ItemMusicalError: Si el nombre no es válido.
        """
        if not isinstance(nombre, str) or not nombre.strip():
            raise ItemMusicalError("El nombre debe ser una cadena no vacía.")

    def _validar_fecha(self, fecha: datetime) -> None:
        """
        Valida que la fecha sea un objeto datetime.
        Raises:
            ItemMusicalError: Si la fecha no es válida.
        """
        if not isinstance(fecha, datetime):
            raise ItemMusicalError("La fecha de creación debe ser un objeto datetime.")

    def _validar_atributos(self, atributos: dict[str, float], etiqueta: str) -> None:
        """
        Valida que los atributos sean un diccionario de valores numéricos.
        Raises:
            ItemMusicalError: Si los atributos no son válidos.
        """
        if not isinstance(atributos, dict) or not all(
            isinstance(nombre, str) and isinstance(valor, (int, float))
            for nombre, valor in atributos.items()
        ):
            raise ItemMusicalError(f"Los atributos {etiqueta} deben ser un diccionario de valores numéricos.")

    def obtener_nombre(self) -> str:
        """
        Obtiene el nombre del elemento musical.
        Returns:
            str: El nombre.
        """
        return self.nombre

    def obtener_fecha(self) -> datetime:
        """
        Obtiene la fecha de creación del elemento musical.
        Returns:
            datetime: La fecha de creación.
        """
        return self.fecha_creacion

    def obtener_atributos_sonoros(self) -> dict[str, float]:
        """
        Obtiene una copia de los atributos sonoros.
        Returns:
            dict[str, float]: Copia de los atributos sonoros.
        """
        return self.atributos_sonoros.copy()

    def obtener_atributos_sentimentales(self) -> dict[str, float]:
        """
        Obtiene una copia de los atributos sentimentales.
        Returns:
            dict[str, float]: Copia de los atributos sentimentales.
        """
        return self.atributos_sentimentales.copy()
    
    def distancia_al_sesion(self, estadisticas: dict[str, any]) -> float:
        """
        Calcula la distancia euclidia al centro de una sesión basado en estadísticas.
        Returns:
            float: La distancia calculada, o infinito si no hay atributos comunes.
        Raises:
            ItemMusicalError: Si las estadísticas son inválidas.
        """
        try:
            medias_sonoras = estadisticas.get("sonoros", {}).get("media", {})
            medias_sentimentales = estadisticas.get("sentimentales", {}).get("media", {})
        except AttributeError as exc:
            raise ItemMusicalError("Estadísticas inválidas.") from exc

        # Calcular distancia para atributos sonoros
        distancia_sonora = sum(
            map(
                lambda par: (par[1] - medias_sonoras[par[0]]) ** 2,
                filter(lambda par: par[0] in medias_sonoras, self.atributos_sonoros.items()),
            )
        )
        # Calcular distancia para atributos sentimentales
        distancia_sentimental = sum(
            map(
                lambda par: (par[1] - medias_sentimentales[par[0]]) ** 2,
                filter(lambda par: par[0] in medias_sentimentales, self.atributos_sentimentales.items()),
            )
        )

        # Contar atributos comunes
        total_atributos = len(list(filter(lambda clave: clave in medias_sonoras, self.atributos_sonoros))) + len(
            list(filter(lambda clave: clave in medias_sentimentales, self.atributos_sentimentales))
        )

        if total_atributos == 0:
            return float("inf")

        return ((distancia_sonora + distancia_sentimental) / total_atributos) ** 0.5
    
    def coincide_con_estadisticas(
        self, estadisticas: dict[str, any], umbral: float = 0.18
    ) -> bool:
        """
        Verifica si el elemento coincide con las estadísticas dentro de un umbral.
        Returns:
            bool: True si coincide, False en caso contrario.
        """
        try:
            return self.distancia_al_sesion(estadisticas) <= umbral
        except ItemMusicalError:
            return False
    
    @staticmethod
    def promedio_atributos(
        canciones: list["Cancion"], tipo: str
    ) -> dict[str, float]:
        """
        Calcula el promedio de atributos para una lista de canciones.
        Returns:
            dict[str, float]: Diccionario con promedios por atributo.
        """
        if not canciones:
            return {}

        def agregar_atributos(acumulador: dict[str, list[float]], cancion: "Cancion") -> dict[str, list[float]]:
            """
            Función auxiliar para acumular valores de atributos.
            Returns:
                dict[str, list[float]]: Acumulador actualizado.
            """
            valores = (
                cancion.obtener_atributos_sonoros()
                if tipo == "sonoros"
                else cancion.obtener_atributos_sentimentales()
            )
            for nombre, valor in valores.items():
                acumulador.setdefault(nombre, []).append(valor)
            return acumulador

        valores_por_atributo = reduce(agregar_atributos, canciones, {})
        return dict(
            map(
                lambda elemento: (elemento[0], mean(elemento[1])),
                valores_por_atributo.items(),
            )
        )


class Cancion(ItemMusical):
    """
    Clase que representa una canción, heredando de ItemMusical.
    Añade atributos específicos como identificador, duración y artistas.
    """

    def __init__(
        self,
        identificador: str,
        nombre: str,
        fecha_creacion: datetime,
        duracion: float,
        artistas: list[str],
        atributos_sonoros: dict[str, float],
        atributos_sentimentales: dict[str, float],
    ):
        """
        Inicializa una Cancion.
        Raises:
            ItemMusicalError: Si los parámetros no son válidos.
        """
        super().__init__(nombre, fecha_creacion, atributos_sonoros, atributos_sentimentales)
        if not isinstance(identificador, str) or not identificador.strip():
            raise ItemMusicalError("El identificador debe ser una cadena no vacía.")
        if not isinstance(duracion, (int, float)) or duracion <= 0:
            raise ItemMusicalError("La duración debe ser un número positivo.")
        if not isinstance(artistas, list) or not all(isinstance(artista, str) for artista in artistas):
            raise ItemMusicalError("Los artistas deben ser una lista de cadenas.")

        self.identificador = identificador
        self.duracion = duracion
        self.artistas = artistas.copy()

    def __str__(self) -> str:
        """
        Representación en cadena de la canción.
        Returns:
            str: Representación de la canción.
        """
        return f"Cancion(id={self.identificador}, nombre={self.nombre})"

class Playlist(ItemMusical):
    """
    Clase que representa una playlist, heredando de ItemMusical.
    Añade atributos específicos como nombre y lista de canciones.
    """
    def __init__(
        self,
        nombre: str,
        fecha_creacion: datetime,
        canciones: List[Cancion],
    ):
        """
        Inicializa una Playlist.
        Raises:
            ItemMusicalError: Si los parámetros no son válidos.
        """
        if not isinstance(canciones, list) or not all(isinstance(cancion, Cancion) for cancion in canciones):
            raise ItemMusicalError("La playlist debe contener una lista de Cancion.")

        atributos_sonoros = self._calcular_atributos(canciones, "sonoros")
        atributos_sentimentales = self._calcular_atributos(canciones, "sentimentales")
        super().__init__(nombre, fecha_creacion, atributos_sonoros, atributos_sentimentales)
        self.canciones = canciones.copy()

    def _calcular_atributos(
        self, canciones: List[Cancion], tipo: str
    ) -> Dict[str, float]:
        return ItemMusical.promedio_atributos(canciones, tipo)

    def __str__(self) -> str:
        return f"Playlist(nombre={self.nombre}, canciones={len(self.canciones)})"

class Artista(ItemMusical):
    """
    Clase que representa un artista, heredando de ItemMusical.
    Añade atributos específicos como nombre, fecha de nacimiento, lista de canciones y géneros musicales
    """
    def __init__(
        self,
        nombre: str,
        fecha_nacimiento: datetime,
        canciones: List[Cancion],
        generos: Optional[List[str]] = None,
    ):
       """
       Inicializa un Artista.
       Raises:
              ItemMusicalError: Si los parámetros no son válidos.
       """ 
        if not isinstance(canciones, list) or not all(isinstance(cancion, Cancion) for cancion in canciones):
            raise ItemMusicalError("El artista debe tener una lista de Cancion.")
        if generos is not None and not all(isinstance(genero, str) for genero in generos):
            raise ItemMusicalError("Los géneros deben ser una lista de cadenas.")

        atributos_sonoros = self._calcular_atributos(canciones, "sonoros")
        atributos_sentimentales = self._calcular_atributos(canciones, "sentimentales")
        super().__init__(nombre, fecha_nacimiento, atributos_sonoros, atributos_sentimentales)
        self.canciones = canciones.copy()
        self.generos = generos or []

    def _calcular_atributos(
        self, canciones: List[Cancion], tipo: str
    ) -> Dict[str, float]:
        return ItemMusical.promedio_atributos(canciones, tipo)

    def __str__(self) -> str:
        return f"Artista(nombre={self.nombre}, canciones={len(self.canciones)})"

class Catalogo:
    """Catálogo musical.

    El catálogo agrupa canciones, artistas y playlists. Permite validar la colección
    al construirse, buscar canciones por identificador y generar catálogos aleatorios.
    """

    def __init__(
        self,
        canciones: Optional[List[Cancion]] = None,
        artistas: Optional[List[Artista]] = None,
        playlists: Optional[List[Playlist]] = None,
    ):
        # Validar que cada lista contenga objetos del tipo correcto
        if canciones is not None and not all(isinstance(c, Cancion) for c in canciones):
            raise CatalogoError("Las canciones deben ser instancias de Cancion.")
        if artistas is not None and not all(isinstance(a, Artista) for a in artistas):
            raise CatalogoError("Los artistas deben ser instancias de Artista.")
        if playlists is not None and not all(isinstance(p, Playlist) for p in playlists):
            raise CatalogoError("Las playlists deben ser instancias de Playlist.")

        self.canciones = canciones or []
        self.artistas = artistas or []
        self.playlists = playlists or []

    def buscar_cancion_por_id(self, identificador: str) -> Optional[Cancion]:
        """Buscar una canción en el catálogo a partir de su identificador."""
        if not isinstance(identificador, str) or not identificador.strip():
            raise CatalogoError("El identificador debe ser una cadena no vacía.")
        return next(
            filter(lambda c: c.identificador == identificador, self.canciones),
            None,
        )

    @staticmethod
    def _nombre_aleatorio(prefijo: str) -> str:
        # Genera un nombre aleatorio para un artista o playlist
        sufijo = "".join(random.choices(string.ascii_uppercase, k=3))
        return f"{prefijo} {sufijo}"

    @classmethod
    def generar_catalogo_aleatorio(
        cls,
        num_canciones: int = 36,
        num_artistas: int = 6,
        num_playlists: int = 5,
    ) -> "Catalogo":
        """Genera un catálogo de ejemplo con canciones, artistas y playlists aleatorias."""
        if not all(
            isinstance(value, int) and value > 0
            for value in (num_canciones, num_artistas, num_playlists)
        ):
            raise CatalogoError("Los parámetros de generación deben ser enteros positivos.")

        # Los atributos sonoros y sentimentales se definen de forma fija
        # para construir ejemplos consistentes y comparar resultados.
        atributos_sonoros = ["ritmo", "tono", "escala"]
        atributos_sentimentales = ["felicidad", "bailabilidad", "energia"]

        def cancion_random(index: int, artista: str) -> Cancion:
            # Las canciones se crean con valores aleatorios uniformes en [0, 1]
            # para que los atributos sean comparables entre items.
            sonido = {nombre: random.random() for nombre in atributos_sonoros}
            sentimiento = {nombre: random.random() for nombre in atributos_sentimentales}

            # La fecha de creación se distribuye en los últimos 10 años para
            # simular un catálogo contemporáneo sin fechas demasiado antiguas.
            fecha = datetime.now() - timedelta(days=random.randint(1, 3650))
            return Cancion(
                identificador=f"c{index:03d}",
                nombre=f"Cancion {index}",
                fecha_creacion=fecha,
                duracion=random.uniform(180, 300),
                artistas=[artista],
                atributos_sonoros=sonido,
                atributos_sentimentales=sentimiento,
            )

        artistas = []
        canciones: List[Cancion] = []

        # Crear artistas y asignarles un subconjunto de canciones generado.
        # Se usa `num_canciones // num_artistas` para distribuir canciones de manera equilibrada.
        for index in range(num_artistas):
            nombre_artista = Catalogo._nombre_aleatorio("Artista")
            canciones_por_artista = list(
                map(
                    lambda i: cancion_random(i + 1 + index * 10, nombre_artista),
                    range(max(1, num_canciones // num_artistas)),
                )
            )

            # Cada artista recibe una edad aleatoria entre aproximadamente 22 y 49 años.
            artista = Artista(
                nombre=nombre_artista,
                fecha_nacimiento=datetime.now() - timedelta(days=random.randint(8000, 18000)),
                canciones=canciones_por_artista,
                generos=[random.choice(["pop", "rock", "dance", "indie"])],
            )
            artistas.append(artista)
            canciones.extend(canciones_por_artista)

        # Crear playlists seleccionando canciones al azar para simular listas de reproducción variadas.
        playlists = list(
            map(
                lambda _: Playlist(
                    nombre=Catalogo._nombre_aleatorio("Playlist"),
                    fecha_creacion=datetime.now() - timedelta(days=random.randint(1, 2000)),
                    canciones=random.sample(canciones, min(6, len(canciones))),
                ),
                range(num_playlists),
            )
        )

        # Construir el catálogo final con todos los elementos generados.
        return cls(canciones=canciones, artistas=artistas, playlists=playlists)