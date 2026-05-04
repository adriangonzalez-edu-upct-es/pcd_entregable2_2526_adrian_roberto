from abc import ABC
from datetime import datetime
from functools import reduce

from numpy import mean

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
