from abc import ABC, abstractmethod
import asyncio
from datetime import datetime
from functools import reduce
from src.catalogo import Cancion, Catalogo, ItemMusical
from src.estrategias import BusquedaAlfabetica, EstrategiaBusqueda
from src.estadisticas import CalculadorSentimental, CalculadorSonoro

class RecomendadorError(Exception):
    pass


class RecomendadorComponent(ABC):
    """Interfaz base para componentes de recomendación."""

    @abstractmethod
    def recomendar(
        self, catalogo: Catalogo, estadisticas: dict[str, dict], estrategia: EstrategiaBusqueda
    ) -> list[dict[str, ItemMusical]]:
        """Devuelve recomendaciones según el catálogo, estadísticas y estrategia."""
        raise NotImplementedError


class RecomendadorVacio(RecomendadorComponent):
    def recomendar(
        self, catalogo: Catalogo, estadisticas: dict[str, dict], estrategia: EstrategiaBusqueda
    ) -> list[dict[str, ItemMusical]]:
        """Devuelve una lista vacía cuando no hay recomendaciones."""
        return []


class RecomendadorDecorator(RecomendadorComponent):
    """Decorador base que delega la recomendación al componente envuelto."""

    def __init__(self, componente_envuelto: RecomendadorComponent):
        """Inicializa el decorador con el componente que se va a envolver."""
        self.componente_envuelto = componente_envuelto

    def recomendar(
        self, catalogo: Catalogo, estadisticas: dict[str, dict], estrategia: EstrategiaBusqueda
    ) -> list[dict[str, ItemMusical]]:
        """Devuelve recomendaciones producidas por el componente envuelto."""
        return self.componente_envuelto.recomendar(catalogo, estadisticas, estrategia)


class RecomendadorCancionDecorator(RecomendadorDecorator):
    """Decorador que añade una recomendación de canción al resultado."""

    def recomendar(
        self, catalogo: Catalogo, estadisticas: dict[str, dict], estrategia: EstrategiaBusqueda
    ) -> list[dict[str, ItemMusical]]:
        """Agrega la canción seleccionada por la estrategia a la lista de resultados."""
        resultados = super().recomendar(catalogo, estadisticas, estrategia)
        cancion = estrategia.buscar(catalogo.canciones, estadisticas)
        resultados.append({"tipo": "cancion", "item": cancion})
        return resultados


class RecomendadorArtistasDecorator(RecomendadorDecorator):
    """Decorador que añade una recomendación de artista al resultado."""

    def recomendar(
        self, catalogo: Catalogo, estadisticas: dict[str, dict], estrategia: EstrategiaBusqueda
    ) -> list[dict[str, ItemMusical]]:
        """Agrega el artista seleccionado por la estrategia a la lista de resultados."""
        resultados = super().recomendar(catalogo, estadisticas, estrategia)
        artista = estrategia.buscar(catalogo.artistas, estadisticas)
        resultados.append({"tipo": "artista", "item": artista})
        return resultados


class RecomendadorPlaylistsDecorator(RecomendadorDecorator):
    """Decorador que añade una recomendación de playlist al resultado."""

    def recomendar(
        self, catalogo: Catalogo, estadisticas: dict[str, dict], estrategia: EstrategiaBusqueda
    ) -> list[dict[str, ItemMusical]]:
        """Agrega la playlist seleccionada por la estrategia a la lista de resultados."""
        resultados = super().recomendar(catalogo, estadisticas, estrategia)
        playlist = estrategia.buscar(catalogo.playlists, estadisticas)
        resultados.append({"tipo": "playlist", "item": playlist})
        return resultados


class SistemaRecomendacionUsuario:
    _instancias: dict[str, "SistemaRecomendacionUsuario"] = {}

    def __new__(cls, id_usuario: str, *args, **kwargs):
        if not isinstance(id_usuario, str) or not id_usuario.strip():
            raise RecomendadorError("El identificador de usuario debe ser una cadena no vacía.")
        # Si existe una instancia para este usuario, la devolvemos
        if id_usuario in cls._instancias:
            return cls._instancias[id_usuario]
        # Si no existe, la creamos
        instancia = super().__new__(cls)
        cls._instancias[id_usuario] = instancia
        return instancia

    def __init__(self, id_usuario: str):
        if hasattr(self, "id_usuario"):
            return
        self.id_usuario = id_usuario
        self.historial_sesion: list[dict[str, object]] = []
        self.estadisticas: dict[str, dict] = {
            "sonoros": {"media": {}, "desviacion": {}},
            "sentimentales": {"media": {}, "desviacion": {}},
        }
        self.estrategia: EstrategiaBusqueda = BusquedaAlfabetica()
        self.tipos_recomendacion = {"cancion"}
        # Construimos la cadena de recomendadores según los tipos activos
        self._recomendador = self._construir_recomendador()
        # Patron Chain of Responsibility para el calculo de estadisticas
        self._calculador_estadisticas = CalculadorSonoro(sucesor=CalculadorSentimental())
        self._max_canciones = 10

    def _construir_recomendador(self) -> RecomendadorComponent:
        """Construye una cadena de decoradores según los tipos de recomendación activos."""
        opciones = {
            "cancion": RecomendadorCancionDecorator,
            "artista": RecomendadorArtistasDecorator,
            "playlist": RecomendadorPlaylistsDecorator,
        }
        # Patron Decorator, para construir la cadena de recomendadores según los tipos seleccionados
        return reduce(
            lambda acumulado, tipo: opciones[tipo](acumulado),
            sorted(self.tipos_recomendacion),
            RecomendadorVacio(),
        )

    # Para cambiar la estrategia de busqueda
    def set_estrategia(self, estrategia: EstrategiaBusqueda) -> None:
        """Establece la estrategia de búsqueda para las recomendaciones."""
        if not isinstance(estrategia, EstrategiaBusqueda):
            raise RecomendadorError("La estrategia debe implementar EstrategiaBusqueda.")
        self.estrategia = estrategia

    # Para agregar tips de recomendacion para el patron Decorator
    def agregar_tipo_recomendacion(self, tipo: str) -> None:
        """Agrega un tipo de recomendación (canción, artista o playlist) al sistema."""
        if tipo not in {"cancion", "artista", "playlist"}:
            raise RecomendadorError(f"Tipo de recomendación desconocido: {tipo}")
        self.tipos_recomendacion.add(tipo)
        # Reconstruimos la cadena de recomendadores cada vez que se agrega
        self._recomendador = self._construir_recomendador()

    # Para eliminar tips de recomendacion para el patron Decorator
    def remover_tipo_recomendacion(self, tipo: str) -> None:
        """Elimina un tipo de recomendación manteniendo al menos la de canción."""
        self.tipos_recomendacion.discard(tipo)
        if not self.tipos_recomendacion:
            self.tipos_recomendacion.add("cancion")
        # Reconstruimos la cadena de recomendadores cada vez que se agrega
        self._recomendador = self._construir_recomendador()

    def escuchar_cancion(self, cancion: Cancion, fecha_hora: datetime) -> None:
        """Registra una canción escuchada y actualiza las estadísticas del usuario."""
        if not isinstance(cancion, Cancion):
            raise RecomendadorError("Sólo se pueden escuchar objetos de tipo Cancion.")
        if not isinstance(fecha_hora, datetime):
            raise RecomendadorError("La fecha y hora debe ser un objeto datetime.")
        # Agrega el evento al historial de la sesión y mantiene solo las últimas N canciones para el cálculo de estadísticas
        evento = {"cancion": cancion, "fecha_hora": fecha_hora}
        self.historial_sesion.append(evento)
        self.historial_sesion = self.historial_sesion[-self._max_canciones:]
        # Actualiza las estadísticas utilizando el patrón Chain of Responsibility
        canciones_historial = list(map(lambda evento: evento["cancion"], self.historial_sesion))
        self._calculador_estadisticas.manejar(canciones_historial, self.estadisticas)

    # Version asincrona del método escuchar_cancion para no bloquear la ejecución principal
    async def escuchar_cancion_async(self, cancion: Cancion, fecha_hora: datetime) -> None:
        """Versión asincrónica para registrar una canción escuchada."""
        await asyncio.to_thread(self.escuchar_cancion, cancion, fecha_hora)

    def obtener_recomendaciones(self, catalogo: Catalogo) -> list[dict[str, ItemMusical]]:
        """Genera recomendaciones personalizadas basadas en las estadísticas del usuario."""
        if not isinstance(catalogo, Catalogo):
            raise RecomendadorError("El catálogo debe ser una instancia de Catalogo.")
        try:
            # Patron Decorator, para obtener recomendaciones según la cadena de recomendadores construida
            return self._recomendador.recomendar(catalogo, self.estadisticas, self.estrategia)
        except Exception as error:
            raise RecomendadorError("No fue posible obtener recomendaciones.") from error
    # Version asincrona del método obtener_recomendaciones para no bloquear la ejecución principal
    async def obtener_recomendaciones_async(self, catalogo: Catalogo) -> list[dict[str, ItemMusical]]:
        """Versión asincrónica para obtener recomendaciones personalizadas."""
        return await asyncio.to_thread(self.obtener_recomendaciones, catalogo)