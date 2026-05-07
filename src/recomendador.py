from abc import ABC, abstractmethod
from catalogo import Catalogo, ItemMusical
from estrategias import EstrategiaBusqueda

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
