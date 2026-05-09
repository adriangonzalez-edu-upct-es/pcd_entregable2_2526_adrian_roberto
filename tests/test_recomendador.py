from datetime import datetime
import pytest

from src.catalogo import Artista, Cancion, Catalogo, Playlist
from src.estrategias import BusquedaAlfabetica
from src.recomendador import RecomendadorError, SistemaRecomendacionUsuario


def crear_cancion_prueba(identificador: str, nombre: str, fecha: datetime, ritmo: float, felicidad: float) -> Cancion:
    return Cancion(
        identificador=identificador,
        nombre=nombre,
        fecha_creacion=fecha,
        duracion=210.0,
        artistas=["Artista Test"],
        atributos_sonoros={"ritmo": ritmo},
        atributos_sentimentales={"felicidad": felicidad},
    )


def test_sistema_recomendacion_usuario_identificador_invalido():
    with pytest.raises(RecomendadorError):
        SistemaRecomendacionUsuario("")


def test_escuchar_cancion_actualiza_historial_y_estadisticas():
    sistema = SistemaRecomendacionUsuario("usuario_recom_1")
    cancion = crear_cancion_prueba("c001", "Cancion Test", datetime(2024, 1, 1), 0.5, 0.8)

    sistema.escuchar_cancion(cancion, datetime(2024, 1, 2))

    assert len(sistema.historial_sesion) == 1
    assert sistema.historial_sesion[0]["cancion"] == cancion
    assert sistema.estadisticas["sonoros"]["media"]["ritmo"] == pytest.approx(0.5)
    assert sistema.estadisticas["sentimentales"]["media"]["felicidad"] == pytest.approx(0.8)


def test_obtener_recomendaciones_devuelve_cancion_por_defecto():
    sistema = SistemaRecomendacionUsuario("usuario_recom_2")
    cancion = crear_cancion_prueba("c001", "Cancion Test", datetime(2024, 1, 1), 0.5, 0.8)
    catalogo = Catalogo(canciones=[cancion])

    sistema.escuchar_cancion(cancion, datetime(2024, 1, 2))
    recomendaciones = sistema.obtener_recomendaciones(catalogo)

    assert len(recomendaciones) == 1
    assert recomendaciones[0]["tipo"] == "cancion"
    assert recomendaciones[0]["item"] == cancion


def test_agregar_tipos_recomendacion_incluye_artista_y_playlist():
    sistema = SistemaRecomendacionUsuario("usuario_recom_3")
    cancion = crear_cancion_prueba("c001", "Cancion Test", datetime(2024, 1, 1), 0.5, 0.8)
    artista = Artista("Artista Test", datetime(1990, 1, 1), [cancion])
    playlist = Playlist("Playlist Test", datetime(2024, 1, 1), [cancion])
    catalogo = Catalogo(canciones=[cancion], artistas=[artista], playlists=[playlist])

    sistema.escuchar_cancion(cancion, datetime(2024, 1, 2))
    sistema.agregar_tipo_recomendacion("artista")
    sistema.agregar_tipo_recomendacion("playlist")

    recomendaciones = sistema.obtener_recomendaciones(catalogo)
    tipos = [recomendacion["tipo"] for recomendacion in recomendaciones]

    assert tipos == ['artista', 'cancion', 'playlist']
    assert recomendaciones[0]["item"] == artista
    assert recomendaciones[1]["item"] == cancion
    assert recomendaciones[2]["item"] == playlist


def test_set_estrategia_invalida_raise():
    sistema = SistemaRecomendacionUsuario("usuario_recom_4")
    with pytest.raises(RecomendadorError):
        sistema.set_estrategia("no es estrategia")


def test_obtener_recomendaciones_con_catalogo_invalido_raise():
    sistema = SistemaRecomendacionUsuario("usuario_recom_5")
    with pytest.raises(RecomendadorError):
        sistema.obtener_recomendaciones("no es catalogo")
