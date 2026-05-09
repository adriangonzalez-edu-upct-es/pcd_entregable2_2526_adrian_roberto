from datetime import datetime
import pytest

from src.catalogo import (
    Artista,
    Cancion,
    Catalogo,
    CatalogoError,
    ItemMusical,
    ItemMusicalError,
    Playlist,
)


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


def test_promedio_atributos_vacio_devuelve_diccionario_vacio():
    assert ItemMusical.promedio_atributos([], "sonoros") == {}


def test_promedio_atributos_calcula_media_correcta():
    cancion1 = crear_cancion_prueba("c001", "A", datetime(2024, 1, 1), 0.2, 0.4)
    cancion2 = crear_cancion_prueba("c002", "B", datetime(2024, 1, 1), 0.8, 0.6)
    promedio_sonoros = ItemMusical.promedio_atributos([cancion1, cancion2], "sonoros")
    promedio_sentimentales = ItemMusical.promedio_atributos([cancion1, cancion2], "sentimentales")

    assert promedio_sonoros == {"ritmo": pytest.approx(0.5)}
    assert promedio_sentimentales == {"felicidad": pytest.approx(0.5)}


def test_cancion_valida_crea_instancia_y_str():
    cancion = crear_cancion_prueba("c001", "Test", datetime(2024, 1, 1), 0.3, 0.7)
    assert cancion.identificador == "c001"
    assert cancion.obtener_nombre() == "Test"
    assert str(cancion) == "Cancion(id=c001, nombre=Test)"


def test_cancion_validacion_identificador_no_valido():
    with pytest.raises(ItemMusicalError):
        Cancion(
            identificador="",
            nombre="Test",
            fecha_creacion=datetime(2024, 1, 1),
            duracion=210.0,
            artistas=["Artista"],
            atributos_sonoros={"ritmo": 0.5},
            atributos_sentimentales={"felicidad": 0.5},
        )


def test_cancion_validacion_duracion_no_valida():
    with pytest.raises(ItemMusicalError):
        Cancion(
            identificador="c001",
            nombre="Test",
            fecha_creacion=datetime(2024, 1, 1),
            duracion=0,
            artistas=["Artista"],
            atributos_sonoros={"ritmo": 0.5},
            atributos_sentimentales={"felicidad": 0.5},
        )


def test_playlist_calcula_promedio_atributos_y_str():
    cancion1 = crear_cancion_prueba("c001", "A", datetime(2024, 1, 1), 0.2, 0.4)
    cancion2 = crear_cancion_prueba("c002", "B", datetime(2024, 1, 1), 0.8, 0.6)
    playlist = Playlist("Mi Playlist", datetime(2024, 1, 1), [cancion1, cancion2])

    assert playlist.obtener_nombre() == "Mi Playlist"
    assert playlist.obtener_atributos_sonoros() == {"ritmo": pytest.approx(0.5)}
    assert playlist.obtener_atributos_sentimentales() == {"felicidad": pytest.approx(0.5)}
    assert str(playlist) == "Playlist(nombre=Mi Playlist, canciones=2)"


def test_playlist_validacion_canciones_no_validas():
    with pytest.raises(ItemMusicalError):
        Playlist("Mi Playlist", datetime(2024, 1, 1), ["no una cancion"])


def test_artista_calcula_promedio_y_generos_por_defecto():
    cancion1 = crear_cancion_prueba("c001", "A", datetime(2024, 1, 1), 0.1, 0.2)
    cancion2 = crear_cancion_prueba("c002", "B", datetime(2024, 1, 1), 0.9, 0.8)
    artista = Artista("Artista Test", datetime(1990, 1, 1), [cancion1, cancion2])

    assert artista.obtener_nombre() == "Artista Test"
    assert artista.obtener_atributos_sonoros() == {"ritmo": pytest.approx(0.5)}
    assert artista.obtener_atributos_sentimentales() == {"felicidad": pytest.approx(0.5)}
    assert artista.generos == []
    assert str(artista) == "Artista(nombre=Artista Test, canciones=2)"


def test_artista_validacion_generos_no_validos():
    cancion = crear_cancion_prueba("c001", "Test", datetime(2024, 1, 1), 0.5, 0.5)
    with pytest.raises(ItemMusicalError):
        Artista("Artista", datetime(1990, 1, 1), [cancion], generos=["rock", 123])


def test_catalogo_buscar_cancion_por_id_encuentra():
    cancion = crear_cancion_prueba("c001", "Test", datetime(2024, 1, 1), 0.5, 0.5)
    catalogo = Catalogo(canciones=[cancion])
    assert catalogo.buscar_cancion_por_id("c001") == cancion


def test_catalogo_buscar_cancion_por_id_no_encuentra():
    catalogo = Catalogo(canciones=[])
    assert catalogo.buscar_cancion_por_id("c999") is None


def test_catalogo_buscar_cancion_por_id_identificador_invalido():
    catalogo = Catalogo(canciones=[])
    with pytest.raises(CatalogoError):
        catalogo.buscar_cancion_por_id(123)


def test_generar_catalogo_aleatorio_crea_catalogo_correctamente():
    catalogo = Catalogo.generar_catalogo_aleatorio(num_canciones=6, num_artistas=2, num_playlists=2)

    assert isinstance(catalogo, Catalogo)
    assert len(catalogo.canciones) == 6
    assert len(catalogo.artistas) == 2
    assert len(catalogo.playlists) == 2
    assert all(isinstance(c, Cancion) for c in catalogo.canciones)
    assert all(isinstance(a, Artista) for a in catalogo.artistas)
    assert all(isinstance(p, Playlist) for p in catalogo.playlists)


def test_generar_catalogo_aleatorio_parametros_invalidos_raise():
    with pytest.raises(CatalogoError):
        Catalogo.generar_catalogo_aleatorio(num_canciones=-1, num_artistas=2, num_playlists=2)
