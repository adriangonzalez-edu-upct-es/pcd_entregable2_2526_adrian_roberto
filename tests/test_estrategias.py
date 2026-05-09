from datetime import datetime
import pytest

from src.catalogo import Cancion
from src.estrategias import (
    BusquedaAlfabetica,
    BusquedaAleatoria,
    BusquedaTemporal,
    EstrategiaBusqueda,
    EstrategiaError,
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


def test_validar_entrada_lista_no_es_lista():
    estrategia = BusquedaAlfabetica()
    with pytest.raises(EstrategiaError):
        estrategia.buscar("no lista", {})


def test_validar_entrada_lista_vacia():
    estrategia = BusquedaAlfabetica()
    with pytest.raises(EstrategiaError):
        estrategia.buscar([], {})


def test_validar_entrada_estadisticas_no_dict():
    estrategia = BusquedaAlfabetica()
    cancion = crear_cancion_prueba("c001", "Test", datetime(2024, 1, 1), 0.5, 0.5)
    with pytest.raises(EstrategiaError):
        estrategia.buscar([cancion], "no dict")


def test_busqueda_alfabetica_encuentra_primera_coincidente():
    cancion1 = crear_cancion_prueba("c001", "Z Song", datetime(2024, 1, 1), 0.5, 0.5)
    cancion2 = crear_cancion_prueba("c002", "A Song", datetime(2024, 1, 1), 0.5, 0.5)
    lista = [cancion1, cancion2]
    estadisticas = {"sonoros": {"media": {"ritmo": 0.5}}, "sentimentales": {"media": {"felicidad": 0.5}}}
    estrategia = BusquedaAlfabetica()
    resultado = estrategia.buscar(lista, estadisticas)
    assert resultado == cancion2  # A Song primero alfabéticamente


def test_busqueda_alfabetica_no_encuentra_coincidente():
    cancion = crear_cancion_prueba("c001", "Test", datetime(2024, 1, 1), 0.1, 0.1)
    lista = [cancion]
    estadisticas = {"sonoros": {"media": {"ritmo": 0.9}}, "sentimentales": {"media": {"felicidad": 0.9}}}
    estrategia = BusquedaAlfabetica()
    with pytest.raises(EstrategiaError):
        estrategia.buscar(lista, estadisticas)


def test_busqueda_temporal_encuentra_mas_reciente_coincidente():
    cancion1 = crear_cancion_prueba("c001", "Old", datetime(2023, 1, 1), 0.5, 0.5)
    cancion2 = crear_cancion_prueba("c002", "New", datetime(2024, 1, 1), 0.5, 0.5)
    lista = [cancion1, cancion2]
    estadisticas = {"sonoros": {"media": {"ritmo": 0.5}}, "sentimentales": {"media": {"felicidad": 0.5}}}
    estrategia = BusquedaTemporal()
    resultado = estrategia.buscar(lista, estadisticas)
    assert resultado == cancion2  # Más reciente


def test_busqueda_temporal_no_encuentra_coincidente():
    cancion = crear_cancion_prueba("c001", "Test", datetime(2024, 1, 1), 0.1, 0.1)
    lista = [cancion]
    estadisticas = {"sonoros": {"media": {"ritmo": 0.9}}, "sentimentales": {"media": {"felicidad": 0.9}}}
    estrategia = BusquedaTemporal()
    with pytest.raises(EstrategiaError):
        estrategia.buscar(lista, estadisticas)


def test_busqueda_aleatoria_encuentra_coincidente():
    cancion1 = crear_cancion_prueba("c001", "Song1", datetime(2024, 1, 1), 0.5, 0.5)
    cancion2 = crear_cancion_prueba("c002", "Song2", datetime(2024, 1, 1), 0.5, 0.5)
    lista = [cancion1, cancion2]
    estadisticas = {"sonoros": {"media": {"ritmo": 0.5}}, "sentimentales": {"media": {"felicidad": 0.5}}}
    estrategia = BusquedaAleatoria()
    resultado = estrategia.buscar(lista, estadisticas)
    assert resultado in [cancion1, cancion2]


def test_busqueda_aleatoria_no_encuentra_coincidente():
    cancion = crear_cancion_prueba("c001", "Test", datetime(2024, 1, 1), 0.1, 0.1)
    lista = [cancion]
    estadisticas = {"sonoros": {"media": {"ritmo": 0.9}}, "sentimentales": {"media": {"felicidad": 0.9}}}
    estrategia = BusquedaAleatoria()
    with pytest.raises(EstrategiaError):
        estrategia.buscar(lista, estadisticas)


def test_busqueda_aleatoria_con_un_solo_candidato():
    cancion = crear_cancion_prueba("c001", "Test", datetime(2024, 1, 1), 0.5, 0.5)
    lista = [cancion]
    estadisticas = {"sonoros": {"media": {"ritmo": 0.5}}, "sentimentales": {"media": {"felicidad": 0.5}}}
    estrategia = BusquedaAleatoria()
    resultado = estrategia.buscar(lista, estadisticas)
    assert resultado == cancion