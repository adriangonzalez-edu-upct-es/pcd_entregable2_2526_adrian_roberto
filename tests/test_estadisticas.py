from datetime import datetime
from statistics import StatisticsError
import pytest

from src.catalogo import Cancion
from src.estadisticas import (
    CalculadorBase,
    CalculadorSentimental,
    CalculadorSonoro,
    EstadisticaError,
    ManejadorEstadisticas,
)


def crear_cancion_prueba(identificador: str, nombre: str, ritmo: float, tono: float, felicidad: float) -> Cancion:
    return Cancion(
        identificador=identificador,
        nombre=nombre,
        fecha_creacion=datetime(2024, 1, 1),
        duracion=210.0,
        artistas=["Artista Test"],
        atributos_sonoros={"ritmo": ritmo, "tono": tono},
        atributos_sentimentales={"felicidad": felicidad},
    )


def test_manejador_estadisticas_valida_historial_no_es_lista():
    calculador = CalculadorSonoro()
    with pytest.raises(EstadisticaError):
        calculador.manejar("no es una lista", {})


def test_manejador_estadisticas_valida_estadisticas_no_es_dict():
    calculador = CalculadorSonoro()
    cancion = crear_cancion_prueba("c001", "Test", 0.2, 0.8, 0.6)
    with pytest.raises(EstadisticaError):
        calculador.manejar([cancion], "no es un dict")


def test_manejador_estadisticas_valida_historial_contiene_no_canciones():
    calculador = CalculadorSonoro()
    with pytest.raises(EstadisticaError):
        calculador.manejar(["no es cancion"], {})


def test_calculador_sonoro_procesa_una_cancion():
    cancion = crear_cancion_prueba("c001", "Test", 0.3, 0.7, 0.5)
    estadisticas = {}
    CalculadorSonoro().procesar([cancion], estadisticas)
    assert estadisticas == {
        "sonoros": {
            "media": {"ritmo": 0.3, "tono": 0.7},
            "desviacion": {"ritmo": 0.0, "tono": 0.0}
        }
    }


def test_calculador_sonoro_procesa_multiples_canciones():
    cancion1 = crear_cancion_prueba("c001", "A", 0.2, 0.8, 0.5)
    cancion2 = crear_cancion_prueba("c002", "B", 0.6, 0.4, 0.9)
    estadisticas = {}
    CalculadorSonoro().procesar([cancion1, cancion2], estadisticas)
    assert estadisticas["sonoros"]["media"]["ritmo"] == pytest.approx(0.4)
    assert estadisticas["sonoros"]["media"]["tono"] == pytest.approx(0.6)
    # Desviación estándar muestral: sqrt( sum((x-mean)^2) / (n-1) )
    # Para ritmo: ((0.2-0.4)^2 + (0.6-0.4)^2) / 1 = 0.08 / 1 = 0.08, sqrt(0.08) ≈ 0.2828
    assert estadisticas["sonoros"]["desviacion"]["ritmo"] == pytest.approx(0.282842712474619)
    assert estadisticas["sonoros"]["desviacion"]["tono"] == pytest.approx(0.282842712474619)


def test_calculador_sentimental_procesa_una_cancion():
    cancion = crear_cancion_prueba("c001", "Test", 0.3, 0.7, 0.5)
    estadisticas = {}
    CalculadorSentimental().procesar([cancion], estadisticas)
    assert estadisticas == {
        "sentimentales": {
            "media": {"felicidad": 0.5},
            "desviacion": {"felicidad": 0.0}
        }
    }


def test_calculador_sentimental_procesa_multiples_canciones():
    cancion1 = crear_cancion_prueba("c001", "A", 0.2, 0.8, 0.3)
    cancion2 = crear_cancion_prueba("c002", "B", 0.6, 0.4, 0.7)
    estadisticas = {}
    CalculadorSentimental().procesar([cancion1, cancion2], estadisticas)
    assert estadisticas["sentimentales"]["media"]["felicidad"] == pytest.approx(0.5)
    # Desviación muestral: ((0.3-0.5)^2 + (0.7-0.5)^2) / 1 = 0.08, sqrt(0.08) ≈ 0.2828
    assert estadisticas["sentimentales"]["desviacion"]["felicidad"] == pytest.approx(0.282842712474619)


def test_desviacion_segura_con_un_valor():
    assert CalculadorBase._desviacion_segura([0.5]) == 0.0


def test_desviacion_segura_con_multiples_valores():
    assert CalculadorBase._desviacion_segura([0.5, 0.5]) == 0.0
    # Para [0.0, 1.0]: mean 0.5, sum sq = 0.25 + 0.25 = 0.5, /1 = 0.5, sqrt(0.5) ≈ 0.7071
    assert CalculadorBase._desviacion_segura([0.0, 1.0]) == pytest.approx(0.7071067811865476)


def test_cadena_de_responsabilidades_sonoro_y_sentimental():
    sonoro = CalculadorSonoro()
    sentimental = CalculadorSentimental(sucesor=sonoro)
    cancion = crear_cancion_prueba("c001", "Test", 0.3, 0.7, 0.5)
    estadisticas = {}
    sentimental.manejar([cancion], estadisticas)
    assert "sonoros" in estadisticas
    assert "sentimentales" in estadisticas
    assert estadisticas["sonoros"]["media"] == {"ritmo": 0.3, "tono": 0.7}
    assert estadisticas["sentimentales"]["media"] == {"felicidad": 0.5}

