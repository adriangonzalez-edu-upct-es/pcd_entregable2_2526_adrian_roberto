from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import List

from catalogo import ItemMusical


class EstrategiaError(ValueError):
    pass


class EstrategiaBusqueda(ABC):
    @abstractmethod
    def buscar(
        self, lista_items: List[ItemMusical], estadisticas: dict
    ) -> ItemMusical:
        raise NotImplementedError

    def _validar_entrada(self, lista_items: List[ItemMusical], estadisticas: dict) -> None:
        if not isinstance(lista_items, list):
            raise EstrategiaError("La lista de búsqueda debe ser una lista de elementos.")
        if not lista_items:
            raise EstrategiaError("La lista de búsqueda no puede estar vacía.")
        if not isinstance(estadisticas, dict):
            raise EstrategiaError("Las estadísticas deben ser un diccionario.")


class BusquedaAlfabetica(EstrategiaBusqueda):
    def buscar(self, lista_items: List[ItemMusical], estadisticas: dict) -> ItemMusical:
        """Busca el primer elemento que coincida con las estadísticas en orden alfabético."""
        self._validar_entrada(lista_items, estadisticas)
        # Con next obtenemos la primera coincidencia, si no hay coincidencias se devuelve None
        candidato = next(
            filter(
                lambda item: item.coincide_con_estadisticas(estadisticas),
                # Ordenamos la lista alfabéticamente por el nombre del elemento
                sorted(lista_items, key=lambda elemento: elemento.obtener_nombre()),
            ),
            None,
        )
        if candidato is None:
            raise EstrategiaError("No se encontró ningún elemento que coincida en orden alfabético.")
        return candidato

class BusquedaTemporal(EstrategiaBusqueda):
    """Busca el primer elemento que coincida con las estadísticas en orden temporal (más reciente primero)."""
    def buscar(self, lista_items: List[ItemMusical], estadisticas: dict) -> ItemMusical:
        self._validar_entrada(lista_items, estadisticas)
        # Con next obtenemos la primera coincidencia, si no hay coincidencias se devuelve None
        candidato = next(
            filter(
                lambda item: item.coincide_con_estadisticas(estadisticas),
                # Ordenamos la lista por fecha de lanzamiento, del más reciente al más antiguo
                sorted(
                    lista_items,
                    key=lambda elemento: elemento.obtener_fecha(),
                    reverse=True,
                ),
            ),
            None,
        )
        if candidato is None:
            raise EstrategiaError("No se encontró ningún elemento que coincida en orden temporal.")
        return candidato


class BusquedaAleatoria(EstrategiaBusqueda):
    """Busca un elemento aleatorio que coincida con las estadísticas."""
    def buscar(self, lista_items: List[ItemMusical], estadisticas: dict) -> ItemMusical:
        self._validar_entrada(lista_items, estadisticas)
        
        candidatos = list(
            filter(
                lambda item: item.coincide_con_estadisticas(estadisticas),
                lista_items,
            )
        )
        if not candidatos:
            raise EstrategiaError("No se encontró ningún elemento aleatorio que coincida con las estadísticas.")
        try:
            # Devuelve un elemento aleatorio de la lista de candidatos
            return random.choice(candidatos)
        except IndexError as error:
            raise EstrategiaError("Error al seleccionar un candidato aleatorio.") from error