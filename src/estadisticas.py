from abc import ABC, abstractmethod
from functools import reduce
from statistics import mean, pstdev, StatisticsError

from catalogo import Cancion


class EstadisticaError(Exception):
    """Error personalizado para la lógica de cálculo estadístico."""
    pass


class ManejadorEstadisticas(ABC):
    """Base para una cadena de responsabilidades de cálculo de estadísticas."""

    def __init__(self, sucesor: "ManejadorEstadisticas" | None = None):
        self.sucesor = sucesor

    def manejar(self, historial: list[Cancion], estadisticas: dict[str, dict[str, dict[str, float]]]) -> None:
        # Validaciones básicas de tipos para evitar cálculos inválidos.
        if not isinstance(historial, list):
            raise EstadisticaError("El historial debe ser una lista de Cancion.")
        if not isinstance(estadisticas, dict):
            raise EstadisticaError("Las estadísticas deben ser un diccionario.")
        if any(not isinstance(cancion, Cancion) for cancion in historial):
            raise EstadisticaError("El historial sólo puede contener instancias de Cancion.")

        try:
            self.procesar(historial, estadisticas)
            if self.sucesor:
                self.sucesor.manejar(historial, estadisticas)
        except Exception as error:
            raise EstadisticaError("Error al procesar estadísticas.") from error

    @abstractmethod
    def procesar(self, historial: list[Cancion], estadisticas: dict[str, dict[str, dict[str, float]]]) -> None:
        """Procesa el historial y actualiza el diccionario de estadísticas."""
        pass


class CalculadorBase(ManejadorEstadisticas):
    """Calculador común para estadísticas de distintas categorías."""

    def _actualizar_estadisticas(self, historial: list[Cancion], estadisticas: dict[str, dict[str, dict[str, float]]], tipo: str) -> None:
        # Asegura que exista el tipo de estadística antes de escribir en él.
        estadisticas.setdefault(tipo, {})
        medias, desviaciones = self._obtener_media_y_desviacion(historial, tipo)
        estadisticas[tipo]["media"] = medias
        estadisticas[tipo]["desviacion"] = desviaciones

    def _obtener_media_y_desviacion(self, historial: list[Cancion], tipo: str) -> tuple[dict[str, float], dict[str, float]]:
        """Agrupa los valores de cada atributo por nombre y luego calcula
        la media y desviación estándar para cada atributo."""
        valores_por_atributo: dict[str, list[float]] = reduce(
            lambda acumulador, cancion: self._agregar_atributos(acumulador, cancion, tipo),
            historial,
            {},)
        medias = dict(
            map(
                lambda elemento: (elemento[0], mean(elemento[1])),
                valores_por_atributo.items(),))
        desviaciones = dict(
            map(
                lambda elemento: (elemento[0], self._desviacion_segura(elemento[1])),
                valores_por_atributo.items(),))

        return medias, desviaciones

    def _agregar_atributos(self, acumulador: dict[str, list[float]], cancion: Cancion, tipo: str) -> dict[str, list[float]]:
        # Selecciona atributos según el tipo y los acumula para el cálculo.
        atributos = (
            cancion.obtener_atributos_sonoros()
            if tipo == "sonoros"
            else cancion.obtener_atributos_sentimentales()
        )
        for nombre, valor in atributos.items():
            acumulador.setdefault(nombre, []).append(valor)
        return acumulador

    @staticmethod
    def _desviacion_segura(valores: list[float]) -> float:
        # Devuelve desviación 0 si no hay suficientes datos para calcularla.
        try:
            return pstdev(valores)
        except StatisticsError:
            return 0.0


class CalculadorSonoro(CalculadorBase):
    """Calcula estadísticas para atributos sonoros de las canciones."""

    def procesar(self, historial: list[Cancion], estadisticas: dict[str, dict[str, dict[str, float]]]) -> None:
        self._actualizar_estadisticas(historial, estadisticas, "sonoros")


class CalculadorSentimental(CalculadorBase):
    """Calcula estadísticas para atributos sentimentales de las canciones."""

    def procesar(self, historial: list[Cancion], estadisticas: dict[str, dict[str, dict[str, float]]]) -> None:
        self._actualizar_estadisticas(historial, estadisticas, "sentimentales")
