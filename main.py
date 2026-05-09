
import asyncio
from datetime import datetime
from typing import Optional

from src.catalogo import Catalogo, Cancion
from src.recomendador import SistemaRecomendacionUsuario
from src.estrategias import BusquedaAlfabetica, BusquedaTemporal, BusquedaAleatoria


class TerminalApp:
    def __init__(self) -> None:
        self.catalogo: Optional[Catalogo] = None
        self.usuario: Optional[SistemaRecomendacionUsuario] = None
        self.estrategias = {
            "1": ("Alfabética", BusquedaAlfabetica),
            "2": ("Temporal", BusquedaTemporal),
            "3": ("Aleatoria", BusquedaAleatoria),
        }
        self.tipos_recomendacion = {"cancion"}

    def mostrar_menu(self) -> None:
        print("\n" + "="*60)
        print("🎵 RECOMENDADOR MUSICAL - MENÚ PRINCIPAL 🎵")
        print("="*60)
        print("1. Generar catálogo aleatorio")
        print("2. Mostrar catálogo actual")
        print("3. Escuchar canción")
        print("4. Ver historial de sesión")
        print("5. Ver estadísticas de sesión")
        print("6. Obtener recomendaciones")
        print("7. Cambiar estrategia de búsqueda")
        print("8. Configurar tipos de recomendación")
        print("9. Buscar canción por ID")
        print("0. Salir")
        print("="*60)

    def ejecutar_opcion(self, opcion: str) -> bool:
        opciones = {
            "1": self.generar_catalogo,
            "2": self.mostrar_catalogo,
            "3": self.escuchar_cancion,
            "4": self.ver_historial,
            "5": self.ver_estadisticas,
            "6": self.obtener_recomendaciones,
            "7": self.cambiar_estrategia,
            "8": self.configurar_tipos,
            "9": self.buscar_cancion,
            "0": lambda: True,  # Salir
        }

        accion = opciones.get(opcion)
        if accion:
            try:
                if opcion == "0":
                    print("¡Hasta luego! 🎶")
                    return True
                accion()
            except Exception as error:
                print(f"❌ Error: {error}")
        else:
            print("❌ Opción no válida. Intente de nuevo.")
        return False

    def generar_catalogo(self) -> None:
        try:
            num_canciones = int(input("Número de canciones (predeterminado 24): ") or "24")
            num_artistas = int(input("Número de artistas (predeterminado 4): ") or "4")
            num_playlists = int(input("Número de playlists (predeterminado 4): ") or "4")

            self.catalogo = Catalogo.generar_catalogo_aleatorio(
                num_canciones=num_canciones,
                num_artistas=num_artistas,
                num_playlists=num_playlists
            )
            self.usuario = SistemaRecomendacionUsuario(f"usuario-terminal-{int(datetime.now().timestamp())}")
            self.usuario.set_estrategia(BusquedaAlfabetica())

            print("✅ Catálogo generado exitosamente!")
            print(f"   Canciones: {len(self.catalogo.canciones)}")
            print(f"   Artistas: {len(self.catalogo.artistas)}")
            print(f"   Playlists: {len(self.catalogo.playlists)}")

        except ValueError:
            print("❌ Los valores deben ser números enteros positivos.")
        except Exception as error:
            print(f"❌ Error al generar catálogo: {error}")

    def mostrar_catalogo(self) -> None:
        if not self.catalogo:
            print("❌ No hay catálogo generado. Use la opción 1 primero.")
            return

        print("\n🎵 CATÁLOGO ACTUAL 🎵")
        print(f"Canciones ({len(self.catalogo.canciones)}):")
        for i, cancion in enumerate(self.catalogo.canciones[:10], 1):  # Mostrar primeras 10
            print(f"  {i}. {cancion.identificador} - {cancion.nombre}")
        if len(self.catalogo.canciones) > 10:
            print(f"  ... y {len(self.catalogo.canciones) - 10} más")

        print(f"\nArtistas ({len(self.catalogo.artistas)}):")
        for artista in self.catalogo.artistas:
            print(f"  - {artista.nombre} ({len(artista.canciones)} canciones)")

        print(f"\nPlaylists ({len(self.catalogo.playlists)}):")
        for playlist in self.catalogo.playlists:
            print(f"  - {playlist.nombre} ({len(playlist.canciones)} canciones)")

    def escuchar_cancion(self) -> None:
        if not self.catalogo or not self.usuario:
            print("❌ Genere un catálogo primero (opción 1).")
            return

        try:
            id_cancion = input("ID de la canción a escuchar: ").strip()
            cancion = self.catalogo.buscar_cancion_por_id(id_cancion)
            if not cancion:
                print("❌ Canción no encontrada.")
                return

            asyncio.run(self.usuario.escuchar_cancion_async(cancion, datetime.now()))
            print(f"✅ Canción escuchada: {cancion.nombre} ({cancion.identificador})")

        except Exception as error:
            print(f"❌ Error al escuchar canción: {error}")

    def ver_historial(self) -> None:
        if not self.usuario:
            print("❌ No hay usuario activo.")
            return

        historial = self.usuario.historial_sesion
        if not historial:
            print("📝 Historial vacío. Escuche algunas canciones primero.")
            return

        print("\n📝 HISTORIAL DE SESIÓN 📝")
        for i, evento in enumerate(historial[-10:], 1):  # Últimas 10
            cancion = evento["cancion"]
            fecha = evento["fecha_hora"]
            print(f"  {i}. {cancion.nombre} - {fecha.strftime('%Y-%m-%d %H:%M:%S')}")

        if len(historial) > 10:
            print(f"  ... y {len(historial) - 10} más entradas")

    def ver_estadisticas(self) -> None:
        if not self.usuario:
            print("❌ No hay usuario activo.")
            return

        estadisticas = self.usuario.estadisticas
        print("\n📊 ESTADÍSTICAS DE SESIÓN 📊")
        print("Sonoros:")
        print(f"  Media: {estadisticas['sonoros']['media']}")
        print(f"  Desviación: {estadisticas['sonoros']['desviacion']}")
        print("Sentimentales:")
        print(f"  Media: {estadisticas['sentimentales']['media']}")
        print(f"  Desviación: {estadisticas['sentimentales']['desviacion']}")

    def obtener_recomendaciones(self) -> None:
        if not self.catalogo or not self.usuario:
            print("❌ Genere un catálogo y escuche canciones primero.")
            return

        try:
            recomendaciones = asyncio.run(self.usuario.obtener_recomendaciones_async(self.catalogo))
            if not recomendaciones:
                print("🤔 No se encontraron recomendaciones. Intente escuchar más canciones.")
                return

            print("\n🎯 RECOMENDACIONES 🎯")
            for item in recomendaciones:
                sujeto = item["item"]
                tipo = item["tipo"]
                print(f"  {tipo.capitalize()}: {sujeto.obtener_nombre()} ({sujeto.obtener_fecha().date()})")

        except Exception as error:
            print(f"❌ Error al obtener recomendaciones: {error}")

    def cambiar_estrategia(self) -> None:
        if not self.usuario:
            print("❌ No hay usuario activo.")
            return

        print("\n🔍 ESTRATEGIAS DISPONIBLES 🔍")
        for key, (nombre, _) in self.estrategias.items():
            print(f"  {key}. {nombre}")

        opcion = input("Seleccione estrategia (1-3): ").strip()
        estrategia_info = self.estrategias.get(opcion)
        if estrategia_info:
            nombre, clase = estrategia_info
            self.usuario.set_estrategia(clase())
            print(f"✅ Estrategia cambiada a: {nombre}")
        else:
            print("❌ Opción no válida.")

    def configurar_tipos(self) -> None:
        if not self.usuario:
            print("❌ No hay usuario activo.")
            return

        print("\n⚙️ CONFIGURACIÓN DE TIPOS DE RECOMENDACIÓN ⚙️")
        print("Tipos disponibles: cancion, artista, playlist")
        tipos_input = input("Ingrese tipos separados por coma (ej: cancion,artista): ").strip().lower()
        tipos = [t.strip() for t in tipos_input.split(",") if t.strip()]

        self.usuario.tipos_recomendacion = set()  # Reset {"cancion"}
        for tipo in tipos:
            if tipo in {"cancion", "artista", "playlist"}:
                self.usuario.agregar_tipo_recomendacion(tipo)
                print(f"✅ Tipo agregado: {tipo}")
            else:
                print(f"❌ Tipo desconocido: {tipo}")

    def buscar_cancion(self) -> None:
        if not self.catalogo:
            print("❌ No hay catálogo generado.")
            return

        id_cancion = input("ID de la canción a buscar: ").strip()
        try:
            cancion = self.catalogo.buscar_cancion_por_id(id_cancion)
            if cancion:
                print("\n🔍 CANCIÓN ENCONTRADA 🔍")
                print(f"  ID: {cancion.identificador}")
                print(f"  Nombre: {cancion.nombre}")
                print(f"  Fecha: {cancion.fecha_creacion.date()}")
                print(f"  Duración: {cancion.duracion:.1f} segundos")
                print(f"  Artistas: {', '.join(cancion.artistas)}")
                print(f"  Atributos sonoros: {cancion.obtener_atributos_sonoros()}")
                print(f"  Atributos sentimentales: {cancion.obtener_atributos_sentimentales()}")
            else:
                print("❌ Canción no encontrada.")
        except Exception as error:
            print(f"❌ Error en búsqueda: {error}")


def main() -> None:
    app = TerminalApp()
    print("🎵 Bienvenido al Recomendador Musical por Terminal 🎵")
    print("Use los números del menú para navegar.")

    while True:
        app.mostrar_menu()
        opcion = input("Seleccione una opción: ").strip()
        if app.ejecutar_opcion(opcion):
            break


if __name__ == "__main__":
    main()
