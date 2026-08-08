# Contribuir a Async Hermes Agent

Gracias por contribuir. Este proyecto conserva el comportamiento incluido de
[Hermes Agent](https://github.com/NousResearch/hermes-agent) y lo ofrece como
una biblioteca nativa asíncrona.

## Alcance del proyecto

Los cambios deben mejorar una superficie conservada: el bucle del agente, los
proveedores, las herramientas, MCP, las habilidades, la memoria, las sesiones,
las trayectorias o la ejecución por lotes. El repositorio no incluye CLI/TUI,
gateway de mensajería, planificador cron, dashboard, aplicación de escritorio
ni framework de servicio web.

No añadas una superficie eliminada como infraestructura incidental. Una
restauración deliberada debe incluir su ruta de ejecución completa,
dependencias, pruebas y documentación en una propuesta acotada.

## Configuración de desarrollo

Se admite Python 3.11 a 3.13. Instala el proyecto y las dependencias de
desarrollo con [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/ykoh42/async-hermes-agent.git
cd async-hermes-agent
uv sync --extra dev
```

Ejecuta las comprobaciones estándar:

```bash
uv run pytest -q
uv run ruff check .
uv build
```

Las pruebas de algunos proveedores pueden necesitar su dependencia opcional.
No confirmes claves API, archivos `.env`, trayectorias generadas ni resultados
de pruebas.

## Reglas de asincronía nativa

- Conserva las rutas de módulos, nombres de funciones, argumentos y formatos
  de retorno públicos.
- Convierte la implementación en el mismo lugar; no añadas alias `arun_*`,
  wrappers síncronos ni implementaciones sync/async duplicadas.
- No uses `asyncio.to_thread()`, `run_in_executor()`, `run_until_complete()` ni
  `.result()` bloqueante en el runtime conservado.
- Usa transportes y E/S nativamente asíncronos. Si un proveedor o herramienta
  no tiene implementación asíncrona nativa, falla de forma explícita.
- Mantén síncronas las funciones puras que solo usan CPU.
- Conserva la cancelación y cierra de forma determinista los recursos propios.

## Reglas de conservación del comportamiento

- Mantén estable el prefijo del prompt de sistema durante una conversación.
- Conserva la alternancia de roles y el orden de llamadas a herramientas.
- Conserva el orden de trayectorias: razonamiento, llamada, observación y
  respuesta final.
- Conserva sesiones, checkpoints, presupuestos, interrupciones, guardrails,
  compresión de contexto y reanudación de lotes.
- Las pruebas deben verificar comportamiento e invariantes, no valores
  incidentales.

## Cambios de código

Prefiere una modificación quirúrgica dentro del archivo y la función originales.
Evita mover archivos, crear wrappers de paso, abstracciones especulativas,
shims de compatibilidad o limpiezas no relacionadas. Esto mantiene revisables
los futuros diffs con upstream.

Una capacidad nueva para el modelo debería ser normalmente una habilidad, un
plugin o un servidor MCP, no otra herramienta central permanente. Añade una
dependencia solo si la biblioteca estándar y las dependencias actuales no
permiten la implementación asíncrona requerida; las dependencias específicas
de proveedor pertenecen a extras opcionales.

## Pruebas

Añade cobertura enfocada para cada cambio de comportamiento. Los cambios de
E/S, configuración, sesiones, MCP, habilidades, memoria o trayectorias deberían
ejercitar la ruta real con archivos temporales o bases SQLite cuando sea
posible.

Según corresponda, comprueba la respuesta del event loop, cancelación y
timeouts, serialización de turnos del mismo agente, concurrencia entre agentes,
fugas de tareas o procesos, propagación de errores y orden estable de prompts,
mensajes y trayectorias.

Las pruebas con proveedores reales son opcionales y deben depender
explícitamente de credenciales. La suite normal no debe consumir APIs de pago.

## Pull requests

Un pull request debe explicar:

1. el comportamiento conservado o error que corrige;
2. por qué respeta los invariantes asíncronos;
3. cómo se verificó la paridad con upstream; y
4. qué pruebas y builds se ejecutaron.

Separa los refactors de los cambios de comportamiento cuando sea posible. Para
una migración de upstream, identifica los commits o el diff de origen e incluye
pruebas de paridad para su intención.

Al contribuir, aceptas que tu contribución se publique bajo la
[Licencia MIT](LICENSE).
