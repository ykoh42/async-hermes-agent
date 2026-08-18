# Async Hermes Agent

Distribución nativa asíncrona y orientada a biblioteca de
[NousResearch/hermes-agent](https://github.com/NousResearch/hermes-agent), basada
en la etiqueta upstream `v2026.8.16.2` (versión del paquete Python `0.20.3`).

Este repositorio conserva el bucle del agente Hermes, los proveedores de
modelos, la ejecución de herramientas, MCP, habilidades, memoria y sesiones
persistentes, generación de trayectorias, runner y batch runner. La CLI/TUI,
los puentes de mensajería, el planificador, el dashboard y una aplicación
FastAPI quedan intencionadamente fuera de este paquete.

La API pública central mantiene los nombres y rutas de módulos de upstream.
Normalmente, una integración existente solo necesita añadir `await`:

```python
import asyncio
import os

from run_agent import AIAgent


async def main():
    async with AIAgent(
        provider="openrouter",
        model="openrouter/auto",
        api_key=os.environ["OPENROUTER_API_KEY"],
    ) as agent:
        result = await agent.run_conversation("Investiga este repositorio")
        print(result["final_response"])


asyncio.run(main())
```

Dentro de una función asíncrona, la interfaz compacta que devuelve texto y el
ciclo de vida explícito son:

```python
agent = AIAgent(provider="openrouter", model="openrouter/auto")
try:
    answer = await agent.chat("Resume el resultado")
finally:
    await agent.close()
```

`AIAgent.__init__()` solo construye estado. La configuración, los clientes de
proveedor, el almacenamiento de sesiones y las conexiones MCP se inicializan
de forma diferida en el primer límite esperado. Los turnos de una misma
instancia de `AIAgent` se serializan; instancias distintas pueden ejecutarse en
paralelo.

## Instalación

Se admite Python 3.11 a 3.13.

```bash
uv pip install "async-hermes-agent==0.20.3.1"
```

Las versiones se publican en PyPI mediante OIDC Trusted Publishing de GitHub.
La misma rueda, distribución fuente y sumas de comprobación verificadas se
adjuntan a la versión correspondiente de GitHub.

La versión tiene cuatro segmentos numéricos: `0.20.3.1` indica la versión
Python upstream `0.20.3` y la revisión `1` de esta distribución asíncrona. Las
versiones exclusivas del fork incrementan el cuarto segmento; al portar una
nueva versión upstream, los tres primeros segmentos se actualizan y la revisión
vuelve a `1`.

La versión anterior de GitHub, `0.20.4`, usaba el esquema independiente antiguo
según las reglas de versiones de Python. Si
se instaló desde esa etiqueta Git, la migración requiere una reinstalación
explícita una sola vez:

```bash
uv pip install --reinstall "async-hermes-agent==0.20.3.1"
```

Para desarrollo:

```bash
git clone https://github.com/ykoh42/async-hermes-agent.git
cd async-hermes-agent
uv sync --extra dev
```

Las dependencias específicas de proveedor siguen siendo opcionales, por
ejemplo:

```bash
uv sync --extra anthropic
uv sync --extra vertex
uv sync --extra azure-identity
```

## Habilidades, MCP y memoria

Las habilidades siguen la estructura existente de Hermes. `HERMES_HOME` usa
`~/.hermes` por defecto. Coloca cada habilidad activa en:

```text
$HERMES_HOME/skills/<nombre-habilidad>/SKILL.md
```

Cada `SKILL.md` es un documento normal de Hermes con frontmatter YAML:

```markdown
---
name: code-review
description: Revisa un cambio de código antes de integrarlo.
---

# Revisión de código

Lee el cambio, ejecuta sus pruebas e informa primero de los problemas de
corrección.
```

Hermes upstream instala sus habilidades incluidas mediante el instalador del
producto. Esta biblioteca no incluye ese instalador, por lo que los usuarios
de Git o wheel deben añadir los directorios de habilidades explícitamente o
indicar directorios compartidos en `config.yaml`:

```yaml
skills:
  external_dirs:
    - ~/.agents/skills
    - /shared/team-skills
```

Las herramientas `skills_list` y `skill_view` descubren tanto los directorios
locales como los externos configurados. El contenido de una habilidad queda
fuera del esquema de herramientas del modelo hasta que este la selecciona y la
lee.

Los servidores MCP se configuran bajo `mcp_servers` en
`$HERMES_HOME/config.yaml`:

```yaml
mcp_servers:
  filesystem:
    command: npx
    args: ["-y", "@modelcontextprotocol/server-filesystem", "/workspace"]
```

El primer límite esperado del agente descubre los servidores configurados y
registra sus herramientas en el toolset del servidor. Los subprocesos y
sesiones cliente MCP se cierran mediante `await agent.close()` o el gestor de
contexto asíncrono.

La memoria basada en archivos y el perfil de usuario también conservan el home
normal de Hermes en `~/.hermes`. Para un agente con memoria, habilita el toolset
`memory` y sus ajustes correspondientes en `config.yaml`.

## Entrenamiento y trayectorias

Usa `save_trajectories=True` en `AIAgent` para conversaciones individuales. La
secuencia guardada conserva razonamiento, llamadas a herramientas,
observaciones y respuesta final para fine-tuning con pensamiento intercalado.
Las muestras completadas se añaden a `trajectory_samples.jsonl` en el
directorio de trabajo del proceso.

Para datasets, importa `BatchRunner` desde el módulo conservado
`batch_runner.py` y espera su método `run()`. Mantiene concurrencia limitada,
checkpoints, reanudación y salida JSONL. `trajectory_compressor.py` sigue
disponible para procesar las trayectorias generadas.

```python
import asyncio
import os

from batch_runner import BatchRunner


async def main():
    runner = BatchRunner(
        dataset_file="prompts.jsonl",
        batch_size=8,
        run_name="tool-training",
        distribution="terminal_only",
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
        model="openai/gpt-oss-20b:free",
        num_workers=4,
        reasoning_config={"enabled": True, "effort": "low"},
    )
    await runner.run(resume=True)


asyncio.run(main())
```

Cada línea de entrada debe ser JSON con un campo `prompt`. Los resultados se
escriben bajo `data/<run_name>/`: fragmentos JSONL por lote,
`trajectories.jsonl`, `checkpoint.json` y `statistics.json`.

## Integración en servicios

No se incluye ningún framework web. El servicio debe controlar su ciclo de
vida HTTP y esperar directamente a la biblioteca:

```python
from contextlib import asynccontextmanager

from fastapi import FastAPI
from run_agent import AIAgent


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with AIAgent(provider="openrouter", model="openrouter/auto") as agent:
        app.state.agent = agent
        yield


app = FastAPI(lifespan=lifespan)


@app.post("/chat")
async def chat(message: str):
    return await app.state.agent.run_conversation(message)
```

El paquete no usa `asyncio.to_thread()`, `run_in_executor()`,
`run_until_complete()` ni `.result()` bloqueante en la ruta activa conservada.
Los proveedores opcionales sin transporte asíncrono nativo fallan al inicio en
lugar de ejecutar silenciosamente trabajo síncrono en un hilo.

## Verificación

```bash
uv run pytest -q
uv run ruff check agent tools hermes_cli plugins providers \
  run_agent.py model_tools.py batch_runner.py hermes_state.py \
  trajectory_compressor.py
uv build
```

## Contribuciones y seguridad

Consulta [CONTRIBUTING.es.md](CONTRIBUTING.es.md) antes de enviar cambios y
[SECURITY.es.md](SECURITY.es.md) para comunicar vulnerabilidades de forma
privada.

## Relación con upstream

El repositorio conserva los nombres originales de archivos y funciones
centrales para que futuras importaciones desde upstream sean revisables. Es una
distribución asíncrona divergente, no una afirmación de que los cambios puedan
integrarse directamente en el producto síncrono upstream.

Hermes Agent fue creado por [Nous Research](https://nousresearch.com). Esta
distribución conserva la licencia MIT de upstream; consulta [LICENSE](LICENSE).
