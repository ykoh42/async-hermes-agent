# Política de seguridad de Async Hermes Agent

Este documento cubre la biblioteca nativa asíncrona de este repositorio. El
producto Hermes Agent upstream tiene otra superficie y otra política de
seguridad.

## Comunicar una vulnerabilidad

Comunica las vulnerabilidades de forma privada mediante
[GitHub Security Advisories](https://github.com/ykoh42/async-hermes-agent/security/advisories/new).
No publiques credenciales, trayectorias privadas ni detalles de un exploit
funcional en un issue público.

Incluye el commit o versión afectada, versión de Python, sistema operativo,
configuración relevante, reproducción mínima y el límite de seguridad
atravesado. Este proyecto no ofrece un programa de recompensas.

## Versiones compatibles

Las correcciones de seguridad se aplican a la última versión y a `main`.
Confirma el problema contra una de ellas antes de informarlo.

## Superficie conservada

Este paquete es una biblioteca. Conserva:

- clientes de red para proveedores de modelos;
- herramientas de terminal, archivos, ejecución de código, navegador y memoria;
- conexiones MCP y subprocesos iniciados por MCP;
- descubrimiento de plugins y proveedores;
- descubrimiento y carga de habilidades;
- sesiones SQLite, memorias, checkpoints y trayectorias; y
- ejecución concurrente individual y por lotes.

No proporciona servidor HTTP, autenticación, gateway de mensajería, CLI/TUI,
planificador, dashboard ni aplicación de escritorio. La aplicación que integra
la biblioteca debe implementar autenticación y autorización de usuarios,
límites de solicitudes, aislamiento entre tenants, seguridad de transporte y
exposición segura de las respuestas del agente.

## Modelo de confianza

### El modelo no es de confianza

La salida del modelo y el contenido de páginas web, archivos, herramientas,
habilidades, plugins y servidores MCP pueden ser adversarios. Las instrucciones
del prompt, listas permitidas, aprobaciones y escáneres reducen accidentes; no
son mecanismos de contención.

### El sistema operativo es el límite de aislamiento

El backend local de terminal ejecuta con los permisos del proceso anfitrión.
Un backend de contenedor o remoto limita las operaciones que pasan por él, pero
no limita automáticamente plugins Python, clientes MCP, clientes de proveedor
ni otro código dentro del proceso.

Para entradas hostiles o servicios compartidos/de producción, ejecuta toda la
aplicación y sus procesos hijos dentro de un sandbox del sistema operativo con
políticas explícitas de archivos, procesos y red. Expón solo los directorios y
credenciales necesarios.

### Plugins, habilidades y MCP son extensiones de confianza

Los plugins ejecutan Python dentro del proceso del agente y heredan sus
privilegios. Revisa el código y las dependencias antes de instalarlos.

Las habilidades pueden dirigir al modelo para ejecutar comandos o acceder a
sistemas externos. Revisa el directorio completo, incluidos scripts y recursos,
no solo `SKILL.md`.

Los servidores MCP son programas o servicios de confianza independientes.
Pueden recibir datos de conversación y devolver contenido controlado por un
atacante. Fija y revisa los paquetes locales, limita su entorno y usa transporte
cifrado y autenticado para servidores remotos.

### Credenciales

Usa `.env` solo para secretos y `config.yaml` para comportamiento. No incluyas
secretos en prompts, habilidades, trayectorias, código fuente ni logs. Concede
a cada proveedor, herramienta, plugin y servidor MCP el mínimo alcance posible.

No se usan hilos de compatibilidad para ocultar implementaciones síncronas. Es
una garantía de corrección asíncrona, no un límite de seguridad.

### Datos almacenados

Sesiones, memorias, checkpoints y trayectorias pueden contener prompts,
razonamiento, argumentos y observaciones de herramientas, archivos y respuestas
del modelo. Trata los directorios de Hermes y de trayectorias como datos
sensibles. La aplicación integradora debe aplicar permisos, retención, cifrado
y eliminación apropiados.

## Informes dentro de alcance

Por ejemplo:

- filtración de credenciales causada por esta biblioteca;
- validación de autorización o rutas que escape del límite documentado de un
  backend seleccionado;
- deserialización o construcción insegura de comandos o SQL;
- exposición de estado entre sesiones causada por la biblioteca;
- descubrimiento MCP o de plugins que cargue un destino distinto al elegido;
- errores de cancelación o concurrencia que expongan otra conversación; y
- vulnerabilidades de dependencias alcanzables desde una ruta conservada.

## Normalmente fuera de alcance

No constituyen por sí solos una vulnerabilidad de la biblioteca:

- que un modelo siga instrucciones maliciosas sin atravesar un límite del
  sistema operativo o de autorización de la aplicación;
- comandos aprobados intencionadamente por la aplicación integradora;
- un plugin, habilidad o servidor MCP malicioso instalado por el operador;
- credenciales o directorios expuestos deliberadamente al proceso; y
- vulnerabilidades que solo existan en superficies eliminadas de upstream.

Estos casos todavía pueden justificar un issue normal de hardening si se pueden
explicar sin publicar detalles sensibles.

## Dependencias y divulgación

Las dependencias se fijan y revisan deliberadamente. Un informe debe identificar
la ruta conservada alcanzable, no solo una advertencia de paquete. Cuando exista
una corrección, los mantenedores publicarán una versión actualizada y darán
crédito a quien desee reconocimiento.
