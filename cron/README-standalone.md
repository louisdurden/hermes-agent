# Cron independiente del gateway

`hermes cron daemon` ejecuta el ticker nativo sin abrir adaptadores ni iniciar el
kanban del gateway. Sirve los mismos perfiles que el ticker del gateway, vuelve a
enumerarlos en cada ciclo y reutiliza su config, estado, pausa, concurrencia y
entrega. `--single-profile` limita el proceso al perfil de lanzamiento.

Configurar `cron.provider: standalone` en el home del gateway y del escritorio
para que sus proveedores no arranquen otro ticker ni un catch-up externo. El
daemon arranca el motor builtin directamente. `cron.provider` selecciona el
plugin que dispara horarios; no selecciona el proveedor del modelo de los jobs.

El daemon conserva un candado `.scheduler-owner.lock` por home hasta que termina
el drenaje de jobs. Otro daemon falla al arrancar; los ticks embebidos y manuales
ceden mientras ese candado esté ocupado. El candado de tick y las reclamaciones
por ejecución siguen activos. Los jobs solicitados manualmente con `cron run`
se marcan para el siguiente tick del daemon. Si se modifica un perfil servido,
se recomienda reiniciar el daemon para liberar leases de perfiles retirados.

En macOS, materializar `launchd/ai.hermes.cron.plist.in` sustituyendo `__HOME__`
y `__RUNTIME__` por rutas absolutas. KeepAlive mantiene el reloj para cumplir los
horarios, sin agentes o modelos residentes. Arranca jobs solo cuando corresponde.
Con el reloj apagado rigen las ventanas nativas de recuperación de cron; no se
garantiza ejecutar horarios fuera de esas ventanas. SIGTERM deja de admitir
nuevos jobs y drena los ya admitidos; para apagarlo bajo launchd usar bootout.

Después de instalar el cambio, reiniciar primero el gateway y después cargar el
nuevo servicio. No cargar el daemon junto a un gateway con código viejo, que aún
no conoce el candado. La prueba aislada verifica un job no_agent, la exclusión de
un segundo daemon y continuidad frente al reinicio del ticker embebido; no
sustituye la comprobación operacional después de activar los servicios.
