import { resolveGatewayWsUrl } from '@hermes/shared'
import { useCallback, useRef, useState } from 'react'

import { notifyError } from '@/store/notifications'

/**
 * Modo de voz en vivo vía Gemini Live — segundo cerebro de Alfred, en paralelo
 * al modo habitual (STT local -> Codex -> TTS, ver use-voice-conversation.ts).
 * FX-142: sin acceso a tools/gbrain todavía, conversación pura de baja
 * latencia con interrupción real manejada por el propio servidor de Gemini.
 *
 * El navegador nunca ve la API key de Google: se conecta al puente local
 * (`/api/voice/gemini-live`, hermes_cli/web_server.py -> agent/gemini_live_bridge.py)
 * con el mismo mecanismo de ticket que ya usa el resto del Desktop.
 */

export type GeminiLiveStatus = 'idle' | 'connecting' | 'listening' | 'speaking' | 'error'

async function resolveGeminiLiveWsUrl(): Promise<null | string> {
  const desktop = window.hermesDesktop

  if (!desktop?.getConnection) {
    return null
  }

  try {
    const wsUrl = await resolveGatewayWsUrl(desktop, await desktop.getConnection())
    const url = new URL(wsUrl)

    if (!url.pathname.endsWith('/api/ws')) {
      return null
    }

    url.pathname = url.pathname.replace(/\/api\/ws$/, '/api/voice/gemini-live')

    return url.toString()
  } catch {
    return null
  }
}

export function useGeminiLiveConversation() {
  const [status, setStatus] = useState<GeminiLiveStatus>('idle')
  const wsRef = useRef<WebSocket | null>(null)
  const micContextRef = useRef<AudioContext | null>(null)
  const micProcessorRef = useRef<ScriptProcessorNode | null>(null)
  const micStreamRef = useRef<MediaStream | null>(null)
  const playContextRef = useRef<AudioContext | null>(null)
  const nextPlayAtRef = useRef(0)
  const scheduledSourcesRef = useRef<AudioBufferSourceNode[]>([])
  const runningRef = useRef(false)

  const stopAllScheduledAudio = useCallback(() => {
    for (const source of scheduledSourcesRef.current) {
      try {
        source.onended = null
        source.stop()
      } catch {
        // ya terminó / ya estaba detenida
      }
    }

    scheduledSourcesRef.current = []
    nextPlayAtRef.current = playContextRef.current?.currentTime ?? 0
  }, [])

  const playPcmChunk = useCallback((int16: Int16Array, sampleRate: number) => {
    if (!playContextRef.current) {
      playContextRef.current = new AudioContext({ sampleRate })
    }

    const context = playContextRef.current
    const buffer = context.createBuffer(1, int16.length, sampleRate)
    const channel = buffer.getChannelData(0)

    for (let i = 0; i < int16.length; i += 1) {
      channel[i] = int16[i] / 32_768
    }

    const source = context.createBufferSource()
    source.buffer = buffer
    source.connect(context.destination)

    const startAt = Math.max(context.currentTime + 0.02, nextPlayAtRef.current)
    source.start(startAt)
    nextPlayAtRef.current = startAt + buffer.duration
    scheduledSourcesRef.current.push(source)
    setStatus('speaking')

    source.onended = () => {
      scheduledSourcesRef.current = scheduledSourcesRef.current.filter(s => s !== source)

      if (scheduledSourcesRef.current.length === 0) {
        setStatus('listening')
      }
    }
  }, [])

  const startMic = useCallback((ws: WebSocket) => {
    void (async () => {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true }
      })

      micStreamRef.current = stream

      const context = new AudioContext()
      micContextRef.current = context

      const source = context.createMediaStreamSource(stream)
      const processor = context.createScriptProcessor(4_096, 1, 1)
      micProcessorRef.current = processor

      source.connect(processor)
      processor.connect(context.destination)

      processor.onaudioprocess = event => {
        if (!runningRef.current || ws.readyState !== WebSocket.OPEN) {
          return
        }

        const input = event.inputBuffer.getChannelData(0)
        const ratio = context.sampleRate / 16_000
        const outLength = Math.floor(input.length / ratio)
        const out = new Int16Array(outLength)

        for (let i = 0; i < outLength; i += 1) {
          const s = Math.max(-1, Math.min(1, input[Math.floor(i * ratio)]))
          out[i] = s < 0 ? s * 0x80_00 : s * 0x7f_ff
        }

        const bytes = new Uint8Array(out.buffer)
        let binary = ''

        for (let i = 0; i < bytes.length; i += 1) {
          binary += String.fromCharCode(bytes[i])
        }

        ws.send(JSON.stringify({ realtimeInput: { audio: { data: btoa(binary), mimeType: 'audio/pcm;rate=16000' } } }))
      }

      setStatus('listening')
    })().catch(error => notifyError(error, 'No pude activar el micrófono para el modo de voz en vivo.'))
  }, [])

  const stopMic = useCallback(() => {
    if (micProcessorRef.current) {
      try {
        micProcessorRef.current.disconnect()
      } catch {
        // ya desconectado
      }

      micProcessorRef.current = null
    }

    if (micStreamRef.current) {
      micStreamRef.current.getTracks().forEach(track => track.stop())
      micStreamRef.current = null
    }
  }, [])

  const end = useCallback(() => {
    runningRef.current = false
    stopAllScheduledAudio()
    stopMic()

    if (wsRef.current) {
      wsRef.current.close()
      wsRef.current = null
    }

    setStatus('idle')
  }, [stopAllScheduledAudio, stopMic])

  const start = useCallback(async () => {
    setStatus('connecting')

    const wsUrl = await resolveGeminiLiveWsUrl()

    if (!wsUrl) {
      setStatus('error')
      notifyError(new Error('No pude resolver la conexión de Gemini Live'), 'Modo de voz en vivo no disponible.')

      return
    }

    runningRef.current = true

    const ws = new WebSocket(wsUrl)
    wsRef.current = ws

    ws.onmessage = event => {
      let data: {
        error?: string
        serverContent?: {
          modelTurn?: { parts?: Array<{ inlineData?: { data?: string; mimeType?: string } }> }
          interrupted?: boolean
        }
        type?: string
      }

      try {
        data = JSON.parse(typeof event.data === 'string' ? event.data : '{}')
      } catch {
        return
      }

      if (data.type === 'bridge_error') {
        setStatus('error')
        notifyError(
          new Error(data.error || 'gemini-live bridge error'),
          data.error
            ? `El puente de voz en vivo falló: ${data.error}`
            : 'El puente de voz en vivo falló del lado del servidor.'
        )

        return
      }

      const parts = data.serverContent?.modelTurn?.parts

      if (parts) {
        for (const part of parts) {
          if (part.inlineData?.data) {
            const binary = atob(part.inlineData.data)
            const bytes = new Uint8Array(binary.length)

            for (let i = 0; i < binary.length; i += 1) {
              bytes[i] = binary.charCodeAt(i)
            }

            const rate = Number(part.inlineData.mimeType?.match(/rate=(\d+)/)?.[1] ?? 24_000)

            playPcmChunk(new Int16Array(bytes.buffer), rate)
          }
        }
      }

      if (data.serverContent?.interrupted) {
        stopAllScheduledAudio()
        setStatus('listening')
      }
    }

    ws.onopen = () => startMic(ws)

    ws.onerror = () => {
      setStatus('error')
    }

    ws.onclose = () => {
      stopAllScheduledAudio()
      stopMic()
      runningRef.current = false
      setStatus('idle')
    }
  }, [playPcmChunk, startMic, stopAllScheduledAudio, stopMic])

  return { end, start, status }
}
