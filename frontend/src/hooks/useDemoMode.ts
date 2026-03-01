import { useEffect, useRef } from "react";
import type { CallStore } from "../stores/callStore";
import type { ServerEvent } from "../types/events";

const DEMO_SCRIPT: { delayMs: number; event: ServerEvent }[] = [
  {
    delayMs: 800,
    event: {
      event: "transcript",
      call_id: "demo",
      timestamp_utc: new Date().toISOString(),
      seq: 0,
      original_transcript: "Hola, necesito",
      detected_language: "es",
      asr_confidence: 0.72,
      word_timestamps: [],
      is_final: false,
    },
  },
  {
    delayMs: 1400,
    event: {
      event: "transcript",
      call_id: "demo",
      timestamp_utc: new Date().toISOString(),
      seq: 0,
      original_transcript: "Hola, necesito ayuda, hay un incendio en mi casa",
      detected_language: "es",
      asr_confidence: 0.94,
      word_timestamps: [],
      is_final: true,
    },
  },
  {
    delayMs: 500,
    event: {
      event: "analytics",
      call_id: "demo",
      timestamp_utc: new Date().toISOString(),
      seq: 0,
      stress_level: "medium",
      stress_confidence: 0.68,
      prosody: { pitch_mean_hz: 220, pitch_std_hz: 40, rms_energy_db: 0.5, mfcc_1_mean: null },
      keywords_detected: [],
    },
  },
  {
    delayMs: 600,
    event: {
      event: "translation",
      call_id: "demo",
      timestamp_utc: new Date().toISOString(),
      seq: 0,
      source_text: "Hola, necesito ayuda, hay un incendio en mi casa",
      translated_text: "Hello, I need help, there is a fire in my house",
      source_language: "es",
      target_language: "en",
      translation_model: "riva-translate-1.6b",
    },
  },
  {
    delayMs: 500,
    event: {
      event: "analytics",
      call_id: "demo",
      timestamp_utc: new Date().toISOString(),
      seq: 1,
      stress_level: "high",
      stress_confidence: 0.91,
      prosody: { pitch_mean_hz: 340, pitch_std_hz: 65, rms_energy_db: 0.82, mfcc_1_mean: null },
      keywords_detected: [
        { keyword: "fire", detected_at_utc: new Date().toISOString(), source: "transcript" as const, transcript_seq: 0 },
      ],
    },
  },
  {
    delayMs: 2000,
    event: {
      event: "transcript",
      call_id: "demo",
      timestamp_utc: new Date().toISOString(),
      seq: 1,
      original_transcript: "Por favor, vengan rápido, hay niños aquí",
      detected_language: "es",
      asr_confidence: 0.91,
      word_timestamps: [],
      is_final: true,
    },
  },
  {
    delayMs: 700,
    event: {
      event: "translation",
      call_id: "demo",
      timestamp_utc: new Date().toISOString(),
      seq: 1,
      source_text: "Por favor, vengan rápido, hay niños aquí",
      translated_text: "Please, come quickly, there are children here",
      source_language: "es",
      target_language: "en",
      translation_model: "riva-translate-1.6b",
    },
  },
  {
    delayMs: 1500,
    event: {
      event: "barge_in",
      call_id: "demo",
      timestamp_utc: new Date().toISOString(),
      priority: "high",
      reason: "caller_speech_detected_during_playback",
    },
  },
  {
    delayMs: 1200,
    event: {
      event: "transcript",
      call_id: "demo",
      timestamp_utc: new Date().toISOString(),
      seq: 2,
      original_transcript: "¡El fuego está en la cocina! ¡Ayúdenme!",
      detected_language: "es",
      asr_confidence: 0.88,
      word_timestamps: [],
      is_final: true,
    },
  },
  {
    delayMs: 600,
    event: {
      event: "translation",
      call_id: "demo",
      timestamp_utc: new Date().toISOString(),
      seq: 2,
      source_text: "¡El fuego está en la cocina! ¡Ayúdenme!",
      translated_text: "The fire is in the kitchen! Help me!",
      source_language: "es",
      target_language: "en",
      translation_model: "riva-translate-1.6b",
    },
  },
  {
    delayMs: 500,
    event: {
      event: "analytics",
      call_id: "demo",
      timestamp_utc: new Date().toISOString(),
      seq: 2,
      stress_level: "high",
      stress_confidence: 0.97,
      prosody: { pitch_mean_hz: 410, pitch_std_hz: 80, rms_energy_db: 0.93, mfcc_1_mean: null },
      keywords_detected: [
        { keyword: "fire", detected_at_utc: new Date().toISOString(), source: "transcript" as const, transcript_seq: 2 },
      ],
    },
  },
];

export function useDemoMode(store: CallStore): void {
  const timersRef = useRef<ReturnType<typeof setTimeout>[]>([]);

  useEffect(() => {
    store.setConnectionStatus("connected");

    let cumulativeDelay = 500;
    for (const step of DEMO_SCRIPT) {
      cumulativeDelay += step.delayMs;
      const timer = setTimeout(() => {
        store.dispatch(step.event);
      }, cumulativeDelay);
      timersRef.current.push(timer);
    }

    return () => {
      for (const t of timersRef.current) clearTimeout(t);
      timersRef.current = [];
    };
  }, [store]);
}
