export interface WordTimestamp {
  word: string;
  start_s: number;
  end_s: number;
  confidence: number;
}

interface BaseEvent {
  call_id: string;
  timestamp_utc: string;
}

export interface TranscriptEvent extends BaseEvent {
  event: "transcript";
  seq: number;
  original_transcript: string;
  detected_language: string;
  asr_confidence: number;
  word_timestamps: WordTimestamp[];
  is_final: boolean;
}

export interface TranslationEvent extends BaseEvent {
  event: "translation";
  seq: number;
  source_text: string;
  translated_text: string;
  source_language: string;
  target_language: string;
  translation_model: string;
}

export interface ProsodySnapshot {
  pitch_mean_hz: number | null;
  pitch_std_hz: number | null;
  rms_energy_db: number | null;
  mfcc_1_mean: number | null;
}

export interface DetectedKeyword {
  keyword: string;
  detected_at_utc: string;
  source: "transcript";
  transcript_seq: number | null;
}

export interface AnalyticsEvent extends BaseEvent {
  event: "analytics";
  seq: number;
  stress_level: StressLevel;
  stress_confidence: number;
  prosody: ProsodySnapshot;
  keywords_detected: DetectedKeyword[];
}

export interface BargeInEvent extends BaseEvent {
  event: "barge_in";
  priority: "high";
  reason: string;
}

export type StressLevel = "low" | "medium" | "high";

export type ServerEvent =
  | TranscriptEvent
  | TranslationEvent
  | AnalyticsEvent
  | BargeInEvent;

export type ConnectionStatus =
  | "connecting"
  | "connected"
  | "reconnecting"
  | "disconnected";

export type BargeInState = "idle" | "speaking" | "interrupted";

export interface TranscriptEntry {
  seq: number;
  original: string;
  translated: string | null;
  language: string;
  confidence: number;
  isFinal: boolean;
  timestamp: string;
}

export interface AnalyticsSnapshot {
  stressLevel: StressLevel;
  stressConfidence: number;
  prosody: ProsodySnapshot;
  keywords: DetectedKeyword[];
}

export interface CallState {
  transcripts: TranscriptEntry[];
  analytics: AnalyticsSnapshot;
  bargeIn: BargeInState;
  connectionStatus: ConnectionStatus;
}
