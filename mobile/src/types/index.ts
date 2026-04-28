export interface BolEntry {
  time: number;
  end_time: number;
  bol: string;
  confidence: number;
  source: 'both' | 'template' | 'model';
  template_bol: string | null;
  model_bol: string | null;
  model_confidence: number;
  beat_position: number;
  cycle: number;
  model_top_3?: Array<{ label: string; display_name: string; confidence: number }>;
}

export interface TaalTopK {
  label: string;
  display_name: string;
  confidence: number;
  beats?: number;
  vibhag?: string;
}

export interface TaalResult {
  label: string;
  display_name: string;
  confidence: number;
  top_k: TaalTopK[];
}

export interface SequenceResult {
  status: string;
  taal: TaalResult;
  theka: string;
  bol_sequence: BolEntry[];
  bol_count: number;
  cycles_detected: number;
  pattern_length: number;
  audio_info: {
    duration_seconds: number;
    sample_rate: number;
    source: string;
  };
  processing_time_ms: number;
}

export interface HealthCheck {
  status: string;
  models_loaded: boolean;
  device: string;
}

export interface HistoryItem {
  id: string;
  date: string;
  taalName: string;
  taalConfidence: number;
  bolCount: number;
  duration: number;
  processingTime: number;
  result: SequenceResult;
}
