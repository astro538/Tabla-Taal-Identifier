import AsyncStorage from '@react-native-async-storage/async-storage';
import { HistoryItem, SequenceResult } from '../types';

const HISTORY_KEY = 'analysis_history';

export const saveToHistory = async (result: SequenceResult): Promise<void> => {
  const history = await getHistory();
  const item: HistoryItem = {
    id: Date.now().toString(),
    date: new Date().toISOString(),
    taalName: result.taal.display_name,
    taalConfidence: result.taal.confidence,
    bolCount: result.bol_count,
    duration: result.audio_info.duration_seconds,
    processingTime: result.processing_time_ms,
    result,
  };
  history.unshift(item);
  // Keep last 50
  const trimmed = history.slice(0, 50);
  await AsyncStorage.setItem(HISTORY_KEY, JSON.stringify(trimmed));
};

export const getHistory = async (): Promise<HistoryItem[]> => {
  const raw = await AsyncStorage.getItem(HISTORY_KEY);
  return raw ? JSON.parse(raw) : [];
};

export const clearHistory = async (): Promise<void> => {
  await AsyncStorage.removeItem(HISTORY_KEY);
};

export const deleteHistoryItem = async (id: string): Promise<void> => {
  const history = await getHistory();
  const filtered = history.filter((h) => h.id !== id);
  await AsyncStorage.setItem(HISTORY_KEY, JSON.stringify(filtered));
};
