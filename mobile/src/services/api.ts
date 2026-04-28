import AsyncStorage from '@react-native-async-storage/async-storage';
import { SequenceResult, HealthCheck } from '../types';

const DEFAULT_URL = 'http://10.0.2.2:8000';
const STORAGE_KEY = 'api_base_url';

let baseUrl = DEFAULT_URL;

export const loadBaseUrl = async (): Promise<string> => {
  const saved = await AsyncStorage.getItem(STORAGE_KEY);
  baseUrl = saved || DEFAULT_URL;
  return baseUrl;
};

export const setBaseUrl = async (url: string) => {
  baseUrl = url.replace(/\/+$/, '');
  await AsyncStorage.setItem(STORAGE_KEY, baseUrl);
};

export const getBaseUrl = () => baseUrl;

export const checkHealth = async (): Promise<HealthCheck> => {
  const res = await fetch(`${baseUrl}/api/v1/health`, { method: 'GET' });
  if (!res.ok) throw new Error('Server unreachable');
  return res.json();
};

export const predictSequence = async (
  fileUri: string,
  fileName: string = 'audio.wav',
  mimeType: string = 'audio/wav'
): Promise<SequenceResult> => {
  const formData = new FormData();
  formData.append('file', {
    uri: fileUri,
    type: mimeType,
    name: fileName,
  } as any);

  const res = await fetch(`${baseUrl}/api/v1/predict/sequence`, {
    method: 'POST',
    body: formData,
    headers: { 'Content-Type': 'multipart/form-data' },
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Server error' }));
    throw new Error(err.detail || `HTTP ${res.status}`);
  }
  return res.json();
};

export const getModelInfo = async () => {
  const res = await fetch(`${baseUrl}/api/v1/models`);
  if (!res.ok) throw new Error('Failed to fetch model info');
  return res.json();
};
