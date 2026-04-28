import React, { useState, useRef, useEffect, useCallback } from 'react';
import {
  View, Text, StyleSheet, TouchableOpacity, SafeAreaView, ScrollView,
  StatusBar, Alert, ActivityIndicator,
} from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { Audio, AVPlaybackStatus } from 'expo-av';
import * as Haptics from 'expo-haptics';
import { useTheme } from '../theme/ThemeContext';
import { gradients, spacing, fontSize, radius } from '../theme';
import { SequenceResult } from '../types';
import { predictSequence } from '../services/api';
import { saveToHistory } from '../services/storage';
import { GlassCard } from '../components/GlassCard';
import { TaalCard } from '../components/TaalCard';
import { BolTimeline } from '../components/BolTimeline';

export const AnalysisScreen = ({ route, navigation }: any) => {
  const { audioUri, fileName, mimeType } = route.params;
  const { c, mode } = useTheme();
  const [sound, setSound] = useState<Audio.Sound | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [result, setResult] = useState<SequenceResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Load audio on mount
  useEffect(() => {
    loadAudio();
    return () => { sound?.unloadAsync(); };
  }, []);

  const loadAudio = async () => {
    try {
      await Audio.setAudioModeAsync({ allowsRecordingIOS: false, playsInSilentModeIOS: true });
      const { sound: s, status } = await Audio.Sound.createAsync(
        { uri: audioUri },
        { shouldPlay: false, progressUpdateIntervalMillis: 80 },
        onPlaybackUpdate
      );
      setSound(s);
      if (status.isLoaded) setDuration(status.durationMillis ? status.durationMillis / 1000 : 0);
    } catch (err) {
      Alert.alert('Error', 'Could not load audio file.');
    }
  };

  const onPlaybackUpdate = useCallback((status: AVPlaybackStatus) => {
    if (!status.isLoaded) return;
    setCurrentTime((status.positionMillis || 0) / 1000);
    setIsPlaying(status.isPlaying);
    if (status.durationMillis) setDuration(status.durationMillis / 1000);
    if (status.didJustFinish) setIsPlaying(false);
  }, []);

  const togglePlay = async () => {
    if (!sound) return;
    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
    if (isPlaying) {
      await sound.pauseAsync();
    } else {
      const status = await sound.getStatusAsync();
      if (status.isLoaded && status.positionMillis >= (status.durationMillis || 0) - 100) {
        await sound.setPositionAsync(0);
      }
      await sound.playAsync();
    }
  };

  const seekTo = async (time: number) => {
    if (!sound) return;
    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
    await sound.setPositionAsync(time * 1000);
    if (!isPlaying) await sound.playAsync();
  };

  const analyze = async () => {
    setIsAnalyzing(true);
    setError(null);
    setResult(null);
    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);

    try {
      const res = await predictSequence(audioUri, fileName, mimeType);
      setResult(res);
      await saveToHistory(res);
      Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
    } catch (err: any) {
      setError(err.message || 'Analysis failed');
      Haptics.notificationAsync(Haptics.NotificationFeedbackType.Error);
    } finally {
      setIsAnalyzing(false);
    }
  };

  const formatTime = (s: number) => {
    const m = Math.floor(s / 60);
    const sec = Math.floor(s % 60);
    return `${m}:${sec.toString().padStart(2, '0')}`;
  };

  const progressPct = duration > 0 ? (currentTime / duration) * 100 : 0;

  return (
    <SafeAreaView style={[styles.safe, { backgroundColor: c.bg }]}>
      <StatusBar barStyle={mode === 'dark' ? 'light-content' : 'dark-content'} />
      <ScrollView contentContainerStyle={styles.scroll} showsVerticalScrollIndicator={false}>
        {/* Audio Player */}
        <GlassCard>
          <Text style={[styles.sectionLabel, { color: c.textMuted }]}>AUDIO</Text>
          <Text style={[styles.fileName, { color: c.text }]} numberOfLines={1}>{fileName}</Text>

          {/* Progress Bar */}
          <TouchableOpacity
            activeOpacity={0.9}
            onPress={(e) => {
              const x = e.nativeEvent.locationX;
              const width = e.nativeEvent.target ? 300 : 300; // approximate
              seekTo((x / 300) * duration);
            }}
            style={[styles.progressBar, { backgroundColor: c.text + '10' }]}
          >
            <LinearGradient
              colors={gradients.primary as any}
              start={{ x: 0, y: 0 }}
              end={{ x: 1, y: 0 }}
              style={[styles.progressFill, { width: `${progressPct}%` }]}
            />
          </TouchableOpacity>

          <View style={styles.timeRow}>
            <Text style={[styles.timeText, { color: c.textMuted }]}>{formatTime(currentTime)}</Text>
            <Text style={[styles.timeText, { color: c.textMuted }]}>{formatTime(duration)}</Text>
          </View>

          {/* Controls */}
          <View style={styles.controls}>
            <TouchableOpacity onPress={() => seekTo(Math.max(0, currentTime - 5))} style={[styles.seekBtn, { backgroundColor: c.text + '10' }]}>
              <Text style={[styles.seekText, { color: c.textSecondary }]}>-5s</Text>
            </TouchableOpacity>

            <TouchableOpacity onPress={togglePlay} activeOpacity={0.8}>
              <LinearGradient colors={gradients.primary as any} style={styles.playBtn}>
                <Text style={styles.playIcon}>{isPlaying ? '⏸' : '▶️'}</Text>
              </LinearGradient>
            </TouchableOpacity>

            <TouchableOpacity onPress={() => seekTo(Math.min(duration, currentTime + 5))} style={[styles.seekBtn, { backgroundColor: c.text + '10' }]}>
              <Text style={[styles.seekText, { color: c.textSecondary }]}>+5s</Text>
            </TouchableOpacity>
          </View>
        </GlassCard>

        {/* Analyze Button */}
        {!result && (
          <TouchableOpacity onPress={analyze} disabled={isAnalyzing} activeOpacity={0.85}>
            <LinearGradient
              colors={isAnalyzing ? [c.textMuted, c.textMuted] : gradients.primary as any}
              start={{ x: 0, y: 0 }}
              end={{ x: 1, y: 0 }}
              style={styles.analyzeBtn}
            >
              {isAnalyzing ? (
                <View style={styles.analyzingRow}>
                  <ActivityIndicator color="#fff" size="small" />
                  <Text style={styles.analyzeBtnText}>  Analyzing...</Text>
                </View>
              ) : (
                <Text style={styles.analyzeBtnText}>🧠 Analyze Tabla</Text>
              )}
            </LinearGradient>
          </TouchableOpacity>
        )}

        {/* Error */}
        {error && (
          <GlassCard style={{ borderColor: c.rose + '30' }}>
            <Text style={[styles.errorText, { color: c.rose }]}>❌ {error}</Text>
            <TouchableOpacity onPress={analyze} style={[styles.retryBtn, { backgroundColor: c.rose + '15' }]}>
              <Text style={[styles.retryText, { color: c.rose }]}>Retry</Text>
            </TouchableOpacity>
          </GlassCard>
        )}

        {/* Results */}
        {result && (
          <>
            {/* Status */}
            <GlassCard style={{ borderColor: c.green + '20' }}>
              <Text style={[styles.statusText, { color: c.green }]}>
                ✅ Done in {result.processing_time_ms.toFixed(0)}ms · {result.bol_count} Bols · {result.audio_info.duration_seconds}s
              </Text>
            </GlassCard>

            {/* Taal Card */}
            <TaalCard taal={result.taal} theka={result.theka} processingTime={result.processing_time_ms} />

            {/* Bol Timeline */}
            {result.bol_sequence.length > 0 && (
              <BolTimeline
                bols={result.bol_sequence}
                duration={result.audio_info.duration_seconds}
                currentTime={currentTime}
                bolCount={result.bol_count}
                cyclesDetected={result.cycles_detected}
                patternLength={result.pattern_length}
                onSeek={seekTo}
              />
            )}

            {/* Re-analyze */}
            <TouchableOpacity onPress={analyze} activeOpacity={0.85}>
              <GlassCard>
                <Text style={[styles.reanalyze, { color: c.blue }]}>🔄 Re-analyze</Text>
              </GlassCard>
            </TouchableOpacity>
          </>
        )}
      </ScrollView>
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  safe: { flex: 1 },
  scroll: { padding: spacing.xl, paddingBottom: 40 },
  sectionLabel: { fontSize: 9, fontWeight: '700', letterSpacing: 1.5, marginBottom: spacing.xs },
  fileName: { fontSize: fontSize.body, fontWeight: '600', marginBottom: spacing.md },
  progressBar: { height: 6, borderRadius: 3, overflow: 'hidden', marginBottom: spacing.xs },
  progressFill: { height: '100%', borderRadius: 3 },
  timeRow: { flexDirection: 'row', justifyContent: 'space-between', marginBottom: spacing.md },
  timeText: { fontSize: fontSize.xs, fontVariant: ['tabular-nums'] },
  controls: { flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: 20 },
  seekBtn: { paddingHorizontal: 14, paddingVertical: 8, borderRadius: radius.md },
  seekText: { fontSize: fontSize.sm, fontWeight: '600' },
  playBtn: { width: 56, height: 56, borderRadius: 28, alignItems: 'center', justifyContent: 'center' },
  playIcon: { fontSize: 22 },
  analyzeBtn: { padding: 16, borderRadius: radius.md, alignItems: 'center', marginBottom: spacing.md },
  analyzeBtnText: { color: '#fff', fontSize: fontSize.lg, fontWeight: '700' },
  analyzingRow: { flexDirection: 'row', alignItems: 'center' },
  errorText: { fontSize: fontSize.sm, marginBottom: spacing.sm },
  retryBtn: { alignSelf: 'flex-start', paddingHorizontal: 16, paddingVertical: 8, borderRadius: radius.sm },
  retryText: { fontSize: fontSize.sm, fontWeight: '600' },
  statusText: { fontSize: fontSize.sm, fontWeight: '500', textAlign: 'center' },
  reanalyze: { textAlign: 'center', fontSize: fontSize.body, fontWeight: '600' },
});
