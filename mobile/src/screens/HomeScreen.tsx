import React, { useState, useRef } from 'react';
import {
  View, Text, StyleSheet, TouchableOpacity, SafeAreaView, StatusBar, Alert, ActivityIndicator,
} from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { Audio } from 'expo-av';
import * as DocumentPicker from 'expo-document-picker';
import * as Haptics from 'expo-haptics';
import { useTheme } from '../theme/ThemeContext';
import { gradients, spacing, fontSize, radius, screen } from '../theme';
import { GlassCard } from '../components/GlassCard';

export const HomeScreen = ({ navigation }: any) => {
  const { c, mode, toggle } = useTheme();
  const [isRecording, setIsRecording] = useState(false);
  const [recordingTime, setRecordingTime] = useState(0);
  const recordingRef = useRef<Audio.Recording | null>(null);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const startRecording = async () => {
    try {
      const perm = await Audio.requestPermissionsAsync();
      if (!perm.granted) {
        Alert.alert('Permission Required', 'Microphone access is needed to record Tabla audio.');
        return;
      }
      await Audio.setAudioModeAsync({ allowsRecordingIOS: true, playsInSilentModeIOS: true });

      const { recording } = await Audio.Recording.createAsync(
        Audio.RecordingOptionsPresets.HIGH_QUALITY
      );
      recordingRef.current = recording;
      setIsRecording(true);
      setRecordingTime(0);
      Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);

      timerRef.current = setInterval(() => {
        setRecordingTime((t) => {
          if (t >= 29) { stopRecording(); return 30; }
          return t + 1;
        });
      }, 1000);
    } catch (err) {
      Alert.alert('Error', 'Could not start recording. Please try again.');
    }
  };

  const stopRecording = async () => {
    if (!recordingRef.current) return;
    if (timerRef.current) clearInterval(timerRef.current);
    setIsRecording(false);
    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Heavy);

    try {
      await recordingRef.current.stopAndUnloadAsync();
      const uri = recordingRef.current.getURI();
      recordingRef.current = null;
      if (uri) {
        navigation.navigate('Analysis', { audioUri: uri, fileName: 'recording.m4a', mimeType: 'audio/m4a' });
      }
    } catch (err) {
      Alert.alert('Error', 'Could not save recording.');
    }
  };

  const pickFile = async () => {
    try {
      const result = await DocumentPicker.getDocumentAsync({ type: 'audio/*', copyToCacheDirectory: true });
      if (!result.canceled && result.assets?.length > 0) {
        const asset = result.assets[0];
        Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
        navigation.navigate('Analysis', {
          audioUri: asset.uri,
          fileName: asset.name || 'audio.wav',
          mimeType: asset.mimeType || 'audio/wav',
        });
      }
    } catch (err) {
      Alert.alert('Error', 'Could not pick file.');
    }
  };

  const formatTime = (s: number) => `${Math.floor(s / 60)}:${(s % 60).toString().padStart(2, '0')}`;

  return (
    <SafeAreaView style={[styles.safe, { backgroundColor: c.bg }]}>
      <StatusBar barStyle={mode === 'dark' ? 'light-content' : 'dark-content'} />
      <View style={styles.container}>
        {/* Header */}
        <View style={styles.header}>
          <Text style={styles.emoji}>🥁</Text>
          <Text style={[styles.title, { color: c.text }]}>Tabla Identifier</Text>
          <Text style={[styles.subtitle, { color: c.textSecondary }]}>
            AI-powered Taal & Bol detection
          </Text>
          <TouchableOpacity onPress={toggle} style={[styles.themeBtn, { backgroundColor: c.text + '10' }]}>
            <Text style={{ fontSize: 18 }}>{mode === 'dark' ? '☀️' : '🌙'}</Text>
          </TouchableOpacity>
        </View>

        {/* Action Buttons */}
        {isRecording ? (
          <GlassCard>
            <View style={styles.recordingView}>
              <View style={[styles.recDot, { backgroundColor: c.rose }]} />
              <Text style={[styles.recTime, { color: c.text }]}>{formatTime(recordingTime)}</Text>
              <Text style={[styles.recLabel, { color: c.textSecondary }]}>Recording... (max 30s)</Text>
              <TouchableOpacity onPress={stopRecording} activeOpacity={0.8}>
                <LinearGradient colors={[c.rose, '#DC2626']} style={styles.stopBtn}>
                  <Text style={styles.stopBtnText}>⏹ Stop Recording</Text>
                </LinearGradient>
              </TouchableOpacity>
            </View>
          </GlassCard>
        ) : (
          <>
            <TouchableOpacity onPress={startRecording} activeOpacity={0.85}>
              <LinearGradient colors={gradients.primary as any} start={{ x: 0, y: 0 }} end={{ x: 1, y: 1 }} style={styles.actionBtn}>
                <Text style={styles.actionIcon}>🎙️</Text>
                <Text style={styles.actionTitle}>Record Audio</Text>
                <Text style={styles.actionSub}>Record live Tabla playing</Text>
              </LinearGradient>
            </TouchableOpacity>

            <TouchableOpacity onPress={pickFile} activeOpacity={0.85}>
              <GlassCard>
                <View style={styles.uploadInner}>
                  <Text style={styles.actionIcon}>📁</Text>
                  <Text style={[styles.actionTitle, { color: c.text }]}>Upload Audio</Text>
                  <Text style={[styles.actionSub, { color: c.textSecondary }]}>WAV, MP3, OGG, FLAC, M4A</Text>
                </View>
              </GlassCard>
            </TouchableOpacity>
          </>
        )}

        {/* How It Works */}
        <GlassCard>
          <Text style={[styles.sectionLabel, { color: c.textMuted }]}>HOW IT WORKS</Text>
          <View style={styles.steps}>
            {[
              { icon: '🎵', label: 'Record or Upload', desc: 'Tabla audio file' },
              { icon: '🧠', label: 'AI Analysis', desc: 'Deep learning models' },
              { icon: '📊', label: 'See Results', desc: 'Taal + Bol timeline' },
            ].map((step, i) => (
              <View key={i} style={styles.step}>
                <Text style={styles.stepIcon}>{step.icon}</Text>
                <Text style={[styles.stepLabel, { color: c.text }]}>{step.label}</Text>
                <Text style={[styles.stepDesc, { color: c.textMuted }]}>{step.desc}</Text>
              </View>
            ))}
          </View>
        </GlassCard>

        {/* Settings link */}
        <TouchableOpacity onPress={() => navigation.navigate('Settings')} style={styles.settingsLink}>
          <Text style={[styles.settingsText, { color: c.textMuted }]}>⚙️ API Settings</Text>
        </TouchableOpacity>
      </View>
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  safe: { flex: 1 },
  container: { flex: 1, paddingHorizontal: spacing.xl, paddingTop: spacing.lg },
  header: { alignItems: 'center', paddingVertical: spacing.xxl, position: 'relative' },
  emoji: { fontSize: 48, marginBottom: spacing.sm },
  title: { fontSize: fontSize.hero, fontWeight: '800' },
  subtitle: { fontSize: fontSize.md, marginTop: 4 },
  themeBtn: { position: 'absolute', top: spacing.xxl, right: 0, padding: 10, borderRadius: radius.full },
  actionBtn: { borderRadius: radius.lg, padding: spacing.xl, alignItems: 'center', marginBottom: spacing.md },
  actionIcon: { fontSize: 32, marginBottom: spacing.sm },
  actionTitle: { fontSize: fontSize.lg, fontWeight: '700', color: '#fff' },
  actionSub: { fontSize: fontSize.sm, color: 'rgba(255,255,255,0.7)', marginTop: 2 },
  uploadInner: { alignItems: 'center' },
  recordingView: { alignItems: 'center', paddingVertical: spacing.lg },
  recDot: { width: 16, height: 16, borderRadius: 8, marginBottom: spacing.md },
  recTime: { fontSize: 42, fontWeight: '800', fontVariant: ['tabular-nums'] },
  recLabel: { fontSize: fontSize.sm, marginBottom: spacing.lg },
  stopBtn: { paddingHorizontal: 28, paddingVertical: 14, borderRadius: radius.md },
  stopBtnText: { color: '#fff', fontSize: fontSize.body, fontWeight: '700' },
  sectionLabel: { fontSize: 9, fontWeight: '700', letterSpacing: 1.5, marginBottom: spacing.md },
  steps: { flexDirection: 'row', justifyContent: 'space-between' },
  step: { alignItems: 'center', flex: 1 },
  stepIcon: { fontSize: 24, marginBottom: 6 },
  stepLabel: { fontSize: fontSize.sm, fontWeight: '600' },
  stepDesc: { fontSize: fontSize.xs, marginTop: 2, textAlign: 'center' },
  settingsLink: { alignItems: 'center', paddingVertical: spacing.lg },
  settingsText: { fontSize: fontSize.sm },
});
