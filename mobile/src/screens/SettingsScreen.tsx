import React, { useState, useEffect } from 'react';
import {
  View, Text, StyleSheet, SafeAreaView, TextInput, TouchableOpacity,
  StatusBar, Alert, ActivityIndicator, ScrollView,
} from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { useTheme } from '../theme/ThemeContext';
import { gradients, spacing, fontSize, radius } from '../theme';
import { GlassCard } from '../components/GlassCard';
import { loadBaseUrl, setBaseUrl, checkHealth, getBaseUrl } from '../services/api';

export const SettingsScreen = ({ navigation }: any) => {
  const { c, mode, toggle } = useTheme();
  const [url, setUrl] = useState('');
  const [testing, setTesting] = useState(false);
  const [status, setStatus] = useState<'idle' | 'success' | 'error'>('idle');
  const [statusMsg, setStatusMsg] = useState('');

  useEffect(() => {
    loadBaseUrl().then(setUrl);
  }, []);

  const testConnection = async () => {
    setTesting(true);
    setStatus('idle');
    try {
      await setBaseUrl(url);
      const health = await checkHealth();
      if (health.models_loaded) {
        setStatus('success');
        setStatusMsg(`Connected! Device: ${health.device}`);
      } else {
        setStatus('error');
        setStatusMsg('Server running but models not loaded yet.');
      }
    } catch (err: any) {
      setStatus('error');
      setStatusMsg(err.message || 'Cannot reach server.');
    } finally {
      setTesting(false);
    }
  };

  const saveUrl = async () => {
    await setBaseUrl(url);
    Alert.alert('Saved', `API URL set to ${url}`);
  };

  return (
    <SafeAreaView style={[styles.safe, { backgroundColor: c.bg }]}>
      <StatusBar barStyle={mode === 'dark' ? 'light-content' : 'dark-content'} />
      <ScrollView contentContainerStyle={styles.scroll}>
        <Text style={[styles.title, { color: c.text }]}>Settings</Text>

        {/* API URL */}
        <GlassCard>
          <Text style={[styles.label, { color: c.textMuted }]}>API SERVER URL</Text>
          <TextInput
            value={url}
            onChangeText={setUrl}
            placeholder="http://10.0.2.2:8000"
            placeholderTextColor={c.textMuted}
            style={[styles.input, { color: c.text, borderColor: c.border, backgroundColor: c.text + '06' }]}
            autoCapitalize="none"
            autoCorrect={false}
            keyboardType="url"
          />

          <View style={styles.btnRow}>
            <TouchableOpacity onPress={saveUrl} style={[styles.saveBtn, { backgroundColor: c.blue + '20' }]}>
              <Text style={[styles.saveBtnText, { color: c.blue }]}>💾 Save</Text>
            </TouchableOpacity>

            <TouchableOpacity onPress={testConnection} disabled={testing} activeOpacity={0.85}>
              <LinearGradient colors={gradients.taal as any} style={styles.testBtn}>
                {testing ? (
                  <ActivityIndicator color="#fff" size="small" />
                ) : (
                  <Text style={styles.testBtnText}>🔗 Test Connection</Text>
                )}
              </LinearGradient>
            </TouchableOpacity>
          </View>

          {status !== 'idle' && (
            <View style={[styles.statusBox, { backgroundColor: status === 'success' ? c.green + '12' : c.rose + '12', borderColor: status === 'success' ? c.green + '25' : c.rose + '25' }]}>
              <Text style={[styles.statusText, { color: status === 'success' ? c.green : c.rose }]}>
                {status === 'success' ? '✅' : '❌'} {statusMsg}
              </Text>
            </View>
          )}
        </GlassCard>

        {/* Theme */}
        <GlassCard>
          <Text style={[styles.label, { color: c.textMuted }]}>APPEARANCE</Text>
          <TouchableOpacity onPress={toggle} style={[styles.themeRow, { backgroundColor: c.text + '06' }]}>
            <Text style={{ fontSize: 22 }}>{mode === 'dark' ? '🌙' : '☀️'}</Text>
            <View style={{ flex: 1, marginLeft: spacing.md }}>
              <Text style={[styles.themeLabel, { color: c.text }]}>{mode === 'dark' ? 'Dark Mode' : 'Light Mode'}</Text>
              <Text style={[styles.themeSub, { color: c.textMuted }]}>Tap to switch theme</Text>
            </View>
            <Text style={[styles.themeToggle, { color: c.textSecondary }]}>Switch →</Text>
          </TouchableOpacity>
        </GlassCard>

        {/* About */}
        <GlassCard>
          <Text style={[styles.label, { color: c.textMuted }]}>ABOUT</Text>
          <Text style={[styles.about, { color: c.textSecondary }]}>
            Tabla Identifier uses deep learning to recognize Tabla playing. It can identify{' '}
            <Text style={{ color: c.teal, fontWeight: '700' }}>8 Taals</Text> and{' '}
            <Text style={{ color: c.amber, fontWeight: '700' }}>8 Bols</Text> with{' '}
            <Text style={{ color: c.green, fontWeight: '700' }}>99.38%</Text> Bol accuracy and{' '}
            <Text style={{ color: c.green, fontWeight: '700' }}>100%</Text> Taal accuracy.
          </Text>
          <View style={styles.statsRow}>
            {[
              { val: '99.38%', label: 'Bol Accuracy', color: c.amber },
              { val: '100%', label: 'Taal Accuracy', color: c.teal },
              { val: '4', label: 'Models', color: c.purple },
            ].map((stat, i) => (
              <View key={i} style={[styles.stat, { borderColor: c.border }]}>
                <Text style={[styles.statVal, { color: stat.color }]}>{stat.val}</Text>
                <Text style={[styles.statLabel, { color: c.textMuted }]}>{stat.label}</Text>
              </View>
            ))}
          </View>
        </GlassCard>
      </ScrollView>
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  safe: { flex: 1 },
  scroll: { padding: spacing.xl, paddingBottom: 40 },
  title: { fontSize: fontSize.xxl, fontWeight: '800', marginBottom: spacing.xl },
  label: { fontSize: 9, fontWeight: '700', letterSpacing: 1.5, marginBottom: spacing.md },
  input: { borderWidth: 1, borderRadius: radius.sm, padding: spacing.md, fontSize: fontSize.body, marginBottom: spacing.md },
  btnRow: { flexDirection: 'row', gap: spacing.sm },
  saveBtn: { paddingHorizontal: 16, paddingVertical: 10, borderRadius: radius.sm },
  saveBtnText: { fontSize: fontSize.sm, fontWeight: '600' },
  testBtn: { paddingHorizontal: 16, paddingVertical: 10, borderRadius: radius.sm },
  testBtnText: { color: '#fff', fontSize: fontSize.sm, fontWeight: '600' },
  statusBox: { marginTop: spacing.md, padding: spacing.md, borderRadius: radius.sm, borderWidth: 1 },
  statusText: { fontSize: fontSize.sm },
  themeRow: { flexDirection: 'row', alignItems: 'center', padding: spacing.md, borderRadius: radius.sm },
  themeLabel: { fontSize: fontSize.body, fontWeight: '600' },
  themeSub: { fontSize: fontSize.xs, marginTop: 2 },
  themeToggle: { fontSize: fontSize.sm, fontWeight: '600' },
  about: { fontSize: fontSize.md, lineHeight: 22, marginBottom: spacing.lg },
  statsRow: { flexDirection: 'row', gap: spacing.sm },
  stat: { flex: 1, alignItems: 'center', paddingVertical: spacing.md, borderRadius: radius.sm, borderWidth: 1 },
  statVal: { fontSize: fontSize.xl, fontWeight: '800' },
  statLabel: { fontSize: fontSize.xs, marginTop: 2 },
});
