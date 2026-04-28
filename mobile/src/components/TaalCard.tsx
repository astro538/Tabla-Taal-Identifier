import React from 'react';
import { View, Text, StyleSheet } from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { useTheme } from '../theme/ThemeContext';
import { gradients, spacing, fontSize, radius } from '../theme';
import { TaalResult } from '../types';
import { GlassCard } from './GlassCard';

interface Props {
  taal: TaalResult;
  theka: string;
  processingTime: number;
}

export const TaalCard: React.FC<Props> = ({ taal, theka, processingTime }) => {
  const { c } = useTheme();
  const top = taal.top_k[0];
  const confPct = (taal.confidence * 100).toFixed(1);

  return (
    <GlassCard>
      <View style={styles.header}>
        <LinearGradient colors={gradients.taal as any} start={{ x: 0, y: 0 }} end={{ x: 1, y: 0 }} style={styles.badge}>
          <Text style={styles.badgeText}>TAAL DETECTED</Text>
        </LinearGradient>
        <Text style={[styles.time, { color: c.textMuted }]}>{processingTime.toFixed(0)}ms</Text>
      </View>

      <View style={styles.main}>
        <View style={{ flex: 1 }}>
          <Text style={[styles.taalName, { color: c.text }]}>{taal.display_name}</Text>
          {top.beats && (
            <Text style={[styles.sub, { color: c.textSecondary }]}>
              {top.beats} beats · Vibhag: {top.vibhag}
            </Text>
          )}
        </View>
        <LinearGradient colors={gradients.primary as any} start={{ x: 0, y: 0 }} end={{ x: 1, y: 1 }} style={styles.confBadge}>
          <Text style={styles.confText}>{confPct}%</Text>
        </LinearGradient>
      </View>

      <Text style={[styles.thekaLabel, { color: c.textMuted }]}>THEKA (STANDARD PATTERN)</Text>
      <View style={[styles.thekaBox, { backgroundColor: c.teal + '12', borderColor: c.teal + '25' }]}>
        <Text style={[styles.thekaText, { color: c.teal }]}>{theka}</Text>
      </View>

      {/* Confidence Bars */}
      <View style={styles.bars}>
        {taal.top_k.slice(0, 5).map((item, i) => {
          const pct = item.confidence * 100;
          const barW = Math.max(2, (item.confidence / taal.top_k[0].confidence) * 100);
          return (
            <View key={item.label} style={styles.barRow}>
              <Text style={[styles.barLabel, { color: c.textSecondary }]} numberOfLines={1}>
                {item.display_name}
              </Text>
              <View style={[styles.barBg, { backgroundColor: c.text + '08' }]}>
                <LinearGradient
                  colors={gradients.taal as any}
                  start={{ x: 0, y: 0 }}
                  end={{ x: 1, y: 0 }}
                  style={[styles.barFill, { width: `${barW}%` }]}
                />
              </View>
              <Text style={[styles.barVal, { color: c.textMuted }]}>{pct.toFixed(1)}%</Text>
            </View>
          );
        })}
      </View>
    </GlassCard>
  );
};

const styles = StyleSheet.create({
  header: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', marginBottom: spacing.md },
  badge: { paddingHorizontal: 12, paddingVertical: 5, borderRadius: radius.full },
  badgeText: { color: '#fff', fontSize: 9, fontWeight: '700', letterSpacing: 1.2 },
  time: { fontSize: fontSize.xs },
  main: { flexDirection: 'row', alignItems: 'center', marginBottom: spacing.lg },
  taalName: { fontSize: fontSize.xl, fontWeight: '800' },
  sub: { fontSize: fontSize.sm, marginTop: 2 },
  confBadge: { paddingHorizontal: 14, paddingVertical: 8, borderRadius: radius.md },
  confText: { color: '#fff', fontSize: fontSize.xl, fontWeight: '800' },
  thekaLabel: { fontSize: 9, fontWeight: '700', letterSpacing: 1.5, marginBottom: spacing.xs },
  thekaBox: { padding: spacing.md, borderRadius: radius.sm, borderWidth: 1, marginBottom: spacing.lg },
  thekaText: { fontSize: fontSize.md, fontWeight: '600', letterSpacing: 0.3 },
  bars: { gap: 6 },
  barRow: { flexDirection: 'row', alignItems: 'center', gap: 8 },
  barLabel: { width: 80, fontSize: fontSize.sm, fontWeight: '500', textAlign: 'right' },
  barBg: { flex: 1, height: 6, borderRadius: 3, overflow: 'hidden' },
  barFill: { height: '100%', borderRadius: 3 },
  barVal: { width: 44, fontSize: fontSize.xs, fontWeight: '600' },
});
