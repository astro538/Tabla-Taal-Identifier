import React, { useRef, useEffect, useState, useCallback } from 'react';
import { View, Text, StyleSheet, ScrollView, TouchableOpacity, Animated } from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { useTheme } from '../theme/ThemeContext';
import { gradients, spacing, fontSize, radius, sourceColors } from '../theme';
import { BolEntry } from '../types';
import { GlassCard } from './GlassCard';

interface Props {
  bols: BolEntry[];
  duration: number;
  currentTime: number;
  bolCount: number;
  cyclesDetected: number;
  patternLength: number;
  onSeek: (time: number) => void;
}

export const BolTimeline: React.FC<Props> = ({
  bols, duration, currentTime, bolCount, cyclesDetected, patternLength, onSeek,
}) => {
  const { c } = useTheme();
  const scrollRef = useRef<ScrollView>(null);
  const [activeIndex, setActiveIndex] = useState(-1);

  // Find active bol
  useEffect(() => {
    let idx = -1;
    for (let i = bols.length - 1; i >= 0; i--) {
      if (currentTime >= bols[i].time - 0.02) { idx = i; break; }
    }
    if (idx !== activeIndex) setActiveIndex(idx);
  }, [currentTime, bols]);

  const getSourceColor = (source: string) => sourceColors[source as keyof typeof sourceColors] || c.textMuted;
  const getSourceLabel = (source: string) => source === 'both' ? '✓' : source === 'template' ? 'T' : 'M';

  return (
    <GlassCard>
      <View style={styles.header}>
        <LinearGradient colors={gradients.bol as any} start={{ x: 0, y: 0 }} end={{ x: 1, y: 0 }} style={styles.badge}>
          <Text style={styles.badgeText}>BOL SEQUENCE</Text>
        </LinearGradient>
        <Text style={[styles.count, { color: c.textMuted }]}>
          {bolCount} strokes · {cyclesDetected} cycle{cyclesDetected !== 1 ? 's' : ''} · {patternLength} beats/cycle
        </Text>
      </View>

      {/* Legend */}
      <View style={styles.legend}>
        <View style={styles.legendItem}>
          <View style={[styles.legendDot, { backgroundColor: sourceColors.both }]} />
          <Text style={[styles.legendText, { color: c.textSecondary }]}>Both agree</Text>
        </View>
        <View style={styles.legendItem}>
          <View style={[styles.legendDot, { backgroundColor: sourceColors.template }]} />
          <Text style={[styles.legendText, { color: c.textSecondary }]}>Template</Text>
        </View>
        <View style={styles.legendItem}>
          <View style={[styles.legendDot, { backgroundColor: sourceColors.model }]} />
          <Text style={[styles.legendText, { color: c.textSecondary }]}>Model</Text>
        </View>
      </View>

      {/* Timeline Track */}
      <ScrollView horizontal showsHorizontalScrollIndicator={false} style={styles.trackScroll}>
        <View style={[styles.track, { backgroundColor: c.text + '06' }]}>
          {/* Progress overlay */}
          <View style={[styles.progress, {
            width: `${Math.min((currentTime / duration) * 100, 100)}%`,
            backgroundColor: c.amber + '15',
          }]} />
          {bols.map((bol, i) => {
            const left = (bol.time / duration) * 100;
            const width = Math.max(((bol.end_time - bol.time) / duration) * 100, 4);
            const isActive = i === activeIndex;
            const srcColor = getSourceColor(bol.source);

            return (
              <TouchableOpacity
                key={i}
                onPress={() => onSeek(bol.time)}
                activeOpacity={0.7}
                style={[
                  styles.bolSegment,
                  {
                    left: `${left}%`,
                    width: `${width}%`,
                    backgroundColor: isActive ? srcColor + '30' : 'transparent',
                    borderRightColor: c.text + '10',
                  },
                ]}
              >
                <Text style={[styles.bolName, { color: isActive ? '#fff' : c.text }]} numberOfLines={1}>
                  {bol.bol}
                </Text>
                <Text style={[styles.bolConf, { color: srcColor }]}>
                  {(bol.confidence * 100).toFixed(0)}%
                </Text>
                <Text style={[styles.bolTime, { color: c.textMuted }]}>
                  {bol.time.toFixed(2)}s
                </Text>
              </TouchableOpacity>
            );
          })}
        </View>
      </ScrollView>

      {/* Bol Chips with Cycle Dividers */}
      <View style={styles.chips}>
        {bols.map((bol, i) => {
          const isActive = i === activeIndex;
          const srcColor = getSourceColor(bol.source);
          const showCycleDivider = i === 0 || bol.cycle !== bols[i - 1].cycle;

          return (
            <React.Fragment key={i}>
              {showCycleDivider && (
                <View style={styles.cycleDivider}>
                  <Text style={[styles.cycleText, { color: c.textMuted }]}>Cycle {bol.cycle}</Text>
                  <View style={[styles.cycleLine, { backgroundColor: c.border }]} />
                </View>
              )}
              <TouchableOpacity
                onPress={() => onSeek(bol.time)}
                activeOpacity={0.7}
                style={[
                  styles.chip,
                  {
                    backgroundColor: isActive ? srcColor + '20' : c.text + '06',
                    borderColor: isActive ? srcColor : c.border,
                  },
                ]}
              >
                <Text style={[styles.chipName, { color: isActive ? c.text : c.textSecondary }]}>
                  {bol.bol}
                </Text>
                <View style={[styles.srcBadge, { backgroundColor: srcColor }]}>
                  <Text style={styles.srcBadgeText}>{getSourceLabel(bol.source)}</Text>
                </View>
              </TouchableOpacity>
            </React.Fragment>
          );
        })}
      </View>
    </GlassCard>
  );
};

const styles = StyleSheet.create({
  header: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', marginBottom: spacing.sm, flexWrap: 'wrap', gap: 6 },
  badge: { paddingHorizontal: 12, paddingVertical: 5, borderRadius: radius.full },
  badgeText: { color: '#fff', fontSize: 9, fontWeight: '700', letterSpacing: 1.2 },
  count: { fontSize: fontSize.xs },
  legend: { flexDirection: 'row', gap: 14, marginBottom: spacing.md, flexWrap: 'wrap' },
  legendItem: { flexDirection: 'row', alignItems: 'center', gap: 5 },
  legendDot: { width: 8, height: 8, borderRadius: 4 },
  legendText: { fontSize: fontSize.xs },
  trackScroll: { marginBottom: spacing.md },
  track: { height: 65, borderRadius: radius.sm, position: 'relative', minWidth: '100%', flexDirection: 'row' },
  progress: { position: 'absolute', top: 0, left: 0, height: '100%', borderRadius: radius.sm },
  bolSegment: {
    position: 'absolute', top: 0, height: '100%',
    alignItems: 'center', justifyContent: 'center',
    borderRightWidth: 1, paddingHorizontal: 2,
  },
  bolName: { fontSize: fontSize.sm, fontWeight: '700' },
  bolConf: { fontSize: 8, fontWeight: '600', marginTop: 1 },
  bolTime: { fontSize: 7, marginTop: 1 },
  chips: { flexDirection: 'row', flexWrap: 'wrap', gap: 6 },
  cycleDivider: { flexDirection: 'row', alignItems: 'center', gap: 6, width: '100%', paddingVertical: 4 },
  cycleText: { fontSize: 9, fontWeight: '700', letterSpacing: 1, textTransform: 'uppercase' },
  cycleLine: { flex: 1, height: 1 },
  chip: {
    paddingHorizontal: 12, paddingVertical: 6, borderRadius: radius.lg,
    borderWidth: 1, position: 'relative',
  },
  chipName: { fontSize: fontSize.sm, fontWeight: '600' },
  srcBadge: {
    position: 'absolute', top: -5, right: -4,
    paddingHorizontal: 4, paddingVertical: 1, borderRadius: 6,
  },
  srcBadgeText: { fontSize: 7, fontWeight: '800', color: '#fff' },
});
