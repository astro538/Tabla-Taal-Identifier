import React, { useState, useCallback } from 'react';
import {
  View, Text, StyleSheet, SafeAreaView, FlatList, TouchableOpacity, StatusBar, Alert,
} from 'react-native';
import { useFocusEffect } from '@react-navigation/native';
import { useTheme } from '../theme/ThemeContext';
import { spacing, fontSize, radius, sourceColors } from '../theme';
import { HistoryItem } from '../types';
import { getHistory, clearHistory, deleteHistoryItem } from '../services/storage';
import { GlassCard } from '../components/GlassCard';

export const HistoryScreen = ({ navigation }: any) => {
  const { c, mode } = useTheme();
  const [history, setHistory] = useState<HistoryItem[]>([]);

  useFocusEffect(
    useCallback(() => {
      getHistory().then(setHistory);
    }, [])
  );

  const handleClear = () => {
    Alert.alert('Clear History', 'Delete all analysis history?', [
      { text: 'Cancel', style: 'cancel' },
      {
        text: 'Clear All',
        style: 'destructive',
        onPress: async () => {
          await clearHistory();
          setHistory([]);
        },
      },
    ]);
  };

  const handleDelete = async (id: string) => {
    await deleteHistoryItem(id);
    setHistory((prev) => prev.filter((h) => h.id !== id));
  };

  const formatDate = (iso: string) => {
    const d = new Date(iso);
    return d.toLocaleDateString('en-IN', { day: 'numeric', month: 'short', year: 'numeric', hour: '2-digit', minute: '2-digit' });
  };

  const renderItem = ({ item }: { item: HistoryItem }) => (
    <TouchableOpacity
      activeOpacity={0.85}
      onLongPress={() => {
        Alert.alert('Delete', 'Remove this item?', [
          { text: 'Cancel', style: 'cancel' },
          { text: 'Delete', style: 'destructive', onPress: () => handleDelete(item.id) },
        ]);
      }}
    >
      <GlassCard>
        <View style={styles.itemRow}>
          <View style={{ flex: 1 }}>
            <Text style={[styles.itemTaal, { color: c.text }]}>{item.taalName}</Text>
            <Text style={[styles.itemMeta, { color: c.textSecondary }]}>
              {item.bolCount} bols · {item.duration.toFixed(1)}s · {(item.taalConfidence * 100).toFixed(0)}% conf
            </Text>
            <Text style={[styles.itemDate, { color: c.textMuted }]}>{formatDate(item.date)}</Text>
          </View>
          <View style={[styles.confCircle, { borderColor: sourceColors.both }]}>
            <Text style={[styles.confVal, { color: c.text }]}>{(item.taalConfidence * 100).toFixed(0)}%</Text>
          </View>
        </View>
      </GlassCard>
    </TouchableOpacity>
  );

  return (
    <SafeAreaView style={[styles.safe, { backgroundColor: c.bg }]}>
      <StatusBar barStyle={mode === 'dark' ? 'light-content' : 'dark-content'} />
      <View style={styles.header}>
        <Text style={[styles.title, { color: c.text }]}>History</Text>
        {history.length > 0 && (
          <TouchableOpacity onPress={handleClear}>
            <Text style={[styles.clearBtn, { color: c.rose }]}>Clear All</Text>
          </TouchableOpacity>
        )}
      </View>

      {history.length === 0 ? (
        <View style={styles.empty}>
          <Text style={styles.emptyIcon}>📋</Text>
          <Text style={[styles.emptyText, { color: c.textMuted }]}>No analysis history yet</Text>
          <Text style={[styles.emptySub, { color: c.textMuted }]}>Record or upload audio to get started</Text>
        </View>
      ) : (
        <FlatList
          data={history}
          keyExtractor={(item) => item.id}
          renderItem={renderItem}
          contentContainerStyle={{ padding: spacing.xl }}
          showsVerticalScrollIndicator={false}
        />
      )}
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  safe: { flex: 1 },
  header: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', paddingHorizontal: spacing.xl, paddingTop: spacing.xl, paddingBottom: spacing.md },
  title: { fontSize: fontSize.xxl, fontWeight: '800' },
  clearBtn: { fontSize: fontSize.sm, fontWeight: '600' },
  itemRow: { flexDirection: 'row', alignItems: 'center' },
  itemTaal: { fontSize: fontSize.lg, fontWeight: '700' },
  itemMeta: { fontSize: fontSize.sm, marginTop: 3 },
  itemDate: { fontSize: fontSize.xs, marginTop: 3 },
  confCircle: { width: 50, height: 50, borderRadius: 25, borderWidth: 2, alignItems: 'center', justifyContent: 'center' },
  confVal: { fontSize: fontSize.sm, fontWeight: '800' },
  empty: { flex: 1, alignItems: 'center', justifyContent: 'center', paddingBottom: 80 },
  emptyIcon: { fontSize: 48, marginBottom: spacing.md },
  emptyText: { fontSize: fontSize.lg, fontWeight: '600' },
  emptySub: { fontSize: fontSize.sm, marginTop: 4 },
});
