import { Dimensions } from 'react-native';

const { width, height } = Dimensions.get('window');

export const colors = {
  dark: {
    bg: '#06070A',
    bgSecondary: '#0C0E15',
    card: 'rgba(15, 18, 30, 0.85)',
    cardSolid: '#10132A',
    border: 'rgba(100, 120, 200, 0.12)',
    borderActive: 'rgba(100, 140, 255, 0.35)',
    text: '#E8EAF0',
    textSecondary: '#8B92A8',
    textMuted: '#555B70',
    blue: '#4A7CFF',
    purple: '#8B5CF6',
    teal: '#14B8A6',
    amber: '#F59E0B',
    rose: '#F43F5E',
    green: '#22C55E',
    glassBg: 'rgba(20, 24, 45, 0.65)',
    glassBorder: 'rgba(100, 120, 200, 0.18)',
    overlay: 'rgba(0, 0, 0, 0.5)',
  },
  light: {
    bg: '#F5F7FA',
    bgSecondary: '#FFFFFF',
    card: 'rgba(255, 255, 255, 0.85)',
    cardSolid: '#FFFFFF',
    border: 'rgba(0, 0, 50, 0.08)',
    borderActive: 'rgba(74, 124, 255, 0.3)',
    text: '#1A1D2E',
    textSecondary: '#5A6178',
    textMuted: '#9CA3AF',
    blue: '#4A7CFF',
    purple: '#8B5CF6',
    teal: '#0D9488',
    amber: '#D97706',
    rose: '#E11D48',
    green: '#16A34A',
    glassBg: 'rgba(255, 255, 255, 0.7)',
    glassBorder: 'rgba(0, 0, 50, 0.1)',
    overlay: 'rgba(0, 0, 0, 0.3)',
  },
};

export const gradients = {
  primary: ['#4A7CFF', '#8B5CF6'],
  taal: ['#14B8A6', '#4A7CFF'],
  bol: ['#F59E0B', '#F43F5E'],
  success: ['#22C55E', '#14B8A6'],
  darkCard: ['rgba(15, 18, 30, 0.9)', 'rgba(10, 12, 20, 0.95)'],
  lightCard: ['rgba(255, 255, 255, 0.9)', 'rgba(245, 247, 250, 0.95)'],
};

export const sourceColors = {
  both: '#22C55E',
  template: '#14B8A6',
  model: '#F59E0B',
};

export const spacing = {
  xs: 4,
  sm: 8,
  md: 12,
  lg: 16,
  xl: 20,
  xxl: 28,
  xxxl: 36,
};

export const radius = {
  sm: 8,
  md: 12,
  lg: 16,
  xl: 20,
  full: 999,
};

export const fontSize = {
  xs: 10,
  sm: 12,
  md: 14,
  body: 15,
  lg: 17,
  xl: 22,
  xxl: 28,
  hero: 34,
};

export const screen = { width, height };

export type ThemeMode = 'dark' | 'light';
export type ThemeColors = typeof colors.dark;
