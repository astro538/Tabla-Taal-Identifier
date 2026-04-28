import React, { useEffect } from 'react';
import { ThemeProvider } from './src/theme/ThemeContext';
import { AppNavigator } from './src/navigation/AppNavigator';
import { loadBaseUrl } from './src/services/api';

const AppContent = () => {
  useEffect(() => {
    loadBaseUrl();
  }, []);

  return <AppNavigator />;
};

export default function App() {
  return (
    <ThemeProvider>
      <AppContent />
    </ThemeProvider>
  );
}
