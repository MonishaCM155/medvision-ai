import React, { createContext, useContext, useEffect, useMemo, useState, useCallback, useRef } from 'react';
import { AppNotification, EngineStatus, SessionUser } from '../types';
import { DEFAULT_USER, SEED_NOTIFICATIONS } from '../data/mockEnterprise';
import { api } from '../services/api';
import { uid } from '../utils/format';

export interface Toast {
  id: string;
  kind: 'success' | 'info' | 'warning' | 'critical';
  title: string;
  body?: string;
}

type ThemeMode = 'light' | 'dark';

interface AppContextValue {
  theme: ThemeMode;
  toggleTheme: () => void;
  setTheme: (mode: ThemeMode) => void;
  /** Static public-research identity — MedVision AI has no user accounts. */
  user: SessionUser;
  engine: EngineStatus | null;
  refreshEngine: () => Promise<EngineStatus>;
  notifications: AppNotification[];
  unreadCount: number;
  markAllRead: () => void;
  markRead: (id: string) => void;
  pushNotification: (n: Omit<AppNotification, 'id' | 'read' | 'time'>) => void;
  dismissNotification: (id: string) => void;
  toasts: Toast[];
  dismissToast: (id: string) => void;
  sidebarCollapsed: boolean;
  toggleSidebar: () => void;
  /** Minimum AI confidence for a definitive diagnosis (0.5–0.95). */
  confidenceThreshold: number;
  setConfidenceThreshold: (value: number) => void;
}

const AppContext = createContext<AppContextValue | null>(null);

function getInitialTheme(): ThemeMode {
  try {
    const saved = localStorage.getItem('medvision-theme');
    if (saved === 'dark' || saved === 'light') return saved;
  } catch {
    /* ignore */
  }
  return 'light';
}

const DEFAULT_CONFIDENCE_THRESHOLD = 0.75;

function getInitialThreshold(): number {
  try {
    const saved = Number(localStorage.getItem('medvision-confidence-threshold'));
    if (Number.isFinite(saved) && saved >= 0.5 && saved <= 0.95) return saved;
  } catch {
    /* ignore */
  }
  return DEFAULT_CONFIDENCE_THRESHOLD;
}

export const AppProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [theme, setThemeState] = useState<ThemeMode>(getInitialTheme);
  // Static public-research identity — MedVision AI has no accounts, so there is
  // no user state to mutate (deliberately not a useState).
  const user = DEFAULT_USER;
  const [notifications, setNotifications] = useState<AppNotification[]>(SEED_NOTIFICATIONS);
  const [toasts, setToasts] = useState<Toast[]>([]);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [engine, setEngine] = useState<EngineStatus | null>(null);
  const [confidenceThreshold, setConfidenceThresholdState] = useState<number>(getInitialThreshold);
  const pushCounter = useRef(0);

  // Probe the PyTorch engine on mount and keep the badge fresh
  useEffect(() => {
    api.getEngineStatus().then(setEngine).catch(() => setEngine(null));
    const poll = setInterval(() => {
      api.getEngineStatus().then(setEngine).catch(() => undefined);
    }, 45_000);
    return () => clearInterval(poll);
  }, []);

  const refreshEngine = useCallback(async () => {
    const status = await api.getEngineStatus();
    setEngine(status);
    return status;
  }, []);

  // Apply theme class to <html>
  useEffect(() => {
    const root = document.documentElement;
    root.classList.toggle('dark', theme === 'dark');
    try {
      localStorage.setItem('medvision-theme', theme);
    } catch {
      /* ignore */
    }
  }, [theme]);

  const setTheme = useCallback((mode: ThemeMode) => setThemeState(mode), []);
  const toggleTheme = useCallback(() => setThemeState((t) => (t === 'dark' ? 'light' : 'dark')), []);

  const setConfidenceThreshold = useCallback((value: number) => {
    const clamped = Math.min(0.95, Math.max(0.5, value));
    setConfidenceThresholdState(clamped);
    try {
      localStorage.setItem('medvision-confidence-threshold', String(clamped));
    } catch {
      /* ignore */
    }
  }, []);

  const markAllRead = useCallback(() => {
    setNotifications((prev) => prev.map((n) => ({ ...n, read: true })));
  }, []);

  const markRead = useCallback((id: string) => {
    setNotifications((prev) => prev.map((n) => (n.id === id ? { ...n, read: true } : n)));
  }, []);

  const dismissNotification = useCallback((id: string) => {
    setNotifications((prev) => prev.filter((n) => n.id !== id));
  }, []);

  const dismissToast = useCallback((id: string) => {
    setToasts((prev) => prev.filter((t) => t.id !== id));
  }, []);

  const pushNotification = useCallback((n: Omit<AppNotification, 'id' | 'read' | 'time'>) => {
    const toastId = uid('toast');
    setNotifications((prev) => [{ ...n, id: uid('notif'), read: false, time: 'just now' }, ...prev]);
    // Mirror as a visible toast (auto-dismiss handled by ToastHost)
    setToasts((prev) => [...prev.slice(-3), { id: toastId, kind: n.kind, title: n.title, body: n.body }]);
  }, []);

  // Simulated live feed: occasionally push a notification
  useEffect(() => {
    const samples: Omit<AppNotification, 'id' | 'read' | 'time'>[] = [
      { kind: 'info', title: 'Scan queued', body: 'CXR for PAT-220981 added to the inference queue (DenseNet-121).' },
      { kind: 'success', title: 'Report completed', body: 'AI report for PAT-447381 generated successfully.' },
      { kind: 'warning', title: 'Queue latency', body: 'Inference queue backlog growing. Auto-scaler engaging 1 GPU worker.' },
    ];
    const interval = setInterval(() => {
      pushCounter.current += 1;
      if (pushCounter.current % 4 === 0) {
        const sample = samples[Math.floor(Math.random() * samples.length)];
        pushNotification(sample);
      }
    }, 45000);
    return () => clearInterval(interval);
  }, [pushNotification]);

  const unreadCount = useMemo(() => notifications.filter((n) => !n.read).length, [notifications]);

  const value = useMemo<AppContextValue>(
    () => ({
      theme,
      toggleTheme,
      setTheme,
      user,
      engine,
      refreshEngine,
      notifications,
      unreadCount,
      markAllRead,
      markRead,
      pushNotification,
      dismissNotification,
      toasts,
      dismissToast,
      sidebarCollapsed,
      toggleSidebar: () => setSidebarCollapsed((c) => !c),
      confidenceThreshold,
      setConfidenceThreshold,
    }),
    [theme, toggleTheme, setTheme, user, engine, refreshEngine, notifications, unreadCount, markAllRead, markRead, pushNotification, dismissNotification, toasts, dismissToast, sidebarCollapsed, confidenceThreshold, setConfidenceThreshold]
  );

  return <AppContext.Provider value={value}>{children}</AppContext.Provider>;
};

export function useApp(): AppContextValue {
  const ctx = useContext(AppContext);
  if (!ctx) throw new Error('useApp must be used within AppProvider');
  return ctx;
}
