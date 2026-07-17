import { useEffect, useState } from 'react';
import './App.css';
import { RunHistoryPage } from './components/RunHistoryPage';
import { RunPage } from './components/RunPage';
import { SkillsPage } from './components/SkillsPage';
import { TemplatePage } from './components/TemplatePage';

type Tab = 'run' | 'history' | 'skills' | 'template';
type Theme = 'light' | 'dark';

const THEME_KEY = 'resumeopt.theme';

function getInitialTheme(): Theme {
  const stored = localStorage.getItem(THEME_KEY);
  if (stored === 'light' || stored === 'dark') return stored;
  return window.matchMedia?.('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
}

function App() {
  const [tab, setTab] = useState<Tab>('run');
  const [theme, setTheme] = useState<Theme>(getInitialTheme);

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem(THEME_KEY, theme);
  }, [theme]);

  const toggleTheme = () => setTheme((current) => (current === 'dark' ? 'light' : 'dark'));

  const tabs: { id: Tab; label: string }[] = [
    { id: 'run', label: 'Tailor resume' },
    { id: 'history', label: 'History' },
    { id: 'skills', label: 'Skills cache' },
    { id: 'template', label: 'Template' },
  ];

  return (
    <div className="app">
      <header className="app-header">
        <h1>resumeopt</h1>
        <div className="app-header-right">
          <nav>
            {tabs.map(({ id, label }) => (
              <button
                key={id}
                aria-current={tab === id ? 'page' : undefined}
                onClick={() => setTab(id)}
              >
                {label}
              </button>
            ))}
          </nav>
          <button
            type="button"
            className="theme-toggle"
            onClick={toggleTheme}
            aria-label={`Switch to ${theme === 'dark' ? 'light' : 'dark'} mode`}
          >
            {theme === 'dark' ? '☀️ Light' : '🌙 Dark'}
          </button>
        </div>
      </header>

      {/* Every tab stays mounted (just hidden) rather than being unmounted
          on switch, so in-progress edits (e.g. an unsaved template draft)
          aren't silently lost when navigating away and back. */}
      <main>
        <div style={{ display: tab === 'run' ? 'block' : 'none' }}>
          <RunPage />
        </div>
        <div style={{ display: tab === 'history' ? 'block' : 'none' }}>
          <RunHistoryPage />
        </div>
        <div style={{ display: tab === 'skills' ? 'block' : 'none' }}>
          <SkillsPage />
        </div>
        <div style={{ display: tab === 'template' ? 'block' : 'none' }}>
          <TemplatePage />
        </div>
      </main>
    </div>
  );
}

export default App;
