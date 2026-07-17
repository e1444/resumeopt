import { useState } from 'react';
import { api, type RunOptions } from '../api';
import { RunDetail } from './RunDetail';

const DRAFT_KEY = 'resumeopt.postingDraft';

const DEFAULT_OPTIONS: Required<RunOptions> = {
  provider: 'openai',
  model: 'gpt-4o',
  reasoning_model: 'gpt-5-mini',
  screening_model: 'gpt-4o-mini',
  use_llm_parser: true,
  max_concurrency: 24,
};

export function RunPage() {
  const [postingText, setPostingText] = useState(() => localStorage.getItem(DRAFT_KEY) ?? '');
  const [options, setOptions] = useState<Required<RunOptions>>(DEFAULT_OPTIONS);
  const [showOptions, setShowOptions] = useState(false);
  const [runId, setRunId] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handlePostingChange = (value: string) => {
    setPostingText(value);
    localStorage.setItem(DRAFT_KEY, value);
  };

  const updateOption = <K extends keyof Required<RunOptions>>(key: K, value: Required<RunOptions>[K]) => {
    setOptions((current) => ({ ...current, [key]: value }));
  };

  const handleRun = async (event: React.FormEvent) => {
    event.preventDefault();
    if (!postingText.trim()) return;
    setError(null);
    setSubmitting(true);
    try {
      const { run_id } = await api.startRun(postingText, options);
      setRunId(run_id);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setSubmitting(false);
    }
  };

  const handleTextareaKeyDown = (event: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if ((event.metaKey || event.ctrlKey) && event.key === 'Enter') {
      event.preventDefault();
      event.currentTarget.form?.requestSubmit();
    }
  };

  return (
    <div className="page">
      <h2>Tailor a resume</h2>
      <p className="hint">
        Paste the job posting text below (no URLs - copy/paste only). Press{' '}
        <strong>Cmd/Ctrl+Enter</strong> to run.
      </p>

      {error && <div className="banner error">{error}</div>}

      <form onSubmit={handleRun}>
        <textarea
          className="posting-input"
          placeholder="Paste the job posting here…"
          value={postingText}
          onChange={(event) => handlePostingChange(event.target.value)}
          onKeyDown={handleTextareaKeyDown}
          rows={12}
        />

        <details className="config-picker" open={showOptions} onToggle={(e) => setShowOptions(e.currentTarget.open)}>
          <summary>Advanced options (provider &amp; models)</summary>
          <div className="config-grid">
            <label>
              Provider
              <select
                value={options.provider}
                onChange={(event) => updateOption('provider', event.target.value)}
              >
                <option value="openai">openai</option>
                <option value="anthropic">anthropic</option>
                <option value="ollama">ollama</option>
              </select>
            </label>
            <label>
              Judge model
              <input
                type="text"
                value={options.model}
                onChange={(event) => updateOption('model', event.target.value)}
              />
            </label>
            <label>
              Reasoning model
              <input
                type="text"
                value={options.reasoning_model}
                onChange={(event) => updateOption('reasoning_model', event.target.value)}
              />
            </label>
            <label>
              Screening model
              <input
                type="text"
                value={options.screening_model}
                onChange={(event) => updateOption('screening_model', event.target.value)}
              />
            </label>
            <label>
              Max concurrency
              <input
                type="number"
                min={1}
                max={64}
                value={options.max_concurrency}
                onChange={(event) => updateOption('max_concurrency', Number(event.target.value) || 1)}
              />
            </label>
            <label className="checkbox-label">
              <input
                type="checkbox"
                checked={options.use_llm_parser}
                onChange={(event) => updateOption('use_llm_parser', event.target.checked)}
              />
              Use LLM parser (uncheck for deterministic-only, no LLM calls)
            </label>
          </div>
        </details>

        <div className="toolbar">
          <button type="submit" disabled={submitting || !postingText.trim()}>
            {submitting ? 'Starting…' : 'Run'}
          </button>
        </div>
      </form>

      {runId && <RunDetail runId={runId} />}
    </div>
  );
}
