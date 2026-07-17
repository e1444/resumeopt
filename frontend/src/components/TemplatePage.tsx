import { useEffect, useState } from 'react';
import { api } from '../api';
import { Spinner } from './Spinner';

const PLACEHOLDER = '[INSERT SKILLS HERE]';

export function TemplatePage() {
  const [content, setContent] = useState('');
  const [savedContent, setSavedContent] = useState('');
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [saved, setSaved] = useState(false);

  useEffect(() => {
    api
      .getTemplate()
      .then((text) => {
        setContent(text);
        setSavedContent(text);
      })
      .catch((err) => setError((err as Error).message))
      .finally(() => setLoading(false));
  }, []);

  const isDirty = content !== savedContent;

  // Warn on an actual browser close/refresh/tab-close if there are unsaved
  // edits. Switching between the app's OWN tabs doesn't trigger this - all
  // tabs stay mounted (see App.tsx) so in-progress edits simply persist in
  // this component's state instead of being lost, which is a more robust
  // fix than a confirm-before-switch prompt.
  useEffect(() => {
    if (!isDirty) return;
    const handler = (event: BeforeUnloadEvent) => {
      event.preventDefault();
      event.returnValue = '';
    };
    window.addEventListener('beforeunload', handler);
    return () => window.removeEventListener('beforeunload', handler);
  }, [isDirty]);

  const missingPlaceholder = !content.includes(PLACEHOLDER);

  const handleSave = async () => {
    setError(null);
    setSaved(false);
    if (missingPlaceholder) {
      setError(`Template must contain the placeholder ${PLACEHOLDER}`);
      return;
    }
    setSaving(true);
    try {
      await api.saveTemplate(content);
      setSavedContent(content);
      setSaved(true);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setSaving(false);
    }
  };

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => setContent(String(reader.result ?? ''));
    reader.readAsText(file);
    event.target.value = '';
  };

  return (
    <div className="page">
      <h2>Resume template</h2>
      <p className="hint">
        The LaTeX template used to render the tailored resume. Must keep the{' '}
        <code>{PLACEHOLDER}</code> placeholder where the skills section is injected.
      </p>

      {error && <div className="banner error">{error}</div>}
      {saved && !isDirty && <div className="banner success">Template saved.</div>}
      {isDirty && !loading && <div className="banner warning">Unsaved changes.</div>}
      {missingPlaceholder && !loading && (
        <div className="banner warning">
          Placeholder <code>{PLACEHOLDER}</code> is missing - saving is disabled until it's added back.
        </div>
      )}

      <div className="toolbar">
        <label className="file-upload">
          Upload .tex file
          <input type="file" accept=".tex" onChange={handleFileUpload} />
        </label>
        <button onClick={handleSave} disabled={loading || saving || missingPlaceholder || !isDirty}>
          {saving ? 'Saving…' : 'Save template'}
        </button>
      </div>

      {loading ? (
        <Spinner label="Loading template…" />
      ) : (
        <textarea
          className="code-editor"
          value={content}
          onChange={(event) => {
            setContent(event.target.value);
            setSaved(false);
          }}
          spellCheck={false}
          rows={24}
        />
      )}
    </div>
  );
}
