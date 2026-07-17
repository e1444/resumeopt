import { useEffect, useRef, useState } from 'react';
import { api, type RunSummary } from '../api';
import { RunDetail } from './RunDetail';
import { Spinner } from './Spinner';

export function RunHistoryPage() {
  const [runs, setRuns] = useState<RunSummary[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const detailRef = useRef<HTMLDivElement | null>(null);

  const reload = async () => {
    setLoading(true);
    setError(null);
    try {
      setRuns(await api.listRuns());
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    reload();
  }, []);

  useEffect(() => {
    if (selected) {
      detailRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  }, [selected]);

  return (
    <div className="page">
      <h2>Run history</h2>
      <p className="hint">
        Runs triggered from this browser session (in-memory only - restarting the backend
        clears this list; the underlying run folders under <code>build/</code> are still on
        disk).
      </p>

      {error && <div className="banner error">{error}</div>}

      <div className="toolbar">
        <button onClick={reload} disabled={loading}>
          {loading ? 'Refreshing…' : 'Refresh'}
        </button>
      </div>

      {loading && runs.length === 0 ? (
        <Spinner label="Loading run history…" />
      ) : (
        <div className="table-wrapper">
          <table className="data-table">
            <thead>
              <tr>
                <th>Run</th>
                <th>Status</th>
                <th>Created</th>
                <th></th>
              </tr>
            </thead>
            <tbody>
              {runs.map((run) => (
                <tr key={run.run_id} className={selected === run.run_id ? 'selected-row' : ''}>
                  <td>
                    <code>{run.run_id}</code>
                  </td>
                  <td>
                    <span className={`badge ${run.status}`}>{run.status}</span>
                  </td>
                  <td>{new Date(run.created_at).toLocaleString()}</td>
                  <td>
                    <button className="link-button" onClick={() => setSelected(run.run_id)}>
                      View
                    </button>
                  </td>
                </tr>
              ))}
              {runs.length === 0 && !loading && (
                <tr>
                  <td colSpan={4}>
                    <em>No runs yet.</em>
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      )}

      {selected && (
        <div ref={detailRef}>
          <RunDetail runId={selected} />
        </div>
      )}
    </div>
  );
}
