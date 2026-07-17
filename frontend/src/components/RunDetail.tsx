import { useEffect, useRef, useState } from 'react';
import {
  api,
  type MissingSkillsLog,
  type RunMetrics,
  type RunStatus,
  type SelectedSkill,
  type ValidationReportLog,
} from '../api';
import { computeProgressPercent } from '../progress';
import { Spinner } from './Spinner';

const POLL_INTERVAL_MS = 1500;

/** Shared run status/results view - used both right after triggering a run
 * and when selecting a past run from the history tab. */
export function RunDetail({ runId }: { runId: string }) {
  const [run, setRun] = useState<RunStatus | null>(null);
  const [missingSkills, setMissingSkills] = useState<string[]>([]);
  const [selectedSkills, setSelectedSkills] = useState<SelectedSkill[]>([]);
  const [promoting, setPromoting] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const pollHandle = useRef<number | null>(null);

  useEffect(() => {
    setRun(null);
    setMissingSkills([]);
    setSelectedSkills([]);
    setError(null);
    refreshStatus();
    return () => {
      if (pollHandle.current) window.clearInterval(pollHandle.current);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [runId]);

  const refreshStatus = async () => {
    try {
      const status = await api.getRun(runId);
      setRun(status);
      if (status.status === 'running') {
        schedulePoll();
      } else if (status.status === 'completed') {
        loadCompletionDetails();
      }
    } catch (err) {
      setError((err as Error).message);
    }
  };

  const schedulePoll = () => {
    if (pollHandle.current) window.clearInterval(pollHandle.current);
    pollHandle.current = window.setInterval(async () => {
      try {
        const status = await api.getRun(runId);
        setRun(status);
        if (status.status !== 'running') {
          window.clearInterval(pollHandle.current!);
          pollHandle.current = null;
          if (status.status === 'completed') loadCompletionDetails();
        }
      } catch (err) {
        setError((err as Error).message);
        window.clearInterval(pollHandle.current!);
        pollHandle.current = null;
      }
    }, POLL_INTERVAL_MS);
  };

  const loadCompletionDetails = async () => {
    try {
      const log = await api.getRunLog<MissingSkillsLog>(runId, 'missing_skills.json');
      setMissingSkills(log.missing_skills);
    } catch {
      setMissingSkills([]);
    }
    try {
      const report = await api.getRunLog<ValidationReportLog>(runId, 'validation_report.json');
      setSelectedSkills(report.selected_skills ?? []);
    } catch {
      setSelectedSkills([]);
    }
  };

  const handlePromote = async (term: string) => {
    setPromoting(term);
    setError(null);
    try {
      await api.promoteMissingSkill(runId, term);
      setMissingSkills((current) => current.filter((item) => item !== term));
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setPromoting(null);
    }
  };

  if (!run) {
    return <Spinner label="Loading run…" />;
  }

  return (
    <div className="run-detail">
      <h3>
        Run <code>{run.run_id}</code> - <StatusBadge status={run.status} />
      </h3>

      {error && <div className="banner error">{error}</div>}

      {run.status === 'running' && (
        <ProgressBar
          stage={run.current_stage}
          stageIndex={run.stage_index}
          stageTotal={run.stage_total}
          substage={run.substage}
          substageCompleted={run.substage_completed}
          substageTotal={run.substage_total}
        />
      )}

      {run.status === 'failed' && (
        <pre className="error-detail">{run.error ?? 'Unknown error'}</pre>
      )}

      {run.status === 'completed' && (
        <>
          {run.metrics && <UsageSummary metrics={run.metrics} />}

          <div className="pdf-preview">
            <iframe title="Tailored resume" src={api.runPdfUrl(run.run_id)} />
          </div>

          <div className="skill-provenance">
            <h4>Selected skills ({selectedSkills.length})</h4>
            <div className="table-wrapper">
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Skill</th>
                    <th>Match</th>
                    <th>Confidence</th>
                    <th>Evidence</th>
                  </tr>
                </thead>
                <tbody>
                  {selectedSkills.map((skill) => (
                    <tr key={skill.canonical_name}>
                      <td>{skill.canonical_name}</td>
                      <td>{skill.match_type}</td>
                      <td>{skill.confidence?.toFixed?.(2) ?? '—'}</td>
                      <td className="evidence-cell">{skill.evidence}</td>
                    </tr>
                  ))}
                  {selectedSkills.length === 0 && (
                    <tr>
                      <td colSpan={4}>
                        <em>No selected-skill detail available.</em>
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          </div>

          <div className="missing-skills">
            <h4>Missing skills ({missingSkills.length})</h4>
            <p className="hint">
              Extracted from the posting but not in the skills cache. Promote the ones you
              actually want tracked.
            </p>
            <ul>
              {missingSkills.map((term) => (
                <li key={term}>
                  <span>{term}</span>
                  <button
                    className="link-button"
                    disabled={promoting === term}
                    onClick={() => handlePromote(term)}
                  >
                    {promoting === term ? 'Promoting…' : 'Promote to cache'}
                  </button>
                </li>
              ))}
              {missingSkills.length === 0 && (
                <li>
                  <em>None.</em>
                </li>
              )}
            </ul>
          </div>

          <details>
            <summary>Advanced: raw run metrics</summary>
            <pre className="metrics">{JSON.stringify(run.metrics, null, 2)}</pre>
          </details>
        </>
      )}
    </div>
  );
}

function StatusBadge({ status }: { status: RunStatus['status'] }) {
  return <span className={`badge ${status}`}>{status}</span>;
}

const STAGE_LABELS: Record<string, string> = {
  read_posting: 'Reading posting',
  init_llm_provider: 'Initializing LLM providers',
  parse_posting: 'Extracting & matching skills',
  validate_selected_skills: 'Validating selected skills',
  group_skills: 'Grouping into sections',
  rendering: 'Rendering & compiling PDF',
  finalizing: 'Finalizing',
};

const SUBSTAGE_LABELS: Record<string, string> = {
  chunk_screening: 'screening chunks',
  extraction: 'extracting candidates',
  categorization: 'categorizing candidates',
  atomicity: 'checking keyword atomicity',
  redundancy: 'checking redundancy',
};

/** Real, stage-based progress bar - falls back to an indeterminate sliding
 * animation only if the backend hasn't reported a recognized stage yet
 * (e.g. the very first moment right after a run starts). When the backend
 * also reports batch-level substage progress (currently only within the long
 * `parse_posting` stage), that fractional progress is blended into the
 * overall bar width and appended to the label, so the bar keeps moving
 * smoothly through what would otherwise be one single static segment. The
 * actual percent math lives in `computeProgressPercent` (`../progress.ts`),
 * kept as a pure function so it can be unit-tested by simulating a whole
 * run's worth of status snapshots without a live backend/LLM calls. */
function ProgressBar({
  stage,
  stageIndex,
  stageTotal,
  substage,
  substageCompleted,
  substageTotal,
}: {
  stage?: string;
  stageIndex?: number;
  stageTotal?: number;
  substage?: string;
  substageCompleted?: number;
  substageTotal?: number;
}) {
  // Persists across re-renders for as long as this ProgressBar stays
  // mounted (i.e. for the lifetime of one running run - RunDetail unmounts
  // it once the run leaves the "running" status), so it can enforce a
  // monotonically non-decreasing bar width.
  const maxPercentRef = useRef(0);
  const { percent } = computeProgressPercent(
    { stage, stageIndex, stageTotal, substage, substageCompleted, substageTotal },
    maxPercentRef.current,
  );
  if (percent !== null) {
    maxPercentRef.current = percent;
  }
  const substageKnown =
    stage !== undefined &&
    stageIndex !== undefined &&
    !!stageTotal &&
    substage !== undefined &&
    substageCompleted !== undefined &&
    !!substageTotal;
  const label = stage ? STAGE_LABELS[stage] ?? stage : undefined;
  const substageLabel = substageKnown ? SUBSTAGE_LABELS[substage!] ?? substage : undefined;

  return (
    <div className="progress-wrap">
      <div
        className="progress-bar"
        role="progressbar"
        aria-label="Run in progress"
        aria-valuenow={percent ?? undefined}
        aria-valuemin={0}
        aria-valuemax={100}
      >
        {percent !== null ? (
          <div className="progress-bar-fill determinate" style={{ width: `${percent}%` }} />
        ) : (
          <div className="progress-bar-fill" />
        )}
      </div>
      {label && (
        <p className="hint progress-label">
          {label}
          {percent !== null && ` (${stageIndex! + 1}/${stageTotal})`}
          {substageKnown && ` - ${substageLabel} batch ${substageCompleted}/${substageTotal}`}
        </p>
      )}
    </div>
  );
}

function UsageSummary({ metrics }: { metrics: RunMetrics }) {
  const byRole = metrics.llm_usage?.by_role ?? {};
  const roles = Object.entries(byRole);
  if (roles.length === 0) return null;

  return (
    <div className="usage-summary">
      <h4>LLM usage</h4>
      <div className="table-wrapper">
        <table className="data-table">
          <thead>
            <tr>
              <th>Role</th>
              <th>Model</th>
              <th>Calls</th>
              <th>Tokens</th>
            </tr>
          </thead>
          <tbody>
            {roles.map(([role, usage]) => (
              <tr key={role}>
                <td>{role}</td>
                <td>{usage.model ?? '—'}</td>
                <td>{usage.call_count ?? '—'}</td>
                <td>{usage.total_tokens ?? '—'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {metrics.llm_usage?.combined && (
        <p className="hint">
          Total: {metrics.llm_usage.combined.call_count} calls,{' '}
          {metrics.llm_usage.combined.total_tokens} tokens
        </p>
      )}
    </div>
  );
}
