import { useEffect, useRef, useState } from 'react';
import {
  api,
  type ConfirmedSkillsLog,
  type MissingSkillsLog,
  type ReviewableSkill,
  type RunMetrics,
  type RunStatus,
  type SelectedSkill,
  type SkillReview,
  type ValidationReportLog,
} from '../api';
import { computeProgressPercent } from '../progress';
import { Spinner } from './Spinner';

const POLL_INTERVAL_MS = 1500;

/** Shared run status/results view - used both right after triggering a run
 * and when selecting a past run from the history tab. */
export function RunDetail({ runId }: { runId: string }) {
  const [run, setRun] = useState<RunStatus | null>(null);
  const [postingText, setPostingText] = useState<string | null>(null);
  const [missingSkills, setMissingSkills] = useState<string[]>([]);
  const [selectedSkills, setSelectedSkills] = useState<SelectedSkill[]>([]);
  const [confirmedSkills, setConfirmedSkills] = useState<ConfirmedSkillsLog | null>(null);
  const [promoting, setPromoting] = useState<string | null>(null);
  const [reviewOpen, setReviewOpen] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const pollHandle = useRef<number | null>(null);

  useEffect(() => {
    setRun(null);
    setPostingText(null);
    setMissingSkills([]);
    setSelectedSkills([]);
    setConfirmedSkills(null);
    setReviewOpen(false);
    setError(null);
    refreshStatus();
    loadPostingText();
    return () => {
      if (pollHandle.current) window.clearInterval(pollHandle.current);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [runId]);

  const loadPostingText = async () => {
    try {
      setPostingText(await api.getRunPosting(runId));
    } catch {
      // Not fatal to the rest of the view (e.g. an older run predating this
      // feature, or the upload was cleaned up) - just omit the section.
      setPostingText(null);
    }
  };

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
    try {
      // Only present for runs that went through the Phase 9 review
      // checkpoint - absent (404) for older runs, which fall back to the
      // pre-review validation_report.json/missing_skills.json as-is.
      setConfirmedSkills(await api.getRunLog<ConfirmedSkillsLog>(runId, 'confirmed_skills.json'));
    } catch {
      setConfirmedSkills(null);
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

  const handleSkillsConfirmed = () => {
    setReviewOpen(false);
    refreshStatus();
  };

  if (!run) {
    return <Spinner label="Loading run…" />;
  }

  // `confirmed_skills.json` (only present for runs that went through the
  // Phase 9 review checkpoint) is the authoritative source for what's
  // actually on the rendered PDF - `validation_report.json`'s matches and
  // `missing_skills.json`'s candidates are both pre-review artifacts that
  // don't reflect always-include skills, a promoted missing skill, or an
  // unchecked skill. When available, use it to correct both displays;
  // otherwise (an older run) fall back to the pre-review data as-is.
  const finalSkillsLower = confirmedSkills
    ? new Set(confirmedSkills.final_skills.map((name) => name.toLowerCase()))
    : null;

  const displaySelectedSkills: SelectedSkill[] = finalSkillsLower
    ? [
        ...selectedSkills.filter((skill) => finalSkillsLower.has(skill.canonical_name.toLowerCase())),
        ...confirmedSkills!.final_skills
          .filter(
            (name) => !selectedSkills.some((skill) => skill.canonical_name.toLowerCase() === name.toLowerCase())
          )
          .map(
            (name): SelectedSkill => ({
              raw_term: name,
              canonical_name: name,
              match_type: 'always include / added',
              confidence: null as unknown as number,
              relevance_score: 0,
              evidence: '',
            })
          ),
      ]
    : selectedSkills;

  const displayMissingSkills = finalSkillsLower
    ? missingSkills.filter((term) => !finalSkillsLower.has(term.toLowerCase()))
    : missingSkills;

  return (
    <div className="run-detail">
      <h3>
        Run <code>{run.run_id}</code> - <StatusBadge status={run.status} />
      </h3>

      {postingText && (
        <details className="posting-text">
          <summary>Job posting</summary>
          <pre>{postingText}</pre>
        </details>
      )}

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
        <>
          <pre className="error-detail">{run.error ?? 'Unknown error'}</pre>
          {!reviewOpen && run.skill_review && (
            <button onClick={() => setReviewOpen(true)}>Review &amp; retry rendering</button>
          )}
        </>
      )}

      {run.status === 'awaiting_review' && run.skill_review && (
        <SkillReviewPanel
          runId={run.run_id}
          review={run.skill_review}
          previousSelection={confirmedSkills?.included_skills}
          onConfirmed={handleSkillsConfirmed}
        />
      )}

      {reviewOpen && run.status !== 'awaiting_review' && run.skill_review && (
        <SkillReviewPanel
          runId={run.run_id}
          review={run.skill_review}
          previousSelection={confirmedSkills?.included_skills}
          onConfirmed={handleSkillsConfirmed}
        />
      )}

      {run.status === 'completed' && (
        <>
          {(run.metrics?.skills_block?.trim_iterations ?? 0) > 0 && !reviewOpen && (
            <div className="banner warning">
              {run.metrics!.skills_block!.trim_iterations} skill(s) were automatically dropped to fit
              one page.{' '}
              {run.skill_review && (
                <button className="link-button" onClick={() => setReviewOpen(true)}>
                  Review &amp; re-render
                </button>
              )}
            </div>
          )}

          {run.metrics && <UsageSummary metrics={run.metrics} />}

          <div className="pdf-preview">
            <iframe title="Tailored resume" src={api.runPdfUrl(run.run_id)} />
          </div>

          <div className="skill-provenance">
            <h4>Selected skills ({displaySelectedSkills.length})</h4>
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
                  {displaySelectedSkills.map((skill) => (
                    <tr key={skill.canonical_name}>
                      <td>{skill.canonical_name}</td>
                      <td>{skill.match_type}</td>
                      <td>{skill.confidence?.toFixed?.(2) ?? '—'}</td>
                      <td className="evidence-cell">{skill.evidence || '—'}</td>
                    </tr>
                  ))}
                  {displaySelectedSkills.length === 0 && (
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
            <h4>Missing skills ({displayMissingSkills.length})</h4>
            <p className="hint">
              Extracted from the posting but not in the skills cache. Promote the ones you
              actually want tracked.
            </p>
            <ul>
              {displayMissingSkills.map((term) => (
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
              {displayMissingSkills.length === 0 && (
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

          {!reviewOpen && run.skill_review && !(run.metrics?.skills_block?.trim_iterations ?? 0) && (
            <button className="review-rerender-button" onClick={() => setReviewOpen(true)}>
              Review &amp; re-render skills
            </button>
          )}
        </>
      )}
    </div>
  );
}

function StatusBadge({ status }: { status: RunStatus['status'] }) {
  return <span className={`badge ${status}`}>{status}</span>;
}

/** Phase 9 human-in-the-loop skill-review checkpoint: shown while a run is
 * `awaiting_review` (paused right after Stage 7 - matching/validation/rank -
 * and before Stage 8 - grouping/rendering), and re-openable afterward for a
 * completed run (the "allow rerendering" decision) so the user can adjust
 * their selection and re-render without starting a whole new run. Every
 * skill stays a normal, toggleable checkbox (nothing is disabled) - even an
 * always-include skill can be opted out of one specific run's resume;
 * always-include entries are sorted to the end of the list by the backend
 * (`main._build_skill_review_payload`). A `missing` skill isn't in the
 * cache yet - checking it promotes it into `data/skills.yaml` on confirm.
 * `other_cache_skills` is a collapsed escape hatch for adding a skill
 * unrelated to this posting.
 *
 * `previousSelection` (from `confirmed_skills.json`'s `included_skills`) -
 * when reopening review for a re-render, this is what the user actually
 * confirmed last time, so their earlier choices (e.g. a missing skill they
 * checked) are restored instead of resetting to the original
 * `default_checked` values. */
function SkillReviewPanel({
  runId,
  review,
  previousSelection,
  onConfirmed,
}: {
  runId: string;
  review: SkillReview;
  previousSelection?: string[];
  onConfirmed: () => void;
}) {
  const previousSelectionLower = previousSelection ? new Set(previousSelection.map((s) => s.toLowerCase())) : null;

  const [checked, setChecked] = useState<Record<string, boolean>>(() => {
    const initial: Record<string, boolean> = {};
    for (const skill of review.reviewable_skills) {
      initial[skill.name] = previousSelectionLower
        ? previousSelectionLower.has(skill.name.toLowerCase())
        : skill.default_checked;
    }
    return initial;
  });
  const [extraChecked, setExtraChecked] = useState<Record<string, boolean>>(() => {
    if (!previousSelectionLower) return {};
    const reviewableLower = new Set(review.reviewable_skills.map((skill) => skill.name.toLowerCase()));
    const initial: Record<string, boolean> = {};
    for (const name of review.other_cache_skills) {
      if (previousSelectionLower.has(name.toLowerCase()) && !reviewableLower.has(name.toLowerCase())) {
        initial[name] = true;
      }
    }
    return initial;
  });
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const toggle = (skill: ReviewableSkill) => {
    setChecked((current) => ({ ...current, [skill.name]: !current[skill.name] }));
  };

  const toggleExtra = (name: string) => {
    setExtraChecked((current) => ({ ...current, [name]: !current[name] }));
  };

  const handleConfirm = async () => {
    setSubmitting(true);
    setError(null);
    const included = [
      ...review.reviewable_skills.filter((skill) => checked[skill.name]).map((skill) => skill.name),
      ...Object.keys(extraChecked).filter((name) => extraChecked[name]),
    ];
    try {
      await api.confirmSkills(runId, included);
      onConfirmed();
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="skill-review">
      <p className="hint">
        Review the skills the pipeline matched before finalizing the resume - extraction/matching
        quality is never perfect, so this is your chance to catch gaps. Skills already in your
        cache are checked by default; a newly-checked skill not yet in your cache will be added to
        it when you confirm.
      </p>

      <ul className="skill-review-list">
        {review.reviewable_skills.map((skill) => (
          <li key={skill.name} className={skill.low_confidence ? 'low-confidence' : undefined}>
            <label>
              <input type="checkbox" checked={!!checked[skill.name]} onChange={() => toggle(skill)} />
              <span className="skill-name">{skill.name}</span>
              {skill.is_always_include && <span className="badge">always include</span>}
              {skill.low_confidence && <span className="badge warning">low confidence</span>}
              {skill.source === 'missing' &&
                (checked[skill.name] ? (
                  <span className="badge new-skill">will be added to cache</span>
                ) : (
                  <span className="badge muted">not yet in cache</span>
                ))}
            </label>
            {skill.evidence && <p className="hint evidence-cell">{skill.evidence}</p>}
          </li>
        ))}
        {review.reviewable_skills.length === 0 && (
          <li>
            <em>No matched or missing skills to review.</em>
          </li>
        )}
      </ul>

      {review.other_cache_skills.length > 0 && (
        <details className="add-other-skills">
          <summary>Add another skill from your cache ({review.other_cache_skills.length})</summary>
          <ul className="skill-review-list">
            {review.other_cache_skills.map((name) => (
              <li key={name}>
                <label>
                  <input type="checkbox" checked={!!extraChecked[name]} onChange={() => toggleExtra(name)} />
                  <span className="skill-name">{name}</span>
                </label>
              </li>
            ))}
          </ul>
        </details>
      )}

      {error && <div className="banner error">{error}</div>}

      <button onClick={handleConfirm} disabled={submitting}>
        {submitting ? 'Confirming…' : 'Confirm selection'}
      </button>
    </div>
  );
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
