export interface Skill {
  name: string;
  aliases: string[];
  always_include: boolean;
}

export interface RunSummary {
  run_id: string;
  status: 'running' | 'awaiting_review' | 'completed' | 'failed';
  created_at: string;
}

export interface RunMetrics {
  parse?: { matched_skill_count?: number; missing_skill_count?: number };
  validation?: { selected_skill_count?: number };
  skills_block?: { active_sections?: string[]; canonical_skill_count?: number; trim_iterations?: number };
  pdf_validation?: { status?: string; page_count?: number; skills_section_line_count?: number };
  timings_ms?: Record<string, number>;
  llm_usage?: {
    by_role?: Record<string, { model?: string; call_count?: number; total_tokens?: number }>;
    combined?: { call_count?: number; total_tokens?: number };
  };
  [key: string]: unknown;
}

export interface RunStatus extends RunSummary {
  error?: string;
  metrics?: RunMetrics;
  current_stage?: string;
  stage_index?: number;
  stage_total?: number;
  substage?: string;
  substage_completed?: number;
  substage_total?: number;
  skill_review?: SkillReview;
}

export interface MissingSkillsLog {
  missing_skills: string[];
  count: number;
}

/** What the user confirmed at the Phase 9 review checkpoint - written once
 * `run_pipeline_from_review` finishes. `included_skills` is their full,
 * pre-trim intent (used to restore checkbox state when re-opening review);
 * `final_skills` is what actually survived the fit-to-budget trim loop and
 * is on the rendered PDF (used to correct the post-completion "Selected
 * skills"/"Missing skills" display, which otherwise only reflects the
 * pre-review `validation_report.json`/`missing_skills.json`). */
export interface ConfirmedSkillsLog {
  included_skills: string[];
  final_skills: string[];
}

export interface SelectedSkill {
  raw_term: string;
  canonical_name: string;
  match_type: string;
  confidence: number;
  relevance_score: number;
  evidence: string;
}

export interface ValidationReportLog {
  status: string;
  notes: string[];
  issues: unknown[];
  selected_skills: SelectedSkill[];
}

/** One entry in the Phase 9 human-in-the-loop skill-review checkpoint's
 * reviewable list - see `main._build_skill_review_payload`. */
export interface ReviewableSkill {
  name: string;
  source: 'matched' | 'always_include' | 'missing';
  match_type: string | null;
  confidence: number | null;
  evidence: string | null;
  low_confidence: boolean;
  default_checked: boolean;
  is_always_include: boolean;
}

export interface SkillReview {
  reviewable_skills: ReviewableSkill[];
  other_cache_skills: string[];
}

/** Optional overrides matching the backend's `RunIn` model - all optional,
 * the backend falls back to its own defaults when omitted. */
export interface RunOptions {
  provider?: string;
  model?: string;
  reasoning_model?: string;
  screening_model?: string;
  use_llm_parser?: boolean;
  max_concurrency?: number;
}


async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    headers: { 'Content-Type': 'application/json' },
    ...init,
  });
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const body = await response.json();
      detail = body.detail ?? detail;
    } catch {
      // response had no JSON body - keep statusText
    }
    throw new Error(detail);
  }
  if (response.status === 204) {
    return undefined as T;
  }
  return response.json() as Promise<T>;
}

export const api = {
  listSkills: () => request<Skill[]>('/api/skills'),
  addSkill: (name: string, aliases: string[]) =>
    request<Skill[]>('/api/skills', {
      method: 'POST',
      body: JSON.stringify({ name, aliases }),
    }),
  deleteSkill: (name: string) =>
    request<Skill[]>(`/api/skills/${encodeURIComponent(name)}`, { method: 'DELETE' }),
  updateSkill: (name: string, updates: { aliases?: string[]; always_include?: boolean }) =>
    request<Skill[]>(`/api/skills/${encodeURIComponent(name)}`, {
      method: 'PATCH',
      body: JSON.stringify(updates),
    }),

  getTemplate: async (): Promise<string> => {
    const response = await fetch('/api/template');
    if (!response.ok) {
      throw new Error(`Failed to load template (${response.status})`);
    }
    return response.text();
  },
  saveTemplate: (content: string) =>
    request<{ status: string }>('/api/template', {
      method: 'POST',
      body: JSON.stringify({ content }),
    }),

  startRun: (postingText: string, options: RunOptions = {}) =>
    request<{ run_id: string; status: string }>('/api/runs', {
      method: 'POST',
      body: JSON.stringify({ posting_text: postingText, ...options }),
    }),
  listRuns: () => request<RunSummary[]>('/api/runs'),
  getRun: (runId: string) => request<RunStatus>(`/api/runs/${encodeURIComponent(runId)}`),
  confirmSkills: (runId: string, includedSkills: string[]) =>
    request<{ run_id: string; status: string }>(
      `/api/runs/${encodeURIComponent(runId)}/confirm-skills`,
      { method: 'POST', body: JSON.stringify({ included_skills: includedSkills }) }
    ),
  getRunPosting: async (runId: string): Promise<string> => {
    const response = await fetch(`/api/runs/${encodeURIComponent(runId)}/posting`);
    if (!response.ok) {
      throw new Error(`Failed to load posting text (${response.status})`);
    }
    return response.text();
  },
  getRunLog: <T>(runId: string, logName: string) =>
    request<T>(`/api/runs/${encodeURIComponent(runId)}/logs/${logName}`),
  promoteMissingSkill: (runId: string, term: string) =>
    request<Skill[]>(
      `/api/runs/${encodeURIComponent(runId)}/missing-skills/${encodeURIComponent(term)}/promote`,
      { method: 'POST' }
    ),
  runPdfUrl: (runId: string) => `/api/runs/${encodeURIComponent(runId)}/pdf`,
};
