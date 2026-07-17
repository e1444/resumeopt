/** Pure progress-percent calculation for the run progress bar, decoupled
 * from React/the DOM so it can be unit-tested by simulating a sequence of
 * backend status updates (see `progress.test.ts`) without spending any real
 * LLM tokens or requiring a live run.
 *
 * The backend reports two levels of granularity while a run is `"running"`:
 * - Coarse `stage`/`stageIndex`/`stageTotal` (one of the 7 `PIPELINE_STAGES`
 *   in `src/main.py`).
 * - Within `parse_posting` only, finer `substage`/`substageCompleted`/
 *   `substageTotal` batch-level progress (one of the 5 `SUBSTAGE_ORDER`
 *   sub-stages it runs internally, in order).
 */

export const STAGE_TOTAL = 7;

/** The 5 batched sub-stages `parse_posting` runs internally, in the exact
 * order `parser/pipeline.py` executes them. Used to convert "how far into
 * the CURRENT sub-stage's own batches" into "how far into the WHOLE
 * parse_posting stage", so one sub-stage finishing doesn't get counted as
 * if the entire stage finished. */
export const SUBSTAGE_ORDER = [
  'chunk_screening',
  'extraction',
  'categorization',
  'atomicity',
  'redundancy',
] as const;

/** Stages that internally report batch-level substage progress (currently
 * only `parse_posting`). A substage-capable stage must start its fraction
 * at 0 (not yet started) while awaiting the first batch report - defaulting
 * it to 1 (like a plain coarse stage) would make the bar jump straight to
 * "this stage is already done" the instant it's entered, and the
 * high-water-mark clamp would then permanently block every real, lower
 * substage fraction from ever being reflected. */
export const SUBSTAGE_CAPABLE_STAGES = new Set<string>(['parse_posting']);

export interface ProgressInput {
  stage?: string;
  stageIndex?: number;
  stageTotal?: number;
  substage?: string;
  substageCompleted?: number;
  substageTotal?: number;
}

export interface ProgressResult {
  /** The percent implied by this single status snapshot alone, with no
   * high-water-mark clamping applied - can be lower than a previous call's
   * `percent` (e.g. right when a new coarse stage is entered and its own
   * substage data hasn't arrived yet). Exposed mainly for tests/logging. */
  rawPercent: number | null;
  /** `rawPercent` clamped to never go below `prevMax` - what the bar should
   * actually display, so it never visibly moves backward. */
  percent: number | null;
}

/**
 * Computes the bar's fill percent for one backend status snapshot.
 *
 * `prevMax` is the caller-maintained running high-water mark (the real
 * component keeps this in a `useRef`; tests thread the same running max
 * through a simulated sequence of calls, one per "poll").
 */
export function computeProgressPercent(input: ProgressInput, prevMax: number): ProgressResult {
  const { stage, stageIndex, stageTotal, substage, substageCompleted, substageTotal } = input;
  const known = stage !== undefined && stageIndex !== undefined && !!stageTotal;
  const substageKnown =
    known && substage !== undefined && substageCompleted !== undefined && !!substageTotal;
  const stageHasSubstages = stage !== undefined && SUBSTAGE_CAPABLE_STAGES.has(stage);

  let substageFraction: number;
  if (substageKnown) {
    const substageIndex = SUBSTAGE_ORDER.indexOf(substage as (typeof SUBSTAGE_ORDER)[number]);
    const batchFraction = substageCompleted! / substageTotal!;
    // Blend "how far through the 5 ordered sub-stages we are" with "how far
    // through the current sub-stage's own batches we are" - a single
    // sub-stage completing (e.g. chunk_screening 1/1) should only advance
    // the bar by roughly 1/5th of parse_posting's share, not all of it.
    substageFraction =
      substageIndex >= 0 ? (substageIndex + batchFraction) / SUBSTAGE_ORDER.length : batchFraction;
  } else {
    substageFraction = stageHasSubstages ? 0 : 1;
  }

  const rawPercent = known ? Math.round(((stageIndex! + substageFraction) / stageTotal!) * 100) : null;
  const percent = rawPercent === null ? null : Math.max(prevMax, rawPercent);
  return { rawPercent, percent };
}
