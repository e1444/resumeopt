import { describe, expect, it } from 'vitest';
import { computeProgressPercent, SUBSTAGE_ORDER, STAGE_TOTAL, type ProgressInput } from './progress';

/** Simulates one running run's whole sequence of `GET /api/runs/{id}` status
 * snapshots (what the frontend would actually poll over time) and threads
 * the high-water-mark through them exactly like `ProgressBar` does, without
 * a live backend, browser, or any real LLM calls/tokens. Returns every step
 * so tests can assert on the full progression, and logs each step so the
 * whole bar trajectory is visible without manually watching a browser. */
function simulateRun(steps: ProgressInput[]) {
  let max = 0;
  const results: { input: ProgressInput; rawPercent: number | null; percent: number | null }[] = [];
  for (const input of steps) {
    const { rawPercent, percent } = computeProgressPercent(input, max);
    if (percent !== null) max = percent;
    results.push({ input, rawPercent, percent });
    // eslint-disable-next-line no-console
    console.log(
      `stage=${input.stage ?? '-'} (${input.stageIndex ?? '-'}/${input.stageTotal ?? '-'})` +
        `${input.substage ? ` substage=${input.substage} ${input.substageCompleted}/${input.substageTotal}` : ''}` +
        ` -> raw=${rawPercent ?? '-'} percent=${percent ?? '-'}`,
    );
  }
  return results;
}

/** Builds the coarse-stage-only snapshot for a given `PIPELINE_STAGES` index
 * (no substage data), matching what the backend reports for every stage
 * except `parse_posting`. */
function coarseStage(stage: string, stageIndex: number): ProgressInput {
  return { stage, stageIndex, stageTotal: STAGE_TOTAL };
}

/** Builds a `parse_posting` snapshot reporting batch progress for one of its
 * 5 internal substages. */
function substageStep(substage: string, completed: number, total: number): ProgressInput {
  return { stage: 'parse_posting', stageIndex: 2, stageTotal: STAGE_TOTAL, substage, substageCompleted: completed, substageTotal: total };
}

describe('computeProgressPercent', () => {
  it('fills a plain coarse stage (no substage capability) as soon as it is entered', () => {
    const { percent } = computeProgressPercent(coarseStage('read_posting', 0), 0);
    expect(percent).toBe(Math.round((1 / 7) * 100)); // 14
  });

  it('starts parse_posting at its coarse "not yet started" value, not fully filled, before any substage data arrives', () => {
    const { percent, rawPercent } = computeProgressPercent(coarseStage('parse_posting', 2), 29);
    // Entering stage index 2 with no substage info yet should read the same
    // as "stage 2 done, stage 3 not started" (i.e. 2/7), NOT jump ahead to
    // as if parse_posting were already complete (3/7).
    expect(rawPercent).toBe(Math.round((2 / 7) * 100)); // 29
    expect(percent).toBe(29);
  });

  it('does not treat a single completed substage as the entire parse_posting stage being done', () => {
    // Regression test for the reported "29% -> 43%" jump: chunk_screening
    // completing (1/1) is only sub-stage 1 of 5, so it should advance the
    // bar by roughly 1/5th of parse_posting's ~14-point share, not all of it.
    const { percent } = computeProgressPercent(substageStep('chunk_screening', 1, 1), 29);
    expect(percent).not.toBe(43);
    expect(percent).toBeLessThan(35);
    expect(percent).toBeGreaterThan(29);
  });

  it('advances smoothly and proportionally through all 5 ordered substages', () => {
    const percents = SUBSTAGE_ORDER.map((name) => {
      const { rawPercent } = computeProgressPercent(substageStep(name, 1, 1), 0);
      return rawPercent!;
    });
    // Strictly increasing as we move through chunk_screening -> extraction
    // -> categorization -> atomicity -> redundancy, each fully completing.
    for (let i = 1; i < percents.length; i++) {
      expect(percents[i]).toBeGreaterThan(percents[i - 1]);
    }
    // The last substage (redundancy) fully completing should read the same
    // as parse_posting being fully done (3/7), matching the coarse value the
    // NEXT stage would start from.
    expect(percents[percents.length - 1]).toBe(Math.round((3 / 7) * 100)); // 43
  });

  it('never lets the bar move backward, even when a new substage resets its own batch fraction to 0', () => {
    const first = computeProgressPercent(substageStep('extraction', 2, 2), 0);
    const second = computeProgressPercent(substageStep('categorization', 0, 3), first.percent!);
    // categorization starting at 0/3 has a lower (or, at a substage boundary,
    // equal) raw fraction than extraction just having fully finished - either
    // way the clamp must never let the displayed percent decrease.
    expect(second.rawPercent).toBeLessThanOrEqual(first.percent!);
    expect(second.percent).toBe(first.percent); // clamped, not decreased
  });

  it('falls back to null (indeterminate) when no recognized stage has been reported yet', () => {
    const { percent, rawPercent } = computeProgressPercent({}, 0);
    expect(percent).toBeNull();
    expect(rawPercent).toBeNull();
  });
});

describe('full run simulation (no live backend/LLM calls)', () => {
  it('produces a monotonically non-decreasing, smoothly-advancing percent across an entire run', () => {
    const steps: ProgressInput[] = [
      coarseStage('read_posting', 0),
      coarseStage('init_llm_provider', 1),
      coarseStage('parse_posting', 2), // entered, no substage data yet
      substageStep('chunk_screening', 0, 1),
      substageStep('chunk_screening', 1, 1),
      substageStep('extraction', 0, 2),
      substageStep('extraction', 1, 2),
      substageStep('extraction', 2, 2),
      substageStep('categorization', 0, 2),
      substageStep('categorization', 1, 2),
      substageStep('categorization', 2, 2),
      substageStep('atomicity', 0, 1),
      substageStep('atomicity', 1, 1),
      substageStep('redundancy', 0, 3),
      substageStep('redundancy', 1, 3),
      substageStep('redundancy', 2, 3),
      substageStep('redundancy', 3, 3),
      coarseStage('validate_selected_skills', 3),
      coarseStage('group_skills', 4),
      coarseStage('rendering', 5),
      coarseStage('finalizing', 6),
    ];

    const results = simulateRun(steps);
    const percents = results.map((r) => r.percent);

    // Monotonically non-decreasing throughout the whole run.
    for (let i = 1; i < percents.length; i++) {
      expect(percents[i]!).toBeGreaterThanOrEqual(percents[i - 1]!);
    }

    // Reaches 100% once the last stage (finalizing, index 6/7) is entered.
    expect(percents[percents.length - 1]).toBe(100);

    // No single step-to-step jump within the parse_posting substage
    // sequence should consume the stage's whole ~14-point share in one go
    // (that was the reported bug) - each individual batch update should
    // move the bar by a modest amount.
    const parsePostingIndices = results
      .map((r, i) => (r.input.stage === 'parse_posting' ? i : -1))
      .filter((i) => i >= 0);
    for (const i of parsePostingIndices) {
      if (i === 0) continue;
      const delta = percents[i]! - percents[i - 1]!;
      expect(delta).toBeLessThanOrEqual(5);
    }
  });
});
