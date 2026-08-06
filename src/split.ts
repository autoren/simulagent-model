import type { DatasetSplit, SplitRatios } from './contracts';

const authoredPairGroups: Record<string, string> = {
  'relock-behavior-trap': 'pair:relock-behavior',
  'relock-behavior-control': 'pair:relock-behavior',
  'forced-relock-behavior-trap': 'pair:forced-relock-behavior',
  'forced-relock-behavior-control': 'pair:forced-relock-behavior',
  'announced-forced-relock-behavior-trap': 'pair:announced-forced-relock-behavior',
  'announced-forced-relock-behavior-control': 'pair:announced-forced-relock-behavior',
  'unobservable-forced-relock-behavior-trap': 'pair:unobservable-forced-relock-behavior',
  'unobservable-forced-relock-behavior-control': 'pair:unobservable-forced-relock-behavior',
  'locked-access-behavior-trap': 'pair:access-behavior',
  'open-access-behavior-control': 'pair:access-behavior',
};

export function splitGroupForScenario(scenarioId: string): string {
  const authoredPair = authoredPairGroups[scenarioId];
  if (authoredPair) {
    return authoredPair;
  }

  const behavioral = scenarioId.match(
    /^gen-behavior-.+-(relock|relockshort|powertrip)-(\d+)-(trap|control)$/,
  );
  if (behavioral) {
    return `generated-behavior:${behavioral[1]}:seed-${behavioral[2]}`;
  }

  const catalog = scenarioId.match(/^gen-catalog-[^-]+-(\d+)$/);
  if (catalog) {
    return `generated-catalog:seed-${catalog[1]}`;
  }

  return `scenario:${scenarioId}`;
}

export function createSplitPlan(
  groups: string[],
  ratios: SplitRatios,
  seed: string,
): Map<string, DatasetSplit> {
  validateRatios(ratios);
  const uniqueGroups = Array.from(new Set(groups));
  const ordered = uniqueGroups.sort((left, right) => {
    const leftHash = stableHash(`${seed}:${left}`);
    const rightHash = stableHash(`${seed}:${right}`);
    return leftHash - rightHash || left.localeCompare(right);
  });

  if (ordered.length < 3) {
    return new Map(ordered.map((group) => [group, 'train'] as const));
  }

  const validCount = Math.max(1, Math.round(ordered.length * ratios.valid));
  const testCount = Math.max(1, Math.round(ordered.length * ratios.test));
  const trainCount = Math.max(1, ordered.length - validCount - testCount);
  const adjustedValidCount = Math.max(1, ordered.length - trainCount - testCount);

  return new Map(
    ordered.map((group, index) => {
      const split: DatasetSplit =
        index < trainCount
          ? 'train'
          : index < trainCount + adjustedValidCount
            ? 'valid'
            : 'test';
      return [group, split];
    }),
  );
}

export function stableHash(value: string): number {
  let hash = 0x811c9dc5;
  for (let index = 0; index < value.length; index += 1) {
    hash ^= value.charCodeAt(index);
    hash = Math.imul(hash, 0x01000193);
  }
  return hash >>> 0;
}

function validateRatios(ratios: SplitRatios): void {
  const values = [ratios.train, ratios.valid, ratios.test];
  if (values.some((value) => value < 0 || value > 1)) {
    throw new Error('Split ratios must be between 0 and 1.');
  }
  const total = values.reduce((sum, value) => sum + value, 0);
  if (Math.abs(total - 1) > 1e-9) {
    throw new Error(`Split ratios must sum to 1; received ${total}.`);
  }
}

