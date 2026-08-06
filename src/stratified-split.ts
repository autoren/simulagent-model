import type { DatasetSplit, SplitRatios } from './contracts';
import { stableHash } from './split';

export interface StratifiedGroup {
  id: string;
  features: Record<string, number>;
}

export interface StratifiedSplitResult {
  plan: Map<string, DatasetSplit>;
  objective: number;
  restarts: number;
}

const splits: DatasetSplit[] = ['train', 'valid', 'test'];

export function createStratifiedSplitPlan(
  groups: StratifiedGroup[],
  ratios: SplitRatios,
  seed: string,
  restarts = 32,
): StratifiedSplitResult {
  validate(groups, ratios, restarts);
  const unique = new Map(groups.map((group) => [group.id, group]));
  if (unique.size !== groups.length) {
    throw new Error('Stratified split group ids must be unique.');
  }
  if (groups.length < 3) {
    return {
      plan: new Map(groups.map((group) => [group.id, 'train'] as const)),
      objective: 0,
      restarts,
    };
  }

  const featureTotals = sumFeatures(groups);
  const targets = Object.fromEntries(
    splits.map((split) => [split, scaleFeatures(featureTotals, ratios[split])]),
  ) as Record<DatasetSplit, Record<string, number>>;
  const capacities = groupCapacities(groups.length, ratios);
  let bestPlan: Map<string, DatasetSplit> | undefined;
  let bestObjective = Number.POSITIVE_INFINITY;

  for (let restart = 0; restart < restarts; restart += 1) {
    const ordered = [...groups].sort((left, right) => {
      const rarityDifference = rarityScore(right, featureTotals) - rarityScore(left, featureTotals);
      if (Math.abs(rarityDifference) > 1e-12) {
        return rarityDifference;
      }
      const leftHash = stableHash(`${seed}:${restart}:${left.id}`);
      const rightHash = stableHash(`${seed}:${restart}:${right.id}`);
      return leftHash - rightHash || left.id.localeCompare(right.id);
    });
    const plan = greedyAssignment(ordered, targets, capacities, `${seed}:${restart}`);
    improveBySwapping(plan, unique, targets);
    const score = objective(plan, unique, targets);
    if (score < bestObjective) {
      bestObjective = score;
      bestPlan = plan;
    }
  }

  if (!bestPlan) {
    throw new Error('Unable to create a stratified split plan.');
  }
  return { plan: bestPlan, objective: bestObjective, restarts };
}

function greedyAssignment(
  groups: StratifiedGroup[],
  targets: Record<DatasetSplit, Record<string, number>>,
  capacities: Record<DatasetSplit, number>,
  seed: string,
): Map<string, DatasetSplit> {
  const plan = new Map<string, DatasetSplit>();
  const assignedCounts: Record<DatasetSplit, number> = { train: 0, valid: 0, test: 0 };
  const actual = emptySplitFeatures();
  for (const group of groups) {
    const candidates = splits.filter((split) => assignedCounts[split] < capacities[split]);
    const selected = candidates
      .map((split) => ({
        split,
        delta:
          featureObjective(addFeatures(actual[split], group.features), targets[split]) -
          featureObjective(actual[split], targets[split]),
        hash: stableHash(`${seed}:${group.id}:${split}`),
      }))
      .sort((left, right) => left.delta - right.delta || left.hash - right.hash)[0]?.split;
    if (!selected) {
      throw new Error(`No split capacity remains for ${group.id}.`);
    }
    plan.set(group.id, selected);
    assignedCounts[selected] += 1;
    actual[selected] = addFeatures(actual[selected], group.features);
  }
  return plan;
}

function improveBySwapping(
  plan: Map<string, DatasetSplit>,
  groups: Map<string, StratifiedGroup>,
  targets: Record<DatasetSplit, Record<string, number>>,
): void {
  const ids = [...plan.keys()].sort();
  const actual = emptySplitFeatures();
  for (const [id, split] of plan) {
    actual[split] = addFeatures(actual[split], groups.get(id)!.features);
  }
  for (let pass = 0; pass < 20; pass += 1) {
    let improved = false;
    for (let leftIndex = 0; leftIndex < ids.length; leftIndex += 1) {
      const left = ids[leftIndex];
      const leftSplit = plan.get(left)!;
      for (let rightIndex = leftIndex + 1; rightIndex < ids.length; rightIndex += 1) {
        const right = ids[rightIndex];
        const rightSplit = plan.get(right)!;
        if (leftSplit === rightSplit) {
          continue;
        }
        const leftFeatures = groups.get(left)!.features;
        const rightFeatures = groups.get(right)!.features;
        const previous =
          featureObjective(actual[leftSplit], targets[leftSplit]) +
          featureObjective(actual[rightSplit], targets[rightSplit]);
        const nextLeft = addFeatures(subtractFeatures(actual[leftSplit], leftFeatures), rightFeatures);
        const nextRight = addFeatures(subtractFeatures(actual[rightSplit], rightFeatures), leftFeatures);
        const candidate =
          featureObjective(nextLeft, targets[leftSplit]) +
          featureObjective(nextRight, targets[rightSplit]);
        if (candidate + 1e-12 < previous) {
          plan.set(left, rightSplit);
          plan.set(right, leftSplit);
          actual[leftSplit] = nextLeft;
          actual[rightSplit] = nextRight;
          improved = true;
          break;
        }
      }
      if (improved) {
        break;
      }
    }
    if (!improved) {
      break;
    }
  }
}

function objective(
  plan: Map<string, DatasetSplit>,
  groups: Map<string, StratifiedGroup>,
  targets: Record<DatasetSplit, Record<string, number>>,
): number {
  const actual = emptySplitFeatures();
  for (const [id, split] of plan) {
    actual[split] = addFeatures(actual[split], groups.get(id)!.features);
  }
  return splits.reduce(
    (sum, split) => sum + featureObjective(actual[split], targets[split]),
    0,
  );
}

function featureObjective(
  actual: Record<string, number>,
  target: Record<string, number>,
): number {
  return Object.entries(target).reduce((sum, [feature, expected]) => {
    const difference = (actual[feature] ?? 0) - expected;
    return sum + featureWeight(feature) * difference * difference / Math.max(expected, 1);
  }, 0);
}

function featureWeight(feature: string): number {
  if (feature === 'records') return 12;
  if (feature.startsWith('class:')) return 10;
  if (feature.startsWith('count:')) return 4;
  if (feature.startsWith('family:')) return 3;
  if (feature.startsWith('action:')) return 2;
  return 1;
}

function groupCapacities(total: number, ratios: SplitRatios): Record<DatasetSplit, number> {
  const valid = ratios.valid === 0 ? 0 : Math.max(1, Math.round(total * ratios.valid));
  const test = ratios.test === 0 ? 0 : Math.max(1, Math.round(total * ratios.test));
  const train = total - valid - test;
  if (train < 1) {
    throw new Error('Split ratios leave no training groups.');
  }
  return { train, valid, test };
}

function rarityScore(group: StratifiedGroup, totals: Record<string, number>): number {
  return Object.entries(group.features).reduce(
    (sum, [feature, value]) => sum + featureWeight(feature) * value / Math.max(totals[feature], 1),
    0,
  );
}

function emptySplitFeatures(): Record<DatasetSplit, Record<string, number>> {
  return { train: {}, valid: {}, test: {} };
}

function sumFeatures(groups: StratifiedGroup[]): Record<string, number> {
  return groups.reduce((sum, group) => addFeatures(sum, group.features), {});
}

function addFeatures(
  left: Record<string, number>,
  right: Record<string, number>,
): Record<string, number> {
  const result = { ...left };
  for (const [feature, value] of Object.entries(right)) {
    result[feature] = (result[feature] ?? 0) + value;
  }
  return result;
}

function subtractFeatures(
  left: Record<string, number>,
  right: Record<string, number>,
): Record<string, number> {
  const result = { ...left };
  for (const [feature, value] of Object.entries(right)) {
    result[feature] = (result[feature] ?? 0) - value;
  }
  return result;
}

function scaleFeatures(
  values: Record<string, number>,
  ratio: number,
): Record<string, number> {
  return Object.fromEntries(Object.entries(values).map(([key, value]) => [key, value * ratio]));
}

function validate(groups: StratifiedGroup[], ratios: SplitRatios, restarts: number): void {
  const total = ratios.train + ratios.valid + ratios.test;
  if (Math.abs(total - 1) > 1e-9) {
    throw new Error(`Split ratios must sum to one; received ${total}.`);
  }
  if (!Number.isInteger(restarts) || restarts < 1) {
    throw new Error('Stratification restarts must be a positive integer.');
  }
  for (const group of groups) {
    if (!group.id || Object.values(group.features).some((value) => value < 0)) {
      throw new Error(`Invalid stratified group ${group.id}.`);
    }
  }
}
