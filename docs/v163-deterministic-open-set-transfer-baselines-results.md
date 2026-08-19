# V163 deterministic open-set transfer baselines results

## Outcome

V163 passed every prospectively frozen pipeline, integrity, access, and residual-qualification gate. The
hash-only split contains 96 calibration and 96 evaluation records, exactly 24 per structural class in each.
Character retrieval trained on 2,149 MASSIVE train utterances spanning the 12 declared intents. Calibration
selected known threshold `0.80` and unsupported threshold `0.35` from 124 frozen pairs.

Evaluation results:

| Baseline | Exact decision | Status macro F1 | Known exact | Novel exact | False-known | Mean regret |
|---|---:|---:|---:|---:|---:|---:|
| Complete safe enumeration / abstain | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.1250 |
| Ask always | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.1250 |
| Identifier grammar | 0.4688 | 0.3829 | 0.4375 | 0.0000 | 0.1250 | 3.1875 |
| Character n-gram retrieval | 0.4583 | 0.4910 | 0.3542 | 0.7917 | 0.0208 | 1.9375 |
| Deterministic consensus | 0.1979 | 0.2859 | 0.2292 | 0.0000 | 0.0000 | 0.9531 |
| Oracle | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |

The standalone grammar and retrieval classifiers were more often exactly correct than consensus, but their
costly false acceptances made them worse policies than asking. Consensus accepted only identical complete
decisions from both methods. This lowered mean regret below ask-always and eliminated false-known acceptance.

## Frozen residual

Consensus handled 20 evaluation records without a model. This nonresidual subset had 95% exact decision
accuracy, zero false-known acceptance, and mean regret `0.20`. The one error was therefore bounded by the
frozen asymmetric costs rather than a false-known capability acceptance.

Consensus abstained on 76 records. Residual membership uses only the two deterministic predictions, not
truth or source language. The residual contains:

- 13 familiar-known records;
- 24 unfamiliar-known records;
- 23 valid-undeclared records; and
- 16 unsupported records.

The residual spans all four classes and its canonical identifier-only payload SHA-256 is
`360b9cc079726a06405fc7d331b578dc985133a361d4a210fb891671c0d1a19c`.

Every frozen residual gate passed: count bounds, class coverage, nonresidual count, exactness, false-known
rate, regret, consensus overall false-known rate, and regret relative to ask-always.

## Access and decision

Development language and the declared training archive were each read automatically once. Protected
language was not opened. Manual utterance inspection, model loads, model generations, API calls, training,
service calls, side effects, and execution were zero. Twenty-four missing-observation controls abstained
exactly, and the complete 17-state safe universe was retained by every baseline.

Freeze V163 without changing thresholds, consensus, costs, split, residual membership, or gates. Passing
authorizes only separate preregistration of one pinned local model on the 76 frozen residual identifiers.
The model remains a shadow candidate proposer. It cannot alter nonresidual decisions, prune hypotheses,
update authoritative state or belief, select an action, or execute. Protected access, APIs, training,
ontology induction, planning, action, and execution remain closed.

## Claim boundary

This is record-disjoint MASSIVE deterministic development evidence. It shows that agreement between two
weak deterministic methods can isolate a small high-precision subset and reduce decision regret while
defining a nontrivial residual. It is not protected evidence, V160 relation-codebook confirmation,
unrestricted open-world understanding, ontology learning, or deployment safety.
