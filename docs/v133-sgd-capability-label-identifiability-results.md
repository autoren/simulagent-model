# V133 SGD Capability-label Identifiability Results

## Outcome

V133 failed the preregistered capability-identifiability gates:

> `SGD_service_novelty_is_capability_confounded_retract_pure_capability_novelty_interpretation`

Of the 72 selected novel-valid fixtures, 48 (66.67%) came from service-intent definitions whose normalized
intent name exactly matched a declared known choice. The maximum allowed fraction was 10%. Only one of the
three novel composites had no exact name collision:

| Novel choice | Selected records | Exact-name relationship to declared choices |
| --- | ---: | --- |
| `N01` hotels | 24 | No selected member name collides |
| `N02` movies | 24 | Every member collides (`FindMovies`) |
| `N03` services | 24 | Every member collides (`BookAppointment` / `FindProvider`) |

Two novel choices were therefore entirely name-colliding, versus the frozen maximum of zero.

## Important nuance

No selected novel definition was an exact full-schema duplicate of a declared definition. Normalized
descriptions, required/optional slot signatures, and their joint full signatures had zero exact collisions.
The unseen service versions can encode different service-specific mechanics even when they reuse the same
intent name.

That nuance narrows the conclusion:

> The V131/V132 task is a controlled test of service/schema-version discrimination under withheld novel
> definitions, not a clean test in which every novel label denotes an obviously distinct capability.

V132 remains a valid negative result for its frozen classification task. It should not be interpreted as
showing, by itself, that the model cannot recognize semantically distinct new capabilities. The prompt showed
declared definitions but only generic domain-level meanings for novel choices, while 66.67% of novel records
belonged to same-named service variants.

## Access and boundary

The audit read only the pinned train and test schema files. It read zero dialogue files, utterances, slot
values, or model responses; made zero manual semantic judgments; loaded no model; and performed no API call,
training, action, side effect, or execution.

Freeze V133 without redefining collision after seeing the result. The only authorized successor is a
text-free source/catalog design in which selected novel capability names do not collide with declared known
choices. This does not authorize a V132 rerun, prompt revision, model scaling, induction, protected access,
APIs, training, authority, or execution.
