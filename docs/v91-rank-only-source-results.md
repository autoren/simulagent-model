# V91 Rank-Only Source Results

The prospectively locked V91 source extension passed every gate. The one authorized download matched
the registered `1,965,028` bytes and Git blob `6abbd79ddd2c58c386ca9cfb748bb55a4efd7be5`
before parsing. The structural inventory contains no utterance text, slot values, dialogue history, or
prompt fields.

The fresh `dev/dialogues_003.json` shard contains 866 eligible user records across 128 dialogues: 794
active records and 72 genuine `NONE` records. Four services are available:

- `Alarm_1`: 199 records, with `AddAlarm`, `GetAlarms`, and `NONE`;
- `Homes_1`: 82 records, with `FindApartment` and `NONE`;
- `Services_4`: 437 records, with `BookAppointment`, `FindProvider`, and `NONE`;
- `Weather_1`: 148 records, with `GetWeather` and `NONE`.

The frozen record-index SHA-256 is
`a6e9f4769623666099e307aa4e597ad9629b3e0341110e912c8fa9b622c584a0`.
There was one pinned HTTP download and one code-only payload parse, with zero manual utterance
inspection, new model-weight download, model load, generation, API call, training run, service call, or
external side effect.

This result authorizes only preregistration and pre-language selection of a fresh rank-only shadow
population. It does not authorize model inference yet. The complete deterministic intent set, mandatory
`NONE`, canonical completion rule, controls, prompt, decoding, utility gates, and exact policy-invariance
harness must be frozen before utterance extraction or model access.
