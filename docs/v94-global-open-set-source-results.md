# V94 Global Capability-Catalog Source Result

V94 successfully formed a fresh global catalog but failed its complete source-feasibility gate. The
pinned `dev/dialogues_005.json` shard matched 1,512,309 bytes and Git blob
`80ddca2c6ed633e49e2f827c05615e514b62ca17`. Its automatic text-free inventory contained 666 source
records and 645 class candidates across four fresh services.

The catalog retained three services and held out one service as unsupported. It produced 240 familiar
known, 7 unfamiliar known, 148 source-valid novel, 201 unsupported, and 49 insufficient-evidence
candidates. Known, unsupported, and insufficient-evidence counts and coverage passed. Two structural
interactions caused four gates to fail:

- accumulated user history almost always contained an intent-surface token, leaving only 7 zero-overlap
  unfamiliar records against the frozen minimum of 24;
- five source-supported catalog pairs could not simultaneously retain four declared pairs and hide two,
  so only one pair was hidden and novel coverage reached one service rather than two.

No language or derived tokens were emitted, no utterance was manually inspected, and there were zero
model loads, generations, API calls, training runs, service calls, or side effects. This is source-
feasibility evidence only. It says nothing about model novelty detection or abstention.

Freeze V94 without changing its split or thresholds. A fresh successor may preregister current-turn
lexical unfamiliarity rather than accumulated-history unfamiliarity and reserve two hidden supported
pairs while retaining three declared supported pairs. Those choices must be locked on a new shard before
payload access.
