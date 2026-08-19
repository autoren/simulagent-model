# V125 SGD Catalog and Population Plan

V125 uses only V124's text-free structural inventory. The visible domains are Hotels, Movies, and Services:
the source audit shows that these are the train-seen domains with both exact train/test service-intent pairs
and valid unseen-service test intents. All six qualifying known pairs are declared. Novel hypotheses remain
safe composite sets, one per visible domain; unsupported is the composite of unseen domains; insufficient
is retained as a control. The complete catalog therefore has eleven choices.

All 4,881 train candidates for the six declared pairs form the future retrieval-training population. The
evaluation population is selected by a frozen hash: 32 test records per known pair, 64 novel-valid records
per visible domain, and 48 unsupported records per unseen domain, for 192 records in each of three classes
and 576 total.

No language, dialogue, token, or slot value is read or emitted. A pass can authorize only preregistration
of the cross-dataset retrieval-selectivity experiment; extraction and evaluation require another lock.
