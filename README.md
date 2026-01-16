# woa7015-medvqa

CNN-LSTM (discriminative):

- every epoch logs:
  - loss
  - accuracy (ckpt selection metric)
- report after training:
  - Overall:
    - accuracy
    - top-5 accuracy
    - macro f1
  - OPEN/CLOSED: (evaluating these on OPEN doesn't make too much sense, but it's for us to see how the model performs on these subsets individually)
    - accuracy
    - top-5 accuracy
    - macro f1

BLIP (generative)

- every epoch logs:
  - loss
  - token_f1 (ckpt selection metric)
- report after training:
  - Overall:
    - exact match
    - token f1
  - OPEN
    - exact match
    - token f1
    - BLEU
    - ROUGE-L
    - BERTScore
  - CLOSED
    - exact match
    - token f1

Comparison

- Overall
  - exact match
  - token f1
- CLOSED
  - exact match
  - token f1
- OPEN
  - exact match
  - token f1
