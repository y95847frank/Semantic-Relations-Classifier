# Semantic-Relations-Classifier
SemEval-2010 Task 8: Multi-Way Classification of Semantic Relations between Pairs of Nominals

## Predict

```
sh eval.sh
```
This command would produce the prediction as 'ans.txt' file.

## Evaluation

```
perl semeval2010_task8_scorer-v1.2.pl <proposed_answer>  dataset/answer_key.txt
```
