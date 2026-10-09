# Configuration records

`primary_recipe.json` is a machine-readable transcription of verified settings for the primary v1 comparison. `epoch_budget.json` exports the shared numerical run budget without server scheduling fields. `plans_2d.json` preserves the received plans extract.

Final weights are selected on inner-validation, so the 932-epoch training budget does not imply that every primary arm uses its epoch-932 checkpoint. See the [selection protocol](../docs/evaluation.md).
