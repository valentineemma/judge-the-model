# Decisions

<!-- Fill in each section at the step the activity website gives, then commit and push.
     Write each answer after its label. Keep the headings and labels exactly as they are:
     `python -m analysis` reads them. Limit: 150 words for Before results.

## Before results

For each dataset: positive class; which error matters more for the job (missed positive or false alarm) and why.

* readmission:A positive means returning to the hospital within 30 days of discharge. Missing a positive matters more than a false alarm because the patient could miss a follow-up call or extra support after going home.
* diabetes:A positive means the adult reports a doctor's diagnosis of diabetes or has an HbA1c of 6.5% or higher. Missing a positive matters more than a false alarm because a false alarm can lead to an extra test, while a missed case can mean losing the chance to follow up with someone who may need it.

## Recommendations

* readmission: <model and threshold, or "none is adequate for this job"> because ...
* diabetes: ...

## Design choice and AI use

A few sentences each. The design choice must be your own idea, not an AI tool's.

* design choice: <one thing you chose to show, or leave out, for your clinical lead> because ...
* AI suggestion I turned down: <what an AI tool suggested, and why you didn't use it, or "no AI tools used">

