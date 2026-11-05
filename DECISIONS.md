# Decisions

<!-- Fill in each section at the step the activity website gives, then commit and push.
     Write each answer after its label. Keep the headings and labels exactly as they are:
     `python -m analysis` reads them. Limit: 150 words for Before results.

## Before results

For each dataset: positive class; which error matters more for the job (missed positive or false alarm) and why.

\* readmission:A positive means returning to the hospital within 30 days of discharge. Missing a positive matters more than a false alarm because the patient could miss a follow-up call or extra support after going home.
\* diabetes:A positive means the adult reports a doctor's diagnosis of diabetes or has an HbA1c of 6.5% or higher. Missing a positive matters more than a false alarm because a false alarm can lead to an extra test, while a missed case can mean losing the chance to follow up with someone who may need it.

## Recommendations

\* readmission: None is adequate for this job. At a cutoff of 0.10 Random forest missed 165 of 359 readmissions and raised 1,234 false alarms. I don't think that many extra follow-ups are justified when the model still misses 165 readmissions. 

\* diabetes: I recommend Logistic Regression at a cutoff of 0.12 for further evaluation. It caught 180 of 200 positives, meeting the 90% sensitivity target on the test set, with 34.6 false alarms per 100 adults screened. I chose this over Random Forest because it met the target with fewer false alarms.

## Design choice and AI use

A few sentences each. The design choice must be your own idea, not an AI tool's.

\* design choice: When testing the threshold slider I expected the numbers to automatically move without changing the dataset. Although the slider works within either dataset, I could see other clinical leads getting this confused as well. So I decided to change the format so the readmission baseline gave everyone the same score, so the counts stayed the same until the cutoff crossed that score along with a brief explanation. 

\* AI suggestion I turned down: For Beyond Accuracy, AI suggested showing the selected model and the baseline side by side. I chose to show one model first, with the comparison below. I chose this because I thought it would be easier to read. Having two models side by side can get confusing, so to read it with more clarity I chose to have the comparison below.


