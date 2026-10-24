# Fair Claims Handling and Responsible AI

Fictional policy written for this proof of concept. It sets the rules for how the triage model and the investigator copilot may be used.

## A score is a triage signal, not evidence

The fraud probability ranks claims for attention; it is not evidence of fraud, and a high score is never a reason on its own to delay, reduce or deny a claim. Every adverse decision must rest on facts found by a person and written in the file. Nobody may tell a claimant that a computer flagged their claim as fraudulent.

## Humans decide

The model and the copilot recommend; adjusters, investigators and managers decide. The copilot may summarise a file, list indicators and suggest investigation steps from the playbook. It may not decide coverage, draft a denial or contact anyone. Investigators may disagree with the queue and must record why when they skip or reorder files.

## Treating claimants fairly

Communicate in neutral language, explain what documents are needed and why, and respond within the normal service standards even while a file is under investigation. Claimants who are investigated and found honest must not be treated differently on later claims because they were once investigated.

## Protected grounds and proxies

The triage model does not use age, sex, ethnicity, religion, language, family status, disability or any other protected ground, and it does not use the city or neighbourhood of the loss. Place names are removed from adjuster notes before the text model reads them, because location can stand in for protected characteristics. Provider identities are only used through their behaviour (volumes, cost ratios, repeated pairings), never as a fixed blacklist inside the model.

## Monitoring for unfair outcomes

Every month, compare how often claims are queued and fast-tracked across provinces, loss types and segments. A large difference that the indicators do not explain is investigated like any other model problem. Drift alerts, the fast-track audit and the share of honest customers investigated are reported to the claims leadership team together.

## Privacy

Claim files contain personal information and are used only to settle and investigate claims. Notes, statements and documents must not be pasted into public AI services. The copilot runs on a local model on the company's own machines, so claim data does not leave them; this is enforced in the code, which refuses to call any model server that is not on the same machine. Keep only what the file needs, and follow the retention schedule when a file closes.

## Complaints and review

A claimant can ask for any decision to be reviewed by a manager who was not involved. If a complaint shows the model or the copilot contributed to an unfair outcome, the case is logged, the model owner is informed, and the case is used to test the next model version.
