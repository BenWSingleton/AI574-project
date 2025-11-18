# AI574-project

To Do 
- Improve embeddings with more preprocessing and putting my alphabetical order
- seperate embedding comparison
- try comparing any two embeddings
- manually get set difference of job and resume, score based on number of skills found
- take single skill embeddings, average each one for a resume, do comparisons

Metrics
y_true: the skills missing in the job that are in the resume
y_predicted: the skills we predict are missing from the resume that are in the job

found: skills that were missing from the resume and we predicted
missed: skill we should have predicted that we didn't

unnecessary: skills that weren't missing from the resume but we still predicted / number of skills predicted
redudant: skills predicted that are already present / number of skills predicted

Finding
- semantic relationships extend to multiple concepts