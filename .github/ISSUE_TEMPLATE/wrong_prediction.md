---
name: Wrong Model Prediction Report
about: Report an incorrect prediction made by the model
title: "[WRONG PREDICTION] "
labels: prediction error
assignees: ''
---

## Wrong Prediction Description

A clear and concise description of the incorrect prediction made by the model.

## Evidence of Incorrect Prediction

Please provide evidence showing why the prediction is incorrect. This can include:

- Expected output vs. actual output
- Ground truth labels (if available)
- Any relevant context that supports your claim

### Example

- **Input:** (e.g., SMILES string or other relevant input)
- **Expected Output:** (e.g., expected label or prediction)
- **Actual Output:** (e.g., model's prediction)

## Model Version

Please specify the version of the model used for making the prediction (e.g., commit hash, version number).

## Dataset Used

Provide details about the dataset being used:

- Name of the dataset
- Version of the dataset (if applicable)
- Any specific subset of the data being referenced

## Code to Reproduce

Include the code snippet that can reproduce the wrong prediction. This should include:

- Loading the model
- Preparing the input data
- Making the prediction
- Any other relevant steps

### Example Code

```python
# Example code to reproduce the issue
from hammer import Hammer

# Load model
model = Hammer.load("v1")

# Prepare input data
input_data = 'CCO'

# Make prediction
prediction = model.predict(input_data)

print(f'Predicted: {prediction}')
```
