import pandas as pd
import numpy as np

df = pd.read_csv('C:/pd-speech-crosslingual/23_final_supervisor_evaluation/06_sensitivity_analysis_264/summaries/sensitivity_classification_results_NESTED_CV_COMPACT.csv')
preds = pd.read_csv('C:/pd-speech-crosslingual/23_final_supervisor_evaluation/06_sensitivity_analysis_264/predictions/sensitivity_sample_predictions_NESTED_CV_COMPACT.csv')

print('Total final rows:', len(df))
print('Has NaNs:', df.isnull().values.any())
print('Classifiers:', sorted(df['classifier'].unique()))
print('Models:', sorted(df['representation_model'].unique()))
print('Layers:', sorted(df['layer'].unique()))

print('Total prediction rows:', len(preds))

# Check for duplicate predictions
# We check duplicates based on fold, scenario, model, layer, classifier, and sample_id
dupes = preds.duplicated(subset=['outer_fold', 'scenario', 'representation_model', 'layer', 'classifier', 'file_name'])
print('Duplicate predictions:', dupes.any())

print('Total result blocks/scenarios:', len(df))
print('Validation Complete')
