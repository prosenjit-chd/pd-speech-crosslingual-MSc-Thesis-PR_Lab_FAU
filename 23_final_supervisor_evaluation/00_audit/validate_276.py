import pandas as pd

df = pd.read_csv('C:/pd-speech-crosslingual/23_final_supervisor_evaluation/05_classification/summaries/all_classification_results_NESTED_CV_COMPACT.csv')
print('Total final rows:', len(df))
print('Has NaNs:', df.isnull().values.any())
print('Classifiers:', sorted(df['classifier'].unique()))
print('Models:', sorted(df['representation_model'].unique()))
print('Layers:', sorted(df['layer'].unique()))

preds = pd.read_csv('C:/pd-speech-crosslingual/23_final_supervisor_evaluation/05_classification/predictions/sample_level_predictions_NESTED_CV_COMPACT.csv')
print('Total prediction rows:', len(preds))

# Check for exact block count (72 blocks * 6 scenarios per block = 432 rows)
print('Total result blocks/scenarios:', len(df))
