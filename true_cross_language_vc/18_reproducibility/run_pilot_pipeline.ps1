# Automated Execution Script for 12-File True Cross-Language Pilot
$VENV = "C:\pd-speech-crosslingual\.venv\Scripts\python.exe"
$XTTS_ENV = "C:\pd-speech-crosslingual\true_cross_language_vc\env_xtts\Scripts\python.exe"

Write-Host "Running Audio Validation..." -ForegroundColor Cyan
& $VENV true_cross_language_vc/scripts/07_validate_pilot_audio.py

Write-Host "Running Language Verification..." -ForegroundColor Cyan
& $VENV true_cross_language_vc/scripts/08_verify_language_id.py

Write-Host "Running Content Accuracy Evaluation..." -ForegroundColor Cyan
& $VENV true_cross_language_vc/scripts/09_evaluate_content_accuracy.py

Write-Host "Running Speaker Similarity Evaluation..." -ForegroundColor Cyan
& $VENV true_cross_language_vc/scripts/10_evaluate_speaker_similarity.py

Write-Host "Generating Final Pilot Report..." -ForegroundColor Cyan
& $VENV true_cross_language_vc/scripts/11_generate_pilot_report.py
