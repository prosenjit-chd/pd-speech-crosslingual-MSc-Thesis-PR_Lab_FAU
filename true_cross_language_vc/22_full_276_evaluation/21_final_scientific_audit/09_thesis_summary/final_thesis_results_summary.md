# Final Thesis Results Summary

## Dataset
The evaluation was conducted on a full, balanced cohort of 276 read-text recordings, comprising Spanish healthy controls (HC, $N=50$), Spanish Parkinson's disease patients (PD, $N=50$), German HC ($N=88$), and German PD ($N=88$). Analyses were also partitioned into an independent 80-file validation and a 264-file development-excluded sensitivity cohort.

## Method
True cross-language voice conversion was achieved using the XTTS-v2 neural synthesis model. Speaker biometric features and clinical pathology were implicitly preserved via source-conditioned acoustic embeddings without linguistic entanglement. Downstream diagnostic validation utilized frozen `wav2vec2-large-xlsr-53` Layer 11 embeddings with Linear SVM and Logistic Regression classifiers.

## Language Conversion
Target-language conversion was highly reliable at scale. Autonomous generation of the 276-file cohort was 100% technically successful with no audio clipping. Independent Whisper Language Identification (LID) confirmed a 99.64% target-language identification rate (100.0% for Spanish$\to$German, 99.43% for German$\to$Spanish).

## Content Fidelity
Generated speech content was intelligible but asymmetric across directions. The overall mean Word Error Rate (WER) was 0.5747. German$\to$Spanish conversion exhibited exceptional fidelity (Mean WER = 0.4430), while Spanish$\to$German showed higher error rates (Mean WER = 0.8065), demonstrating variance in out-of-domain phonetic stability.

## Speaker Preservation
Biometric speaker identity was robustly maintained. Matched-pair cosine similarity (ECAPA-TDNN) averaged 0.5610, establishing a substantial discrimination margin of +0.4647 over the different-speaker negative control baseline (0.0963).

## Pathology Preservation
A majority of pathology-related acoustic contrasts were retained in the synthesized speech. Across 22 statistical contrasts, the overall pathology direction preservation was 59.1% (63.6% in ES$\to$DE, 72.7% in DE$\to$ES). Certain subtle perturbation cues were attenuated or reversed during neural resynthesis.

## Normalization Result
Acoustic centroid distance analysis demonstrated that generated PD speech was significantly closer to the Healthy Control centroid ($D_{orig} = 4.3870, D_{gen} = 3.8429, W = 2236.0, p < 0.0001$). This is consistent with partial acoustic normalization of the macroscopic feature distribution, though it did not completely collapse clinically relevant group separations.

## PD Classification
Cross-language Parkinson's disease classification was direction-dependent. German-trained classifiers evaluating Spanish speech converted into German showed initial diagnostic promise, while Spanish-trained classifiers evaluating German speech consistently showed no benefit.

## Independent Validation
In the independent 80-file out-of-sample validation, Spanish$\to$German conversion substantially improved cross-language classification (e.g., Logistic Regression $\Delta\text{UAR} = +0.1500$, nominal $p = 0.025$). German$\to$Spanish classification yielded negative deltas ($\Delta\text{UAR} = -0.1500$).

## Full-Cohort Sensitivity Analysis
Extended to the full 276-subject cohort, the Spanish$\to$German diagnostic gain attenuated to modest, non-significant levels ($\Delta\text{UAR} = +0.0400$ for SVM; FDR $q = 0.992$). German$\to$Spanish continued to show no diagnostic benefit.

## Development-Excluded Analysis
When the 12 pilot development subjects were strictly excluded ($N=264$), the cross-language diagnostic improvement was entirely absent for both Spanish$\to$German ($\Delta\text{UAR} = -0.0213$) and German$\to$Spanish ($\Delta\text{UAR} = -0.0118$).

## Final Conclusion
True cross-language voice conversion successfully overcomes the language barrier at the phonetic and lexical levels, substantially preserving speaker identity and retaining a majority of clinical pathology feature directions. However, acoustic-domain conversion alone does not inherently resolve cross-lingual classification distribution shifts. Downstream diagnostic utility is highly asymmetrical and attenuates broadly outside targeted validation sub-cohorts.

## Limitations
1. Pronounced directional asymmetry in downstream transfer.
2. Attenuation of diagnostic gains across the broader cohort indicates that explicit classifier domain adaptation is required.
3. Partial normalization and attenuation of fine-grained acoustic perturbation cues limit out-of-the-box acoustic biomarker robustness.
