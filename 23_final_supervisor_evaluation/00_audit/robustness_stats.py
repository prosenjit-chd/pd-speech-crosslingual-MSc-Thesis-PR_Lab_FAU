import pandas as pd
import numpy as np

df = pd.read_csv('C:/pd-speech-crosslingual/23_final_supervisor_evaluation/13_thesis_ready/Table2_Sensitivity_N264.csv')
diff = df['Diff'].abs()
print('Mean abs diff:', diff.mean())
print('Median abs diff:', diff.median())
print('Max abs diff:', diff.max())
# Direction preserving: if uar_N276 > 0.5 and uar_N264 > 0.5 (or both < 0.5)
# Actually the prompt says "proportion preserving direction of the conversion effect". 
# The table has diff = uar_N276 - uar_N264. The conversion effect direction is Orig vs Conv. Wait, Table2 is just the max UAR per scenario.
# "proportion preserving direction of the conversion effect" might refer to whether cross-lingual conversion improves or degrades.
# Let's just output the basic stats first.
