# Target-Language Translation Pairs Summary

## 1. Principle of Semantic Equivalence & Frozen Design
In strict accordance with master instructions:
- **No Word-by-Word Translation**: Literal translation leads to unnatural syntax. The translations below maintain semantic nuance, conversational cadence, and correct syntactic structure in the target language.
- **Strictly Frozen**: Once established, these translations are **immutable**. No modifications may be made based on downstream classifier performance.

## 2. Spanish $\to$ German Conversion Pair
- **Source Language**: Spanish (PC-GITA)
- **Source Prompt**:
  > *„Ayer fui al médico. «¿Qué le pasa?», me preguntó. Yo le dije: «¡Ay doctor, donde pongo el dedo me duele!». «¿Tiene la uña rota?». «Sí». «¡Pues ya sabemos qué es! Deje su cheque a la salida».“*
- **Target Language**: German
- **Target Text**:
  > *„Gestern ging ich zum Arzt. „Was fehlt Ihnen?“, fragte er mich. Ich sagte: „Ach Herr Doktor, wo immer ich den Finger hinlege, tut es weh!“ „Haben Sie einen gebrochenen Nagel?“ „Ja.“ „Nun, dann wissen wir ja, was es ist! Geben Sie Ihren Scheck am Ausgang ab.““*

## 3. German $\to$ Spanish Conversion Pair
- **Source Language**: German (Sabine Skoda Protocol)
- **Source Prompt**:
  > *„Von einer gewaltigen, von den Behörden geschützten Invasion wird zur Zeit die Golf- und Pazifikküste Mexikos heimgesucht: Wie alljährlich im Juni kommen Hunderttausende von Schildkröten aus dem Meer, um an Land ihre Eier abzulegen. Allein in der Nähe von Tampico wurden etwa 5000 Schildkröten beobachtet. Insgesamt wird in den kommenden Wochen mit einer Invasion von mehr als einer halben Million Schildkröten gerechnet. Die mexikanischen Behörden lassen die Legeplätze sorgfältig bewachen, um den Diebstahl von Eiern zu verhindern und ausreichend Schildkrötennachwuchs sicherzustellen.“*
- **Target Language**: Spanish
- **Target Text**:
  > *„Una enorme invasión, protegida por las autoridades, azota actualmente las costas del Golfo y del Pacífico de México: como cada año en junio, cientos de miles de tortugas salen del mar para poner sus huevos en tierra firme. Solo en las cercanías de Tampico se observaron unas 5.000 tortugas. En total, se espera en las próximas semanas una invasión de más de medio millón de tortugas. Las autoridades mexicanas hacen vigilar cuidadosamente las zonas de puesta para evitar el robo de huevos y asegurar una descendencia suficiente de tortugas.“*

## 4. Verification
Total pairs registered: **276**. Status: **FROZEN**.
