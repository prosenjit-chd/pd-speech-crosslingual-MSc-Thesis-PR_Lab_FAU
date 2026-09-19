# Target Text Normalization and Segmentation Log

## 1. German Target Text (Spanish $\to$ German Conversion)

### Original Text:
```text
Gestern ging ich zum Arzt. „Was fehlt Ihnen?“, fragte er mich. Ich sagte: „Ach Herr Doktor, wo immer ich den Finger hinlege, tut es weh!“ „Haben Sie einen gebrochenen Nagel?“ „Ja.“ „Nun, dann wissen wir ja, was es ist! Geben Sie Ihren Scheck am Ausgang ab.“
```

### Normalized Text:
```text
Gestern ging ich zum Arzt. Was fehlt Ihnen? fragte er mich. Ich sagte: Ach Herr Doktor, wo immer ich den Finger hinlege, tut es weh! Haben Sie einen gebrochenen Nagel? Ja. Nun, dann wissen wir ja, was es ist! Geben Sie Ihren Scheck am Ausgang ab.
```

### Normalization Actions Applied:
1. **Unicode Dialogue Quote Removal**:
   - Replaced German low quote `„` (U+201E) and high quote `“` (U+201C) with standard conversational punctuation.
   - Removed compound punctuation `?“,` which caused token boundary collisions in XTTS-v2 tokenizer.
2. **Strict Lexical Invariance**:
   - Exactly 46 words in both original and normalized versions.
   - Zero vocabulary alterations, zero semantic shortening, zero structural drift.
3. **Discrete Sentence Segmentation**:
   The passage was segmented into 6 natural conversational dialogue turns:
   - **Segment 1** (5 words): `"Gestern ging ich zum Arzt."`
   - **Segment 2** (6 words): `"Was fehlt Ihnen? fragte er mich."`
   - **Segment 3** (14 words): `"Ich sagte: Ach Herr Doktor, wo immer ich den Finger hinlege, tut es weh!"`
   - **Segment 4** (6 words): `"Haben Sie einen gebrochenen Nagel?"`
   - **Segment 5** (1 word): `"Ja."`
   - **Segment 6** (14 words): `"Nun, dann wissen wir ja, was es ist! Geben Sie Ihren Scheck am Ausgang ab."`

---

## 2. Spanish Target Text (German $\to$ Spanish Conversion)

### Original Text:
```text
Una enorme invasión, protegida por las autoridades, azota actualmente las costas del Golfo y del Pacífico de México: como cada año en junio, cientos de miles de tortugas salen del mar para poner sus huevos en tierra firme. Solo en las cercanías de Tampico se observaron unas 5.000 tortugas. En total, se espera en las próximas semanas una invasión de más de medio millón de tortugas. Las autoridades mexicanas hacen vigilar cuidadosamente las zonas de puesta para evitar el robo de huevos y asegurar una descendencia suficiente de tortugas.
```

### Normalized Text:
```text
Una enorme invasión, protegida por las autoridades, azota actualmente las costas del Golfo y del Pacífico de México. Como cada año en junio, cientos de miles de tortugas salen del mar para poner sus huevos en tierra firme. Solo en las cercanías de Tampico se observaron unas cinco mil tortugas. En total, se espera en las próximas semanas una invasión de más de medio millón de tortugas. Las autoridades mexicanas hacen vigilar cuidadosamente las zonas de puesta para evitar el robo de huevos y asegurar una descendencia suficiente de tortugas.
```

### Normalization Actions Applied:
1. Replaced colon separator with natural sentence boundary to avoid running clause drift.
2. Normalized `5.000` to `cinco mil` to ensure consistent phonetic pronunciation across synthesis engines.
3. Segmented into 5 natural expository sentences:
   - **Segment 1** (18 words): `"Una enorme invasión, protegida por las autoridades, azota actualmente las costas del Golfo y del Pacífico de México."`
   - **Segment 2** (21 words): `"Como cada año en junio, cientos de miles de tortugas salen del mar para poner sus huevos en tierra firme."`
   - **Segment 3** (11 words): `"Solo en las cercanías de Tampico se observaron unas cinco mil tortugas."`
   - **Segment 4** (18 words): `"En total, se espera en las próximas semanas una invasión de más de medio millón de tortugas."`
   - **Segment 5** (22 words): `"Las autoridades mexicanas hacen vigilar cuidadosamente las zonas de puesta para evitar el robo de huevos y asegurar una descendencia suficiente de tortugas."`
