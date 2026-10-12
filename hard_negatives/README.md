# Hard-Negative Dataset for SiteGuard AI

## Overview
This directory contains difficult, hard-negative and edge-case training samples specifically designed to eliminate two major false-positive errors in construction safety PPE detection:
1. **Turban / Cloth Wraps detected as Safety Helmet** (False Positive)
2. **Ordinary Inner Baniyan / Tank Tops detected as Reflective Safety Vest** (False Positive)

## Class Definitions & Annotation Rules
The dataset adheres strictly to the Construction-PPE ontology (11 classes):
* `0: helmet`: **Only** genuine rigid construction safety helmets (hardhats). Must NOT be labeled on turbans, cloth wraps, towels, caps, or bare heads.
* `2: vest`: **Only** genuine retroreflective high-visibility construction safety vests with fluorescent colors (neon yellow/orange/green) and reflective stripes. Must NOT be labeled on ordinary sleeveless undershirts, white cotton baniyans, tank tops, or civilian shirts.
* `5: none`: Represents the unequipped torso/body (e.g., a person wearing an inner baniyan, civilian shirt, or casual clothing without a safety vest).
* `6: Person`: Full bounding box enclosing the construction worker / personnel.
* `7: no_helmet`: Bounding box of the head when no construction safety helmet is worn (including workers wearing turbans, pagris, gamchas, cloth wraps, caps, or bare heads).

## Process for Adding and Annotating New Hard-Negative Images
1. **Image Sourcing**:
   - Collect high-resolution, genuine images of workers under actual site conditions.
   - Include diverse head coverings (turbans, gamchas, towels, scarves, caps) and varied upper clothing (sleeveless inner baniyans, gym tanks, casual t-shirts, untucked shirts).
2. **Annotation Rules**:
   - Worker wearing a turban: Annotate head with class `7 (no_helmet)`. Enclose worker with class `6 (Person)`. Do NOT place class `0 (helmet)`!
   - Worker wearing an inner baniyan: Annotate upper body with class `5 (none)`. Enclose worker with class `6 (Person)`. Do NOT place class `2 (vest)`!
   - Worker with genuine hardhat: Annotate hardhat with class `0 (helmet)`.
   - Worker with genuine reflective vest: Annotate vest with class `2 (vest)`.
3. **Storage**:
   - Save `.jpg` images to `hard_negatives/images/{train|val|test}/`
   - Save corresponding `.txt` YOLO annotations to `hard_negatives/labels/{train|val|test}/`
4. **Data Leakage Prevention**:
   - Never place near-identical or transformed frames from the same video or camera burst across both train and test splits.
