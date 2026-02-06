# SSAR: Surface Anomaly Recognition Model

This repository contains the final evaluation results for the **SSAR (Semiconductor Surface Anomaly Recognition)** model. The project focuses on high-precision defect detection and classification in semiconductor manufacturing.

## Performance Overview

The model achieves near-perfect classification across 9 distinct categories. The evaluation was performed on a balanced test set of approximately 3,600 samples.

### Final Confusion Matrix
Below is the visualization of the model's performance on the test dataset:

![Confusion Matrix](<img width="932" height="527" alt="Screenshot 2026-02-06 212030" src="https://github.com/user-attachments/assets/8eceb4af-f603-42ec-9f54-30e6796cdb0e" />
)

## Analysis of Results

The model demonstrates exceptional accuracy, particularly in identifying critical electrical defects like **Shorts**, **Opens**, and **Corrosion**.

### Key Observations:
* **High Precision:** Categories such as `Corrosion` and `Shorts` show 0% false positives from other classes.
* **Minor Confusion:** There is a slight overlap between `Clean` and `Other` categories (approx. 4% error rate). This is expected as "Other" often contains subtle artifacts that resemble a clean surface.
* **Particle Sensitivity:** The `Particles` class acts as a slight attractor for minor misclassifications from `LER` and `Scratches`, likely due to shared geometric features.

## Classes Identified
| Category | Description |
| :--- | :--- |
| **Clean** | No defects present. |
| **Corrosion** | Chemical degradation of the metallic surface. |
| **LER** | Line Edge Roughness; deviations in the edge of a feature. |
| **MalformedVia** | Incorrectly formed vertical interconnect access points. |
| **Opens** | Breaks in the intended conductive path. |
| **Particles** | Foreign material or dust on the surface. |
| **Scratches** | Physical abrasions on the wafer surface. |
| **Shorts** | Unintended connections between conductive paths. |
| **Other** | Miscellaneous anomalies not covered by the above. |

---
