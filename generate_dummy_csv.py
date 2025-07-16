import os
import pandas as pd
import numpy as np  # ← importer numpy
from datetime import datetime, timedelta

# 1. Créez le dossier s'il n'existe pas
output_dir = "data/07_model_output"
os.makedirs(output_dir, exist_ok=True)

# 2. Simulez quelques fichiers CSV
# par exemple ici 3 fichiers, chacun contenant 10 lignes horodatées
for part in range(3):
    df = pd.DataFrame(
        {
            "flight_id": range(part * 10, part * 10 + 10),
            "delay_minutes": (np.random.rand(10) * 60).astype(int),  # ← usar np.random au lieu de pd.np
            "timestamp": [(datetime.now() - timedelta(days=i)).isoformat() for i in range(10)],
        }
    )
    filename = f"part-{part:05d}.csv"
    df.to_csv(os.path.join(output_dir, filename), index=False)
    print(f"→ {filename} généré ({len(df)} lignes)")
