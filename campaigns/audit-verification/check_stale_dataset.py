import glob
import json

for f in sorted(glob.glob("data/**/*manifest*.json", recursive=True))[:3]:
    m = json.load(open(f))
    print(f)
    tr = m.get("trajectories", [{}])[0]
    print("  observation_times[:6]      :", tr.get("observation_times", [])[:6])
    print("  actual_observation_times    :", "actual_observation_times" in tr)
    print("  observation_scheduler       :", m.get("observation_scheduler"))
