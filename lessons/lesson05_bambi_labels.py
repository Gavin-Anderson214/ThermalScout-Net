import csv
from pathlib import Path
from collections import Counter

# Locate the downloaded BAMBI annotations.
root = Path(
    r"D:\College Work\Intro To AI\Intro to AI project"
    r"\ThermalScout-Net_GitHub_Ready\ThermalScout-Net"
    r"\data\bambi_source"
)

files = list(root.rglob("47_gt.txt"))

if not files:
    raise FileNotFoundError("Flight 47 annotations not found")

# Use one copy so duplicate downloads aren't counted twice.
annotation_file = files[0]
print("Reading:", annotation_file)

species_counts = Counter()
class_ids = {}
frames = set()

with annotation_file.open(encoding="utf-8") as file:
    reader = csv.reader(file)

    for row in reader:
        if len(row) != 13:
            continue

        frame = int(row[0])
        class_id = int(row[7])
        species = row[9].strip()

        frames.add(frame)
        species_counts[species] += 1
        class_ids.setdefault(species, set()).add(class_id)

print("\nSpecies found:")

for species, count in species_counts.most_common():
    print(f"{species}: {count} annotations")
    print(f"  class IDs: {sorted(class_ids[species])}")

human_count = 0
animal_count = 0
excluded_count = 0

for species, count in species_counts.items():
    if species.startswith("Homo sapiens"):
        human_count += count
    elif species.lower().startswith(("unknown", "no-animal")):
        excluded_count += count
    else:
        animal_count += count

print("\nThermalScout-Net class totals:")
print("Human annotations:", human_count)
print("Animal annotations:", animal_count)
print("Excluded annotations:", excluded_count)
print("Unique annotated frames:", len(frames))
