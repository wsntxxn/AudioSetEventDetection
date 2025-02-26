from pathlib import Path
import json
import pandas as pd


AUDIOSET_ROOT_DIR = "/mnt/cloudstorfs/public/shared/data/raa/AudioSet"
AUDIOSET_ROOT_DIR = Path(AUDIOSET_ROOT_DIR)


weak_mids = []
mid_to_label = {}
df = pd.read_csv(AUDIOSET_ROOT_DIR / "metadata/class_labels_indices.csv")
for _, row in df.iterrows():
    mid = row["mid"]
    label = row["display_name"]
    # "[event_label]", remove ""
    weak_mids.append(mid)
    mid_to_label[mid] = label
weak_mids = set(weak_mids)

ontology = json.load(open(AUDIOSET_ROOT_DIR / "metadata/ontology.json"))
all_mids = []
all_mid_to_label = {}
for event_label in ontology:
    all_mids.append(event_label["id"])
    all_mid_to_label[event_label["id"]] = event_label["name"]
all_mids = set(all_mids)

print("---------------event label from csv---------------")
train_df = pd.read_csv(AUDIOSET_ROOT_DIR / "strong_label/train_strong.tsv", sep="\t")
train_mids = set(train_df["label"].unique())

eval_df = pd.read_csv(AUDIOSET_ROOT_DIR / "strong_label/eval_strong.tsv", sep="\t")
eval_mids = set(eval_df["label"].unique())

eval_framed_posneg_df = pd.read_csv(AUDIOSET_ROOT_DIR / "strong_label/eval_strong_framed_posneg.tsv", sep="\t")
eval_framed_posneg_mids = set(eval_framed_posneg_df["label"].unique())

print("train: ", len(train_mids))
print("eval: ", len(eval_mids))
print("eval_framed_posneg: ", len(eval_framed_posneg_mids))
print("--------------------------------------------------")


print("---------------event label from metadata---------------")
strong_mids = []
with open(AUDIOSET_ROOT_DIR / "strong_label/mid_to_display_name.tsv", "r") as reader:
    for line in reader.readlines():
        mid, label = line.strip().split("\t")
        # "[event_label]", remove ""
        strong_mids.append(mid)
        if mid not in mid_to_label:
            mid_to_label[mid] = label
strong_mids = set(strong_mids)
print("all strong mids: ", len(strong_mids))
print("-------------------------------------------------------")

assert train_mids.issubset(strong_mids)
assert eval_mids.issubset(strong_mids)
assert eval_framed_posneg_mids.issubset(strong_mids)

only_weak = weak_mids - (weak_mids & strong_mids)
only_strong = strong_mids - (weak_mids & strong_mids)

print("---------------weak label only---------------")
for mid in only_weak:
    print(mid_to_label[mid])
print("---------------------------------------------")

print("---------------strong label only--------------")
for mid in only_strong:
    print(mid_to_label[mid])
print("---------------------------------------------")

if strong_mids.issubset(all_mids):
    print("all strong mids are from the original 632 mids")
else:
    only_strong = strong_mids - (all_mids & strong_mids)
    print("---------------strong label only--------------")
    for mid in only_strong:
        print(mid_to_label[mid])
    print("---------------------------------------------")
