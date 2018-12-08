import json

base_fn = 'answer-base.json'
ao_fn = 'model_out_ao/answer.json'
merge_fn = 'answer-merged.json'

with open(base_fn) as f:
    base = json.load(f)
with open(ao_fn) as f:
    ao = json.load(f)

new = {}
for key in ao:
    if ao[key] != "":
        new[key] = base[key]
    else:
        new[key] = ""
with open(merge_fn, 'w') as f:
    json.dump(new, f)
