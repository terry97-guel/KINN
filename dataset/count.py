# %%
import json





for dataset in ["FINGER", 'ABAQUS', 'ELASTICA']:
    data = json.load(open(f'{dataset}.json', 'r'))

    total_length = 0
    for split in ['train', 'val', 'test', 'ext']:
        length = len(data[split]['position'])
        print(f"dataset[{split}]['position'] legnth", length)
        total_length = total_length + length
        
    print(f"dataset[total] length {total_length}")
    print("====================================")