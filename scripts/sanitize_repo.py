import json
import os

notebook_path = r'c:\Users\Harsha\Documents\AToM-FM_Prototype\notebooks\AToM_FM_Training.ipynb'
script_path = r'c:\Users\Harsha\Documents\AToM-FM_Prototype\scripts\update_notebook.py'
key_to_hide = "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"
placeholder = "YOUR_WANDB_API_KEY"

# 1. Update Notebook
if os.path.exists(notebook_path):
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source = "".join(cell['source'])
            if key_to_hide in source:
                cell['source'] = [line.replace(key_to_hide, placeholder) for line in cell['source']]
    
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    print("Masked key in notebook.")

# 2. Update Script
if os.path.exists(script_path):
    with open(script_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    if key_to_hide in content:
        content = content.replace(key_to_hide, placeholder)
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(content)
    print("Masked key in update script.")

print("Sensitive information masked successfully.")
