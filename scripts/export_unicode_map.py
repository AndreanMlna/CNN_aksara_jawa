# File: scripts/export_unicode_map.py
import json
import os
from core import config
from core.aksara_data import get_aksara_details

def export_map():
    with open(config.CLASS_MAP_PATH, 'r') as f:
        class_indices = json.load(f)

    all_data = {}
    grouped_by_category = {}

    for class_name, idx in class_indices.items():
        details = get_aksara_details(class_name)
        details['index'] = idx
        all_data[class_name] = details

        cat = details['category']
        if cat not in grouped_by_category:
            grouped_by_category[cat] = []
        grouped_by_category[cat].append(details)

    os.makedirs('web/js', exist_ok=True)
    js_content = f"""// Auto-generated Aksara Jawa Metadata and Unicode Map
// Generated from trained model class index
window.AKSARA_DATABASE = {json.dumps(all_data, ensure_ascii=False, indent=2)};
window.AKSARA_CATEGORIES = {json.dumps(grouped_by_category, ensure_ascii=False, indent=2)};
"""
    with open('web/js/unicode_map.js', 'w', encoding='utf-8') as f:
        f.write(js_content)
    print("✅ Successfully generated web/js/unicode_map.js with", len(all_data), "classes!")

if __name__ == '__main__':
    export_map()
