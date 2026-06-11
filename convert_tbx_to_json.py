import xml.etree.ElementTree as ET
import json
import sys
import os

def parse_tbx(tbx_path):
    """
    Parses a TBX file (both TBX V2 and V3 structures) and returns a list of terminological entries.
    """
    if not os.path.exists(tbx_path):
        print(f"Error: File '{tbx_path}' does not exist.")
        return None

    try:
        # We parse using ElementTree
        tree = ET.parse(tbx_path)
        root = tree.getroot()
    except ET.ParseError as e:
        print(f"Error parsing XML: {e}")
        return None

    # Namespace handling
    # XML tags in TBX often have namespaces. We can strip namespaces or use them.
    # To be safe and generic, we will search with and without namespaces,
    # or implement a helper to get tag names ignoring namespaces.
    
    entries = []

    # Helper to strip namespace from tag name
    def clean_tag(tag):
        if '}' in tag:
            return tag.split('}', 1)[1]
        return tag

    # Find all entry elements. TBX V2 uses 'termEntry', TBX V3 uses 'conceptEntry'.
    # We can walk the tree to find them.
    for elem in root.iter():
        tag_name = clean_tag(elem.tag)
        if tag_name in ('termEntry', 'conceptEntry'):
            entry_id = elem.attrib.get('id', '')
            entry_data = {
                'id': entry_id,
                'languages': {}
            }
            
            # Extract basic fields at the entry level if any (e.g. descrip, admin, transac)
            # Typically these are metadata.
            metadata = {}
            for child in elem:
                child_tag = clean_tag(child.tag)
                if child_tag in ('descrip', 'admin', 'transac'):
                    type_attr = child.attrib.get('type', 'generic')
                    metadata[type_attr] = child.text.strip() if child.text else ''
            if metadata:
                entry_data['metadata'] = metadata

            # Look for langSet elements inside this entry
            for lang_set in elem.iter():
                if clean_tag(lang_set.tag) == 'langSet':
                    # Get language code (often xml:lang)
                    lang_code = None
                    for key, val in lang_set.attrib.items():
                        if clean_tag(key) == 'lang':
                            lang_code = val
                            break
                    if not lang_code:
                        continue
                    
                    if lang_code not in entry_data['languages']:
                        entry_data['languages'][lang_code] = []

                    # Look for term groups: 'tig' or 'ntig' (TBX V2), or 'termSec' (TBX V3)
                    for group in lang_set:
                        group_tag = clean_tag(group.tag)
                        if group_tag in ('tig', 'ntig', 'termSec'):
                            term_info = {}
                            
                            # Find the term
                            for sub_child in group.iter():
                                sub_tag = clean_tag(sub_child.tag)
                                if sub_tag == 'term':
                                    term_info['term'] = sub_child.text.strip() if sub_child.text else ''
                                elif sub_tag == 'termNote':
                                    type_attr = sub_child.attrib.get('type', 'note')
                                    term_info[f'note_{type_attr}'] = sub_child.text.strip() if sub_child.text else ''
                                elif sub_tag in ('descrip', 'admin', 'transac'):
                                    type_attr = sub_child.attrib.get('type', 'generic')
                                    term_info[type_attr] = sub_child.text.strip() if sub_child.text else ''
                            
                            if 'term' in term_info:
                                entry_data['languages'][lang_code].append(term_info)
            
            entries.append(entry_data)

    return entries

def main():
    if len(sys.argv) < 2:
        print("Usage: python convert_tbx_to_json.py <path_to_tbx_file> [path_to_output_json]")
        print("Example: python convert_tbx_to_json.py my_dict.tbx output.json")
        return

    tbx_file = sys.argv[1]
    if len(sys.argv) >= 3:
        json_file = sys.argv[2]
    else:
        # Default to replacing extension with .json
        base, _ = os.path.splitext(tbx_file)
        json_file = base + ".json"

    print(f"Parsing '{tbx_file}'...")
    data = parse_tbx(tbx_file)
    
    if data is not None:
        # Convert to list of "ENG_term == VIE_term" strings
        bilingual_list = []
        seen = set()
        
        for entry in data:
            en_terms = []
            vi_terms = []
            for lang, term_list in entry.get('languages', {}).items():
                lang_lower = lang.lower()
                if lang_lower.startswith('en'):
                    for t in term_list:
                        if 'term' in t and t['term']:
                            en_terms.append(t['term'])
                elif lang_lower.startswith('vi'):
                    for t in term_list:
                        if 'term' in t and t['term']:
                            vi_terms.append(t['term'])
            
            # Map each English term to the Vietnamese term(s)
            if en_terms and vi_terms:
                # If there are multiple Vietnamese terms, join them with a semicolon
                vi_translation = "; ".join(vi_terms)
                for en_term in en_terms:
                    pair_str = f"{en_term} == {vi_translation}"
                    if pair_str not in seen:
                        seen.add(pair_str)
                        bilingual_list.append(pair_str)

        try:
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(bilingual_list, f, ensure_ascii=False, indent=2)
            print(f"Successfully converted '{tbx_file}' to '{json_file}' (ENG == VIE format)")
            print(f"Total entries processed: {len(data)}")
            print(f"Total bilingual pairs: {len(bilingual_list)}")
        except Exception as e:
            print(f"Error writing to JSON file: {e}")

if __name__ == "__main__":
    main()
