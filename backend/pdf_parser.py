import pdfplumber
import re
import io

def extract_data_from_pdf(file_source):
    """
    Scans a PDF for ALL soil nutrient values required by complex models.
    Matches variations like 'OC', 'Org. Carbon', 'Zn', 'Cu', etc.
    """
    # 1. Initialize Default Dictionary
    # We set default values to 0.0 or neutral (pH 7.0) to prevent model crashes if data is missing.
    extracted_data = {
        # Primary Nutrients
        'Nitrogen': 0.0, 'N': 0.0,
        'Phosphorous': 0.0, 'P': 0.0,
        'Potassium': 0.0, 'K': 0.0,
        
        # Physical Properties
        'ph': 7.0,          # Neutral default
        'EC': 0.0,          # Electrical Conductivity
        'OC': 0.0,          # Organic Carbon
        
        # Secondary/Micro Nutrients
        'Sulphur': 0.0, 'S': 0.0,
        'Zinc': 0.0, 'Zn': 0.0,
        'Iron': 0.0, 'Fe': 0.0,
        'Copper': 0.0, 'Cu': 0.0,
        'Manganese': 0.0, 'Mn': 0.0,
        'Boron': 0.0, 'B': 0.0
    }

    text = ""
    
    # 2. Extract Text from PDF (Stream or File Path)
    try:
        if isinstance(file_source, str):
            with pdfplumber.open(file_source) as pdf:
                for page in pdf.pages:
                    text += (page.extract_text() or "") + "\n"
        else:
            with pdfplumber.open(file_source) as pdf:
                for page in pdf.pages:
                    text += (page.extract_text() or "") + "\n"
    except Exception as e:
        print(f"❌ Error reading PDF: {e}")
        return None

    # Debug: Check text content (Optional)
    # print(f"📄 Extracted Text: {text[:500]}...")

    # 3. Robust Regex Patterns
    # These patterns look for the Label + separators (: - =) + Number
    patterns = {
        # --- Primary ---
        'Nitrogen': [r'Nitrogen.*?(\d+\.?\d*)', r'Available N.*?(\d+\.?\d*)', r'\bN\s*[:=-]\s*(\d+\.?\d*)'],
        'Phosphorous': [r'Phosphorus.*?(\d+\.?\d*)', r'Available P.*?(\d+\.?\d*)', r'\bP\s*[:=-]\s*(\d+\.?\d*)'],
        'Potassium': [r'Potassium.*?(\d+\.?\d*)', r'Available K.*?(\d+\.?\d*)', r'\bK\s*[:=-]\s*(\d+\.?\d*)'],
        
        # --- Physical ---
        'ph': [r'\bpH\b.*?(\d+\.?\d*)', r'Reaction\s*\(pH\).*?(\d+\.?\d*)'],
        'EC': [r'Electrical Conductivity.*?(\d+\.?\d*)', r'\bEC\b.*?(\d+\.?\d*)'],
        'OC': [r'Organic Carbon.*?(\d+\.?\d*)', r'\bOC\b.*?(\d+\.?\d*)'],
        
        # --- Secondary ---
        'Sulphur': [r'Sulphur.*?(\d+\.?\d*)', r'\bS\b\s*[:=-]\s*(\d+\.?\d*)'],
        'Zinc': [r'Zinc.*?(\d+\.?\d*)', r'\bZn\b\s*[:=-]\s*(\d+\.?\d*)'],
        'Iron': [r'Iron.*?(\d+\.?\d*)', r'\bFe\b\s*[:=-]\s*(\d+\.?\d*)'],
        'Copper': [r'Copper.*?(\d+\.?\d*)', r'\bCu\b\s*[:=-]\s*(\d+\.?\d*)'],
        'Manganese': [r'Manganese.*?(\d+\.?\d*)', r'\bMn\b\s*[:=-]\s*(\d+\.?\d*)'],
        'Boron': [r'Boron.*?(\d+\.?\d*)', r'\bB\b\s*[:=-]\s*(\d+\.?\d*)']
    }

    # 4. Search and Extract
    print("\n🔍 Scanning PDF for Soil Parameters...")
    for key, regex_list in patterns.items():
        for pattern in regex_list:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                value = float(match.group(1))
                extracted_data[key] = value
                
                # Normalize keys for Models (Sync short codes with full names)
                # This ensures if we find "Zn", extracted_data['Zinc'] also gets set.
                if key == 'Nitrogen': extracted_data['N'] = value
                if key == 'Phosphorous': extracted_data['P'] = value
                if key == 'Potassium': extracted_data['K'] = value
                if key == 'Sulphur': extracted_data['S'] = value
                if key == 'Zinc': extracted_data['Zn'] = value
                if key == 'Iron': extracted_data['Fe'] = value
                if key == 'Copper': extracted_data['Cu'] = value
                if key == 'Manganese': extracted_data['Mn'] = value
                if key == 'Boron': extracted_data['B'] = value
                
                # print(f"   ✅ Found {key}: {value}") # Debug print
                break 

    # 5. Add Placeholders for External Data (Weather)
    # app.py will fill these in using an API or user input
    extracted_data['Temperature'] = None
    extracted_data['Humidity'] = None
    extracted_data['Rainfall'] = None
    extracted_data['Moisture'] = None 

    return extracted_data