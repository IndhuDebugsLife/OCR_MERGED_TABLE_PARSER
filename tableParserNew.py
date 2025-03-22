from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer, LTChar

def extract_text_with_coordinates(pdf_path, page_num=0):
    """Extracts text and coordinates from a PDF page."""
    text_data = []
    for page_layout in extract_pages(pdf_path, page_numbers=[page_num]):
        for element in page_layout:
            if isinstance(element, LTTextContainer):
                text = element.get_text().strip()
                if text:
                    x0, y0, x1, y1 = element.bbox
                    text_data.append((text, x0, y0, x1, y1))
    return text_data

def analyze_merged_cells(text_data):
    """Analyzes text coordinates to identify potential merged cells."""
    merged_cells = []
    i = 0
    while i < len(text_data) - 1:
        text1, x0_1, y0_1, x1_1, y1_1 = text_data[i]
        text2, x0_2, y0_2, x1_2, y1_2 = text_data[i + 1]

        # Check for horizontal overlap (potential horizontal merge)
        if abs(y0_1 - y0_2) < 5 and (x1_1 >= x0_2 or x1_2>=x0_1): #5 is a tolerance value.
            merged_cells.append((text1, text2, x0_1, y0_1, x1_2, y1_1)) #store the merged cell values.
            i += 2 # Skip the next element
        else:
            i += 1
    return merged_cells

if __name__ == "__main__":
    pdf_file = "./data/HDFCNew.pdf"  # Replace with your PDF path
    text_data = extract_text_with_coordinates(pdf_file)
    merged_cells = analyze_merged_cells(text_data)

    for cell in merged_cells:
        print(f"Potential merged cell: {cell}")