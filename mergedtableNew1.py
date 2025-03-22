import os
import tempfile
import pandas as pd
from pdf2image import convert_from_path
from img2table.ocr import TesseractOCR
from img2table.document import Image

def pdf_to_table(pdf_path, output_csv=None, merge_tables=True):
    """
    Convert a PDF file to images, then extract tables from those images.
    Supports merging tables that span multiple pages.
    Replaces newlines in table cells with spaces.
    
    Args:
        pdf_path (str): Path to the PDF file
        output_csv (str, optional): Path to save CSV output. If None, returns data without saving.
        merge_tables (bool): Whether to attempt to merge tables across pages
    
    Returns:
        list: List of extracted tables data
    """
    print(f"Processing PDF: {pdf_path}")
    
    # Step 1: Convert PDF to images
    print("Converting PDF to images...")
    with tempfile.TemporaryDirectory() as temp_dir:
        images = convert_from_path(pdf_path)
        image_paths = []
        
        # Save images temporarily
        for i, image in enumerate(images):
            image_path = os.path.join(temp_dir, f'page_{i+1}.png')
            image.save(image_path, 'PNG')
            image_paths.append(image_path)
        
        print(f"Converted {len(images)} pages to images")
        
        # Step 2: Extract tables from each image
        print("Extracting tables from images...")
        page_tables = []  # List of tables by page
        
        # Initialize OCR engine (using Tesseract)
        ocr = TesseractOCR()
        
        for page_num, image_path in enumerate(image_paths):
            # Process image and extract tables
            print(f"Processing image: {image_path}")
            img_doc = Image(image_path)
            
            # Extract tables
            tables = img_doc.extract_tables(ocr=ocr)
            
            if tables:
                print(f"Found {len(tables)} tables in {os.path.basename(image_path)}")
                
                # Store tables with page number
                page_tables.append({
                    'page_num': page_num + 1,
                    'tables': []
                })
                
                for i, table in enumerate(tables):
                    # Try different ways to access the DataFrame
                    try:
                        # Try common methods to access DataFrame based on different versions
                        if hasattr(table, 'df'):
                            df = table.df
                        elif hasattr(table, 'get_df'):
                            df = table.get_df()
                        elif hasattr(table, 'to_dataframe'):
                            df = table.to_dataframe()
                        else:
                            # Try to access the data attribute
                            if hasattr(table, 'data'):
                                df = pd.DataFrame(table.data)
                            else:
                                print(f"Could not extract DataFrame from table on page {page_num + 1}")
                                # Inspect table object
                                print(f"Table attributes: {dir(table)}")
                                continue
                        
                        if not df.empty:
                            # Replace newlines with spaces in all cells
                            df = df.applymap(lambda x: x.replace('\n', ' ') if isinstance(x, str) else x)
                            
                            # Store table structure info
                            table_info = {
                                'table_num': i + 1,
                                'df': df,
                                'num_cols': len(df.columns),
                                'num_rows': len(df)
                            }
                            page_tables[-1]['tables'].append(table_info)
                    except Exception as e:
                        print(f"Error extracting table: {str(e)}")
            else:
                print(f"No tables found in {os.path.basename(image_path)}")
        
        # Step 3: Identify and merge tables that span multiple pages
        all_tables = []
        if merge_tables and len(page_tables) > 1:
            print("\nAttempting to identify and merge multi-page tables...")
            
            current_table = None
            merged_tables = []
            
            for page_data in page_tables:
                page_num = page_data['page_num']
                
                for table_info in page_data['tables']:
                    df = table_info['df']
                    
                    # If this is the first table or the column count doesn't match the previous one,
                    # treat it as a new table
                    if current_table is None or table_info['num_cols'] != current_table['num_cols']:
                        # If we were tracking a table, add it to the merged list
                        if current_table is not None:
                            merged_tables.append(current_table['df'])
                        
                        # Start tracking a new table
                        header_row = df.iloc[0] if not df.empty else None
                        current_table = {
                            'df': df,
                            'num_cols': table_info['num_cols'],
                            'header_row': header_row
                        }
                    else:
                        # This table likely continues from the previous page
                        # Concatenate with the current table, but don't duplicate headers
                        if df.iloc[0].equals(current_table['header_row']):
                            # If the first row matches the header of the previous table part,
                            # skip it to avoid duplicate headers
                            df_to_append = df.iloc[1:] if len(df) > 1 else pd.DataFrame(columns=df.columns)
                        else:
                            df_to_append = df
                        
                        current_table['df'] = pd.concat([current_table['df'], df_to_append], 
                                                        ignore_index=True)
                        
                        print(f"Merged table part from page {page_num}")
            
            # Add the last table if it exists
            if current_table is not None:
                merged_tables.append(current_table['df'])
            
            all_tables = merged_tables
            print(f"Created {len(all_tables)} merged tables")
        else:
            # Just flatten the tables if not merging
            for page_data in page_tables:
                for table_info in page_data['tables']:
                    all_tables.append(table_info['df'])
        
        # Step 4: Save tables if output path is provided
        if output_csv and all_tables:
            for i, df in enumerate(all_tables):
                base_name = os.path.splitext(output_csv)[0]
                ext = os.path.splitext(output_csv)[1]
                table_csv = f"{base_name}_table{i+1}{ext}"
                df.to_csv(table_csv, index=False)
                print(f"Saved merged table to: {table_csv}")
    
    return all_tables

if __name__ == "__main__":
    # Example usage
    pdf_path = "./data/HDFCNew.pdf"  # Replace with your PDF file
    output_csv = "extracted_tables.csv"
    
    # Process the PDF and extract tables with table merging
    tables = pdf_to_table(pdf_path, output_csv, merge_tables=True)
    
    # Display some information about extracted tables
    print(f"\nExtraction complete. Found {len(tables)} tables total.")
    for i, table in enumerate(tables):
        print(f"\nTable {i+1} shape: {table.shape}")
        print("Table preview:")
        print(table.head())