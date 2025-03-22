from pypdf import PdfReader, PdfWriter

def extract_pdf_pages(input_pdf_path, output_pdf_path, start_page, end_page):
    """
    Extracts a range of pages from an input PDF and saves them to a new PDF.

    Args:
        input_pdf_path (str): Path to the input PDF file.
        output_pdf_path (str): Path to save the output PDF file.
        start_page (int): The starting page number (1-based index).
        end_page (int): The ending page number (1-based index).
    """
    try:
        with open(input_pdf_path, "rb") as input_file:
            reader = PdfReader(input_file)
            writer = PdfWriter()

            if start_page < 1 or end_page > len(reader.pages) or start_page > end_page:
                raise ValueError("Invalid page range.")

            for page_num in range(start_page - 1, end_page):  # Convert to 0-based index
                writer.add_page(reader.pages[page_num])

            with open(output_pdf_path, "wb") as output_file:
                writer.write(output_file)

        print(f"Pages {start_page} to {end_page} extracted successfully to {output_pdf_path}")

    except FileNotFoundError:
        print(f"Error: Input PDF file not found at {input_pdf_path}")
    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# Example usage:
input_pdf = "./data/HDFCNew.pdf"  # Replace with your input PDF file path
output_pdf = "./data/HDFC1.pdf" # Replace with your desired output PDF file path
start_page_num = 1
end_page_num = 1

extract_pdf_pages(input_pdf, output_pdf, start_page_num, end_page_num)