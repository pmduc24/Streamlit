import base64
from fpdf import FPDF
from io import BytesIO
import time

def create_download_link(val, filename):
    b64 = base64.b64encode(val)  # val looks like b'...'
    return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="{filename}.pdf">Download file</a>'

def export_as_pdf(df, total_money):
    pdf = FPDF()
    pdf.add_page()
    
    # Title
    pdf.set_font('Arial', 'B', 26)
    pdf.cell(0, 10, 'SHOP RECEIPT', 0, 1, 'C')
    pdf.ln(3)
    
    current_time = time.strftime("%Y-%m-%d %H:%M:%S")
    pdf.set_font('Arial', '', 12)
    pdf.cell(0, 6, f'Date and Time: {current_time}', 0, 1, 'C')
    pdf.ln(5)

    pdf.set_font('Arial', '', 12)
    pdf.cell(0, 6, '--------------------------------------------------------------------------------------------------------------------', 0, 1, 'C')
    pdf.cell(0, 6, 'International Supermarket', 0, 1, 'C')
    pdf.cell(0, 6, 'Building C, E, HACINCO Student Village, 79 Nguy Nhu Kon Tum, Thanh Xuan, Hanoi', 0, 1, 'C')
    pdf.cell(0, 6, 'Tel: 024.3557.5992 --- Email: truongquocte@vnuis.edu.vn', 0, 1, 'C')
    
    pdf.cell(0, 6, '--------------------------------------------------------------------------------------------------------------------', 0, 1, 'C')
    
    pdf.ln(3)
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, 'PRODUCT RECEIPT', 0, 1, 'C')
    pdf.ln(3)

    pdf.cell(0, 6, '--------------------------------------------------------------------------------------------------------', 0, 1, 'C')
    pdf.set_font('Arial', 'B', 12)
    cols = df.columns
    for col in cols:
        pdf.cell(45, 10, col, 0)
    pdf.ln()
    pdf.cell(0, 6, '--------------------------------------------------------------------------------------------------------', 0, 1, 'C')

    pdf.set_font('Arial', '', 12)
    for _, row in df.iterrows():
        for col in cols:
            pdf.cell(45, 10, str(row[col]), border=0)
        pdf.ln()

    pdf.cell(0, 6, '--------------------------------------------------------------------------------------------------------', 0, 1, 'C')
    pdf.ln(5)

    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, f'Total money for all of the products: {total_money} VND', 0, 1)

    vat = total_money * 0.15

    pdf.cell(0, 10, f'Vat 15%: {int(vat)} VND', 0, 1)
    pdf.cell(0, 10, f'Total money for payment: {total_money + vat} VND', 0, 1)
    
    pdf.cell(0, 6, '______________________________', 0, 1, 'C')
    pdf.cell(0, 6, 'Payment Debit Information:', 0, 1, 'C')
    pdf.cell(0, 6, 'VietcomBank 0011.00.1932418 - Mat Tran To Quoc Viet Nam - Ban Cuu Tro Trung Uong', 0, 1, 'C')
    
    image_width = 45
    x = (pdf.w - image_width) / 2
    y = 245

    pdf.image('/home/duck/Desktop/streamlit/qr.jpg', x=x, y=y, w=image_width)

    return pdf.output(dest='S').encode('latin-1')