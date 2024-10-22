import cv2
import streamlit as st
# from ultralytics import solutions
import pandas as pd
from transformers import pipeline
import torch
import sys
import matplotlib.pyplot as plt
from ultralytics import solutions
from ultralytics.solutions import ObjectCounter
import os
import datetime
from pdfcash import create_download_link, export_as_pdf

price_list = {
    'Bim bim slides': 10000,
    'Oreo': 15000,
    'Teaplus': 12000,
    'Xi dau': 30000,
    'Tuong ot': 12000,
    'Hao hao': 3500,
    'Keo dua hau': 8000,
    'Milo': 11000,
    'Keo bac ha': 6000,
    'Omo': 20000
}

model_path = "/home/duck/Desktop/streamlit/Train_YOLOv9-20241009T111231Z-001/Train_YOLOv9/runs/detect/train/weights/best.pt"

def ai_recommendation(receipt_info, user_question):
    messages = [
    {"role": "system", "content": 
     """
     You are an AI chatbot tasked with answering information related to invoices and customer questions related to the supermarket. 
     Only provide answers when you are certain about the invoice information, otherwise respond to the customer by saying you don’t know. 
     Note that you should consider yourself a professional salesperson at International Supermarket.
     Some information about the supermarket:

     International Supermarket
     Building C, E, HACINCO Student Village, 79 Nguy Nhu Kon Tum, Thanh Xuan, Hanoi
     Tel: 024.3557.5992 
     Email: truongquocte@vnuis.edu.vn
     Website link: https://www.facebook.com/International-Supermarket
     VNUIS market is the abbreviation for the Supermarket of Hanoi International University.

     Products sold at the supermarket:
        - Bim bim slides: 10,000 VND
        - Teaplus: 20,000 VND
        - Oreo: 15,000 VND
        - Xi dau: 30,000 VND
        - Tuong ot: 12,000 VND
        - Hao hao: 35,000 VND
        - Keo dua hau: 8,000 VND
        - Milo: 11,000 VND
        - Keo bac ha: 6,000 VND
        - Omo: 20,000 VND
        - Chan chau duong den: 10,000 VND
        - Matcha: 25,000 VND
        - Keo sua: 5,000 VND
        - Keo deo: 7,000 VND
        - Keo gau: 9,000 VND
        - Soda: 10,000 VND
        - Nuoc ngot: 12,000 VND
        - Nuoc suoi: 5,000 VND
        - Nuoc loc: 3,000 VND
        - Ta tre em: 20,000 VND
     """},

    {"role": "user", "content": f"{receipt_info}\n{user_question}"}
    ]
    
    outputs = st.session_state.ai_chatbot(
        messages,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.7,
        pad_token_id=st.session_state.ai_chatbot.tokenizer.eos_token_id
    )
    
    return outputs[0]["generated_text"][-1]["content"]

def process_video(video_path, save_path="/tmp/output_video.mp4"):
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), "Error reading video file"
    
    w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

    width_cm = 2  
    width_pixels = int(width_cm * 4)  
    center_x = w // 2  
    region_height = h  

    region_points = [
        (center_x - width_pixels // 2, 0),
        (center_x + width_pixels // 2, 0),
        (center_x + width_pixels // 2, region_height),
        (center_x - width_pixels // 2, region_height)
    ]

    # if 'counter' not in st.session_state:
    #     st.session_state.counter = ObjectCounter(
    #             show=False,
    #             region=region_points,
    #             model=model_path
    #         )

    # counter = st.session_state.counter

    counter = ObjectCounter(
            show=False,
            region=region_points,
            model=model_path
        )

    classwise_counts = {}

    out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        
        frame, classwise_counts = counter.count(frame)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        out.write(frame)
        st.session_state.video_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)

    out.release()
    cap.release()
    cv2.destroyAllWindows()  

    total_counts = {key: value['IN'] + value['OUT'] for key, value in classwise_counts.items()}
    total_price = sum([int(price_list[key]) * value for key, value in total_counts.items() if key in price_list])

    return total_counts, total_price, save_path

def show_sales_data(sale_data):
 
    df = pd.DataFrame(sale_data)

    df.set_index('Times', inplace=True)

    df['Total'] = df.sum(axis=1)  

    ax = df.iloc[:, :-1].plot(kind='bar', stacked=True, figsize=(10, 5), width=0.6)

    total_amounts = df['Total'].values
    x_pos = range(len(total_amounts))  
    ax.plot(x_pos, total_amounts, marker='o', color='black', linewidth=2, label='Total Amount')


    plt.title('Sales Data Visualization')
    plt.xlabel('Times')
    plt.ylabel('Amount (VND)')
    plt.xticks(rotation=45)

    ax.set_ylim(0, total_amounts.max() + 10000) 

    plt.legend()

    st.pyplot(plt)

st.title("Object Counting and Billing Demo")
st.session_state.ai_button = True

if "ai_chatbot" not in st.session_state:
    st.session_state.ai_chatbot = pipeline(
    "text-generation",
    model="meta-llama/Llama-3.2-3B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
    
if "sale_data" not in st.session_state:
    st.session_state.sale_data = {
        "Times": [],
        "Bim bim slides": [],
        "Oreo": [],
        "Teaplus": [],
        "Xi dau": [],
        "Tuong ot": [],
        "Hao hao": [],
        "Keo dua hau": [],
        "Milo": [],
        "Keo bac ha": [],
        "Omo": [],
    }


with st.sidebar:
    uploaded_video = st.file_uploader("Upload a Video", type=["mp4", "avi", "mov"])

if uploaded_video is not None:
    video_path = "upload_video.mp4"
    with open(video_path, "wb") as f:
        f.write(uploaded_video.read())

    if "video_placeholder" not in st.session_state:
        st.session_state.video_placeholder = st.empty()

if "df" not in st.session_state:
    st.session_state.df = None
if 'processed' not in st.session_state:
    st.session_state.processed = False

if st.sidebar.button("Process Video"):
    st.session_state.ai_button = True

    if uploaded_video is not None:
        # reset the counter
        if 'counter' in st.session_state:
            st.session_state.counter.reset()

        total_counts, total_price, st.session_state.save_path = process_video(video_path)
        data = []
        st.session_state.sale_data['Times'].append(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

        for product, count in total_counts.items():
            if count > 0:
                unit_price = price_list.get(product, 0)
                product_total_price = unit_price * count
                
                data.append({
                    'Product': product,
                    'VND': unit_price,
                    'Quantity': count,
                    'Total price': product_total_price
                })

        for x in st.session_state.sale_data.keys():
            if x == 'Times':    
                continue
            if x in total_counts.keys():
                st.session_state.sale_data[x].append(total_counts[x] * price_list[x]) 
            else:
                st.session_state.sale_data[x].append(0)

            
        print(st.session_state.sale_data)
        
        df = pd.DataFrame(data)
        
        # st.title("Bảng Đơn Giá và Tổng Tiền")
        st.session_state.df = df
        df_str = df.to_string(index=False)
        grand_total = df['Total price'].sum()

        st.session_state.receipt_info = f"""Price List and Total Amount:

        {df_str}

        Total: {grand_total} VND
        """

        st.session_state.processed = True
        st.session_state.grand_total = grand_total
        print(st.session_state.receipt_info)
    else:
        st.warning("Please upload your video before pressing 'Process Video'.")
        st.session_state.ai_button = False

if 'save_path' not in st.session_state:
    st.session_state.save_path = ""

if "ai_answer" not in st.session_state:
    st.session_state.ai_answer = ""

if st.session_state.save_path != "":
    st.session_state.video_placeholder = st.video(st.session_state.save_path)

if 'grand_total' not in st.session_state:
    st.session_state.grand_total = 0

if st.session_state.df is not None:
    st.table(st.session_state.df)

if st.session_state.ai_button:
    st.sidebar.title("Chat with AI")
    receipt_info = st.session_state.get('receipt_info', "")

    user_question = st.sidebar.text_input("Your question:")
    if st.sidebar.button("Send"):
        if user_question:
            if receipt_info:
                answer = ai_recommendation(receipt_info, user_question)
                st.session_state['ai_answer'] = answer
            else:
                st.sidebar.warning("Invoice information not found.")
        else:
            st.sidebar.warning("Please enter a question before sending.")
    
    if 'ai_answer' in st.session_state:
        st.sidebar.write(f"**Answer:** {st.session_state['ai_answer']}")

if st.sidebar.button("Show Revenue Chart"):
        show_sales_data(st.session_state.sale_data)


if st.sidebar.button("Export as PDF"):
    if st.session_state.df is not None:
        pdf_data = export_as_pdf(st.session_state.df, st.session_state.grand_total)

        st.sidebar.download_button(
            label="Download PDF",
            data=pdf_data,
            file_name="bill_and_total_money.pdf",
            mime="application/pdf"
        )
        st.success("Bill exported successfully!")
    else:
        st.error("DataFrame is empty or not initialized.")
