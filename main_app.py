import uuid
import openai
from dotenv import load_dotenv
from PIL import Image
from io import BytesIO
import pypdfium2 as pdfium
import streamlit as st
import multiprocessing
import pandas as pd
import json
import os
import io
import base64
import cv2
import numpy as np

load_dotenv()

st.set_page_config(page_title="Land Titles Data Extraction", page_icon="📜", layout="wide")

openai.api_key = os.getenv("OPENAI_API_KEY")

# Convert PDF file into images via pypdfium2
def convert_pdf_to_images(file_path, scale=500/72):
    pdf_file = pdfium.PdfDocument(file_path)
    page_indices = [i for i in range(len(pdf_file))]
    renderer = pdf_file.render(
        pdfium.PdfBitmap.to_pil,
        page_indices=page_indices,
        scale=scale,
    )
    final_images = []
    for i, image in zip(page_indices, renderer):
        image_byte_array = BytesIO()
        image.save(image_byte_array, format='jpeg', optimize=True)
        image_byte_array = image_byte_array.getvalue()
        final_images.append(dict({i: image_byte_array}))
    if hasattr(pdf_file, "close"):
        pdf_file.close()
    return final_images

# Preprocess image
def preprocess_image(image):
    open_cv_image = np.array(image)
    open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)
    img = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cntsSorted = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
        cnt = cntsSorted[0]
        x, y, w, h = cv2.boundingRect(cnt)
        img = img[y:y+h, x:x+w]
    _, result = cv2.threshold(img, 210, 235, cv2.THRESH_BINARY)
    adaptive_result = cv2.adaptiveThreshold(
        result, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 41, 5
    )
    processed_image = Image.fromarray(adaptive_result)
    return processed_image

# Encode image to base64
def encode_image(image_bytes):
    return base64.b64encode(image_bytes).decode('utf-8')

# Extract raw text from images via OpenAI Vision
def extract_raw_text_from_img_openai(list_dict_final_images):
    raw_texts = []
    for data in list_dict_final_images:
        image_bytes = list(data.values())[0]
        image = Image.open(io.BytesIO(image_bytes))
        processed_image = preprocess_image(image)

        buffered = BytesIO()
        processed_image.save(buffered, format="JPEG")
        encoded_image = encode_image(buffered.getvalue())
        
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": """
                        Please extract all the text from this image. Just output the extracted text from the document and no need for conversational messages.
                        """.strip()},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}}
                    ]
                }
            ],
            max_tokens=2048
        )
        
        raw_text = response['choices'][0]['message']['content']
        raw_texts.append((encoded_image, raw_text))

    return raw_texts

# Extract structured info from text via LLM
def extract_structured_data(content: str):
    template = """
    You are an expert at extracting structured information from unstructured text. Please extract the following information:

    - Transfer Certificate of Title Number
    - Landowner
    - Location
    - Land Description
    - Land Area

    If any information is not available, please return "Not available".

    Here is the content:

    {content}

    Please return the extracted information in the following format:

    Transfer Certificate of Title Number: 
    Landowner:
    Location:
    Land Description: 
    Land Area:
    """
    
    response = openai.ChatCompletion.create(
        model="gpt-4-turbo",
        messages=[
            {
                "role": "user",
                "content": template.format(content=content)
            }
        ],
        max_tokens=500
    )
    
    return response.choices[0].message['content'].strip()

# Streamlit app
def main():
    st.header("📜 Land Titles Data Extraction")

    uploaded_files = st.file_uploader("Upload file(s):", accept_multiple_files=True, type=["pdf", "jpg", "jpeg"])

    if uploaded_files:
        results = []
        for file in uploaded_files:
            try:
                if file.name.endswith('.pdf'):
                    temp_filename = f"temp_{uuid.uuid4()}.pdf"
                    with open(temp_filename, 'wb') as f:
                        f.write(file.getbuffer())
                    images_list = convert_pdf_to_images(temp_filename)
                    os.remove(temp_filename)
                elif file.name.lower().endswith(('.jpg', '.jpeg')):
                    image_bytes = file.getbuffer()
                    images_list = [{0: image_bytes}]
                else:
                    raise ValueError("Unsupported file format")

                raw_texts = extract_raw_text_from_img_openai(images_list)
                for idx, (encoded_image, raw_text) in enumerate(raw_texts):
                    cols = st.columns(3)
                    with cols[0]:
                        st.image(f"data:image/jpeg;base64,{encoded_image}", caption=f'Processed Image - {file.name}', use_column_width=True)

                    with cols[1]:
                        st.subheader("Raw Extracted Information")
                        st.text_area("Raw Text", value=raw_text, height=200, key=f'raw_text_{file.name}_{idx}')

                    with cols[2]:
                        st.subheader("Structured Information")
                        json_key = f'{file.name}_{idx}_json'
                        initial_value = """
                            Transfer Certificate of Title Number: 
                            Landowner:
                            Location:
                            Land Description: 
                            Land Area:
                        """

                        if json_key not in st.session_state:
                            extracted_json = extract_structured_data(raw_text)
                            st.session_state[json_key] = extracted_json

                        json_input = st.text_area("Edit the extracted information: ", value=st.session_state.get(json_key, initial_value), key=json_key, height=300)

            except Exception as e:
                st.error(f"Error processing file {file.name}: {e}")

        if st.button("Extract to Excel"):
            for key in st.session_state:
                if key.endswith("_json"):
                    extracted_data = st.session_state[key].strip().split('\n')
                    data_dict = {
                        "Transfer Certificate of Title Number": extracted_data[0].split(":")[1].strip() if len(extracted_data) > 0 else "Not available",
                        "Landowner": extracted_data[1].split(":")[1].strip() if len(extracted_data) > 1 else "Not available",
                        "Location": extracted_data[2].split(":")[1].strip() if len(extracted_data) > 2 else "Not available",
                        "Land Description": extracted_data[3].split(":")[1].strip() if len(extracted_data) > 3 else "Not available",
                        "Land Area": extracted_data[4].split(":")[1].strip() if len(extracted_data) > 4 else "Not available",
                        "File Name": key.rsplit('_', 2)[0],
                    }
                    results.append(data_dict)

            if results:
                df = pd.DataFrame(results)
                st.subheader("Results")
                st.dataframe(df)

                output = BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    df.to_excel(writer, index=False, columns=["Transfer Certificate of Title Number", "Landowner", "Location", "Land Description", "Land Area"])
                output.seek(0)

                st.download_button(
                    label="Download extracted data",
                    data=output,
                    file_name='extracted_data.xlsx',
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
