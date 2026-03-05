import argparse
import json
import shutil
from pathlib import Path
from paddleocr import PaddleOCR  

OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

ocr = PaddleOCR(
    device="gpu",
    use_doc_orientation_classify=False, # Disables document orientation classification model via this parameter
    use_doc_unwarping=False, # Disables text image rectification model via this parameter
    use_textline_orientation=False, # Disables text line orientation classification model via this parameter
)
# ocr = PaddleOCR(lang="en") # Uses English model by specifying language parameter
# ocr = PaddleOCR(ocr_version="PP-OCRv4") # Uses other PP-OCR versions via version parameter
# ocr = PaddleOCR(device="gpu") # Enables GPU acceleration for model inference via device parameter
# ocr = PaddleOCR(
#     text_detection_model_name="PP-OCRv5_mobile_det",
#     text_recognition_model_name="PP-OCRv5_mobile_rec",
#     use_doc_orientation_classify=False,
#     use_doc_unwarping=False,
#     use_textline_orientation=False,
# ) # Switch to PP-OCRv5_mobile models

result = ocr.predict("C:/Users/druiv/shots/frame_067.00s_002010.jpg")  
output_dir = str(OUTPUT_DIR)
for res in result:  
    res.print()  
    res.save_to_img(output_dir)  
    res.save_to_json(output_dir)
