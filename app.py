import streamlit as st
import json
import ollama 
from ultralytics import YOLO
from PIL import Image
import numpy as np

st.set_page_config(page_title="Smart Retail Assistant", layout="wide")

MODEL_PATH = r'runs\detect\retail_assistant_quick\weights\best.pt'
ONTOLOGY_PATH = r'src\ontology.json'

# Load Model & Ontology
@st.cache_resource
def load_assets():
    model = YOLO(MODEL_PATH)
    with open(ONTOLOGY_PATH, 'r') as f:
        ontology = json.load(f)
    return model, ontology

model, ontology = load_assets()

#AGENT FUNCTION 
def get_semantic_reasoning(detected_items, ontology_data):
    cart_summary = []
    for item in set(detected_items):
        count = detected_items.count(item)
        details = ontology_data.get(item, {})
        cart_summary.append({
            "item": item,
            "quantity": count,
            "category": details.get("category", "Unknown"),
            "promo": details.get("promo", "none")
        })


    prompt = f"""
    You are a Smart Retail AI. Analyze this shopping cart: {json.dumps(cart_summary)}
    
    Tasks:
    1. Summarize the cart in one friendly sentence.
    2. Identify any restricted items (e.g., Alcohol) and provide a legal warning.
    3. Apply semantic logic: If they bought snacks, suggest a beverage. If they have a promo like '2 for $4', confirm if it was applied.
    
    Keep the tone professional and helpful.
    """
    
    try:
        response = ollama.chat(model='llama3.2:3b', messages=[
            {'role': 'user', 'content': prompt},
        ])
        return response['message']['content']
    except Exception as e:
        return f"Semantic Agent error: {str(e)}. (Make sure Ollama is running!)"

st.title("🛍️ Smart AI Retail Checkout")
st.markdown("Upload your cart image to see **Computer Vision** and **Semantic Reasoning** in action.")

uploaded_file = st.file_uploader("Upload cart image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img_array = np.array(image)

    # Upload image to model and get predictions
    results = model.predict(source=img_array, conf=0.5)
    detected_classes = [model.names[int(box.cls[0])] for box in results[0].boxes]

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("👁️ Vision: Object Detection")
        st.image(results[0].plot(), use_container_width=True)

    with col2:
        st.subheader("🧾 Digital Receipt")
        if not detected_classes:
            st.info("No items detected.")
        else:
            total = 0.0
            for item in set(detected_classes):
                count = detected_classes.count(item)
                item_info = ontology.get(item, {"price": 0.0, "promo": "none"})
                line_total = item_info['price'] * count
                total += line_total
                st.write(f"**{item.capitalize()}** x{count} - ${line_total:.2f}")
            
            st.metric("Grand Total", f"${total:.2f}")

            # LLama resoning 
            st.divider()
            st.subheader("🧠 Semantic AI Agent")
            with st.spinner("Agent is reasoning..."):
                reasoning = get_semantic_reasoning(detected_classes, ontology)
                st.info(reasoning)

            if "jinro" in detected_classes:
                st.error("⚠️ AGE VERIFICATION REQUIRED AT COUNTER")