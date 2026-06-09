import gradio as gr
import skops.io as sio

pipe = sio.load("./Model/ticket_pipeline.skops", trusted=True)

def predict_ticket_type(ticket_description):
    if not ticket_description.strip():
        return "Ju lutem shkruani një përshkrim valid."
    
    
    predicted_type = pipe.predict([ticket_description])[0]
    return f"Kategoria e Parashikuar: {predicted_type}"

# UI input dhe output
inputs = [gr.Textbox(lines=5, label="Përshkruani problemin tuaj (Ticket Description)", placeholder="Shkruaj këtu...")]
outputs = [gr.Label(num_top_classes=3)]

examples = [
    ["My screen turned black and the laptop won't turn on even when charging."],
    ["I was charged twice for my premium subscription this month. I want a refund."],
    ["How can I change my password and update my profile phone number?"]
]

title = "Customer Support Ticket Classifier"
description = "Sistemi automatik i klasifikimit të tiketave mbështetëse duke përdorur Machine Learning dhe CI/CD."

gr.Interface(
    fn=predict_ticket_type,
    inputs=inputs,
    outputs=outputs,
    examples=examples,
    title=title,
    description=description,
    theme=gr.themes.Soft(),
).launch()