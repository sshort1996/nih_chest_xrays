import tkinter as tk
from PIL import Image, ImageDraw
import io


window = tk.Tk()
canvas = tk.Canvas(window, width=280, height=280, bg='white')
canvas.pack()

def on_mouse_drag(event):
    x = event.x
    y = event.y
    radius = 5
    canvas.create_oval(x - radius, y - radius, x + radius, y + radius, fill='black')

canvas.bind('<B1-Motion>', on_mouse_drag)
def clear_canvas():
    canvas.delete('all')

def run_model_prediction():
    image = canvas.postscript(colormode='gray')
    img = Image.open(io.BytesIO(image.encode('utf-8')))
    img = img.resize((28, 28))
    # Process the image and feed it into your model for prediction
    # prediction = model.predict(np.expand_dims(img, axis=0))
    # predicted_label = np.argmax(prediction)

    # print("Prediction:", predicted_label)


clear_button = tk.Button(window, text='Clear', command=clear_canvas)
clear_button.pack()

predict_button = tk.Button(window, text='Predict', command=run_model_prediction)
predict_button.pack()

window.mainloop()