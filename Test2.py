# ==================================================
# MEDICAL DISEASE DETECTION – TESTING GUI
# CARDIOVASCULAR & DIABETES 
# ==================================================

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
from tensorflow.keras.models import load_model
import time
import os

# ==================================================
# GLOBAL VARIABLES
# ==================================================
input_data = {"cardio": None, "diabetes": None}
models = {"cardio": None, "diabetes": None}

BASE_DIR = os.getcwd()
CARDIO_DATA_DIR = os.path.join(BASE_DIR, "cardio_vascular")
DIABETES_DATA_DIR = os.path.join(BASE_DIR, "Diabetes")

# ==================================================
# DATA FORMATTER (ROW ALIGNED)
# ==================================================
def format_dataframe(df, max_rows=4):
    df = df.head(max_rows)

    col_widths = {
        col: max(len(col), df[col].astype(str).map(len).max())
        for col in df.columns
    }

    header = "  ".join(f"{col:<{col_widths[col]}}" for col in df.columns)
    lines = [header]

    for _, row in df.iterrows():
        line = "  ".join(f"{str(row[col]):<{col_widths[col]}}" for col in df.columns)
        lines.append(line)

    return "\n".join(lines)

# ==================================================
# FLOW DIAGRAM FUNCTIONS
# ==================================================
def draw_flow_cardio(canvas):
    canvas.delete("all")

    canvas.create_rectangle(30, 30, 230, 180, fill="#D6EAF8", width=2)
    canvas.create_text(130, 45, text="Cardio Input Data (CSV)",
                       font=("Arial", 10, "bold"))

    data_text = canvas.create_text(
        130, 65,
        text="",
        anchor="n",
        font=("Courier", 8),
        width=170
    )

    canvas.create_rectangle(280, 30, 480, 180, fill="#D5F5E3", width=2)
    canvas.create_text(380, 85, text="Cardio AI-LSTM\nModel (.h5)",
                       font=("Arial", 11, "bold"))

    canvas.create_rectangle(530, 30, 730, 180, fill="#FADBD8", width=2)
    canvas.create_text(630, 80,
                       text="Cardio Vascular\nPrediction",
                       font=("Arial", 11, "bold"))

    output_text = canvas.create_text(
        630, 150,
        text="",
        font=("Arial", 10, "bold"),
        fill="blue"
    )

    canvas.create_line(230, 105, 280, 105, arrow=tk.LAST, width=2)
    canvas.create_line(480, 105, 530, 105, arrow=tk.LAST, width=2)

    return data_text, output_text


def draw_flow_diabetes(canvas):
    canvas.delete("all")

    canvas.create_rectangle(30, 30, 230, 180, fill="#E8DAEF", width=2)
    canvas.create_text(130, 45, text="Diabetes Input Data (CSV)",
                       font=("Arial", 10, "bold"))

    data_text = canvas.create_text(
        130, 45,
        text="",
        anchor="n",
        font=("Courier", 8),
        width=170
    )

    canvas.create_rectangle(280, 30, 480, 180, fill="#D5F5E3", width=2)
    canvas.create_text(380, 85, text="Diabetes AI-DNN\nModel (.h5)",
                       font=("Arial", 11, "bold"))

    canvas.create_rectangle(530, 30, 730, 180, fill="#FADBD8", width=2)
    canvas.create_text(630, 80,
                       text="Diabetes Prediction",
                       font=("Arial", 11, "bold"))

    output_text = canvas.create_text(
        630, 150,
        text="",
        font=("Arial", 10, "bold"),
        fill="purple"
    )

    canvas.create_line(230, 105, 280, 105, arrow=tk.LAST, width=2)
    canvas.create_line(480, 105, 530, 105, arrow=tk.LAST, width=2)

    return data_text, output_text

# ==================================================
# ANIMATION
# ==================================================
def animate_flow(canvas, disease_type):
    color = "blue" if disease_type == "cardio" else "purple"
    dot = canvas.create_oval(235, 100, 245, 110, fill=color, outline="")

    for _ in range(8):
        canvas.move(dot, 5, 0)
        root.update()
        time.sleep(0.02)

    canvas.itemconfigure(dot, state="hidden")
    for _ in range(20):
        canvas.move(dot, 5, 0)
        root.update()
        time.sleep(0.01)

    canvas.itemconfigure(dot, state="normal")
    for _ in range(28):
        canvas.move(dot, 5, 0)
        root.update()
        time.sleep(0.02)

    canvas.delete(dot)

# ==================================================
# CORE FUNCTIONS
# ==================================================
def upload_input_data(disease_type, canvas, data_text_id):
    allowed_dir = CARDIO_DATA_DIR if disease_type == "cardio" else DIABETES_DATA_DIR

    file_path = filedialog.askopenfilename(
        initialdir=allowed_dir,
        filetypes=[("CSV Files", "*.csv")]
    )
    if not file_path:
        return

    df = pd.read_csv(file_path)
    input_data[disease_type] = df

    preview = format_dataframe(df)
    canvas.itemconfigure(data_text_id, text=preview)


def upload_model(disease_type):
    file_path = filedialog.askopenfilename(filetypes=[("Keras Model", "*.h5")])
    if file_path:
        models[disease_type] = load_model(file_path)
        messagebox.showinfo("Success", "Model loaded successfully")


def analyze_data(disease_type, canvas, output_text_id):
    if input_data[disease_type] is None or models[disease_type] is None:
        messagebox.showerror("Error", "Load input data and model first")
        return

    animate_flow(canvas, disease_type)

    X = input_data[disease_type].values
    preds = models[disease_type].predict(X)

    result = "Disease Detected" if preds.mean() > 0.5 else "No Disease Detected"
    canvas.itemconfigure(output_text_id, text=f"Output: {result}")

# ==================================================
# GUI DESIGN
# ==================================================
root = tk.Tk()
root.title("Disease Detection – Testing Module")
root.geometry("900x620")
root.resizable(False, False)

notebook = ttk.Notebook(root)
notebook.pack(expand=True, fill="both", padx=10, pady=10)

# ================= CARDIO TAB =====================
cardio_tab = ttk.Frame(notebook)
notebook.add(cardio_tab, text="Cardiovascular Disease")

tk.Label(cardio_tab, text="Cardiovascular Disease Testing",
         font=("Arial", 16, "bold")).pack(pady=5)

cardio_canvas = tk.Canvas(cardio_tab, width=760, height=230, bg="white")
cardio_canvas.pack(pady=10)
cardio_data_text, cardio_output_text = draw_flow_cardio(cardio_canvas)

tk.Button(cardio_tab, text="Upload Cardio Input Data",
          width=40,
          command=lambda: upload_input_data("cardio", cardio_canvas, cardio_data_text)
          ).pack(pady=5)

tk.Button(cardio_tab, text="Upload Cardio Trained Model (.h5)",
          width=40,
          command=lambda: upload_model("cardio")
          ).pack(pady=5)

tk.Button(cardio_tab, text="Analyze & Predict",
          width=40, bg="#5DADE2",
          command=lambda: analyze_data("cardio", cardio_canvas, cardio_output_text)
          ).pack(pady=10)

# ================= DIABETES TAB ===================
diabetes_tab = ttk.Frame(notebook)
notebook.add(diabetes_tab, text="Diabetes Disease")

tk.Label(diabetes_tab, text="Diabetes Disease Testing",
         font=("Arial", 16, "bold")).pack(pady=5)

diabetes_canvas = tk.Canvas(diabetes_tab, width=760, height=230, bg="white")
diabetes_canvas.pack(pady=10)
diabetes_data_text, diabetes_output_text = draw_flow_diabetes(diabetes_canvas)

tk.Button(diabetes_tab, text="Upload Diabetes Input Data",
          width=40,
          command=lambda: upload_input_data("diabetes", diabetes_canvas, diabetes_data_text)
          ).pack(pady=5)

tk.Button(diabetes_tab, text="Upload Diabetes Trained Model (.h5)",
          width=40,
          command=lambda: upload_model("diabetes")
          ).pack(pady=5)

tk.Button(diabetes_tab, text="Analyze & Predict",
          width=40, bg="#5DADE2",
          command=lambda: analyze_data("diabetes", diabetes_canvas, diabetes_output_text)
          ).pack(pady=10)

# ==================================================
# RUN
# ==================================================
root.mainloop()
