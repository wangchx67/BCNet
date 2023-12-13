import tkinter as tk
from tkinter import filedialog

import torch
from PIL import Image, ImageTk, ImageEnhance
from models.archs.BCNet import BCNet

class ImageProcessor:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Processor")
        self.netG = BCNet()
        model_path = "experiments/BCNet_magAttn_LSRW_Huawei_nc16_ps512_chrominance_colorAug/models/250000.pth"
        self.netG.load_state_dict(torch.load(model_path))
        self.netG.cuda()

        self.canvas_width = 500
        self.canvas_height = 500

        self.input_frame = tk.Frame(root)
        self.input_frame.grid(row=0, column=0, padx=10, pady=10)

        self.reference_frame = tk.Frame(root)
        self.reference_frame.grid(row=0, column=1, padx=10, pady=10)

        self.output_frame = tk.Frame(root)
        self.output_frame.grid(row=0, column=2, padx=10, pady=10)

        self.input_label = tk.Label(self.input_frame, text="Input Image")
        self.input_label.grid(row=0, column=0)
        self.input_canvas = tk.Canvas(self.input_frame, width=self.canvas_width, height=self.canvas_height)
        self.input_canvas.grid(row=1, column=0)
        self.load_input_button = tk.Button(self.input_frame, text="Load Input Image", command=self.load_input_image)
        self.load_input_button.grid(row=2, column=0, pady=10)
        self.input_image = None

        self.reference_label = tk.Label(self.reference_frame, text="Reference Image")
        self.reference_label.grid(row=0, column=0)
        self.reference_canvas = tk.Canvas(self.reference_frame, width=self.canvas_width, height=self.canvas_height)
        self.reference_canvas.grid(row=1, column=0)
        self.load_reference_button = tk.Button(self.reference_frame, text="Load Reference Image", command=self.load_reference_image)
        self.load_reference_button.grid(row=2, column=0, pady=10)
        self.reference_image = None

        self.output_label = tk.Label(self.output_frame, text="Output Image")
        self.output_label.grid(row=0, column=0)
        self.output_canvas = tk.Canvas(self.output_frame, width=self.canvas_width, height=self.canvas_height)
        self.output_canvas.grid(row=1, column=0)
        self.enhance_button = tk.Button(self.output_frame, text="Enhance", command=self.enhance_image)
        self.enhance_button.grid(row=2, column=0, pady=10)
        self.output_image = None

        self.saturation_label = tk.Label(root, text="Saturation:")
        self.saturation_label.grid(row=1, column=0)
        self.saturation_scale = tk.Scale(root, from_=0, to=2, resolution=0.1, orient="horizontal")
        self.saturation_scale.set(1.0)
        self.saturation_scale.grid(row=1, column=1)

    def load_image(self, label, canvas, label_text):
        # 删除旧图像
        canvas.delete("all")

        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp")])
        if file_path:
            image = Image.open(file_path)
            label_text.config(text=label_text.cget("text"))
            image.thumbnail((self.canvas_width, self.canvas_height))
            photo = ImageTk.PhotoImage(image)
            canvas.create_image(0, 0, anchor=tk.NW, image=photo)
            canvas.image = photo
            return image

    def load_input_image(self):
        self.input_image = self.load_image(self.input_label, self.input_canvas, self.input_label)

    def load_reference_image(self):
        self.reference_image = self.load_image(self.reference_label, self.reference_canvas, self.reference_label)

    def enhance_image(self):
        if self.input_image is not None:
            img = self.input_image

            saturation = self.saturation_scale.get()
            enhancer = ImageEnhance.Color(self.input_image)
            self.output_image = enhancer.enhance(saturation)
            self.display_image(self.output_image, self.output_canvas)

    def display_image(self, image, canvas):
        if image:
            image.thumbnail((self.canvas_width, self.canvas_height))
            photo = ImageTk.PhotoImage(image)
            canvas.create_image(0, 0, anchor=tk.NW, image=photo)
            canvas.image = photo

def main():
    root = tk.Tk()
    processor = ImageProcessor(root)
    root.geometry("1500x600")  # 设置窗口初始大小
    root.mainloop()

if __name__ == '__main__':
    main()
