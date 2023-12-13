import tkinter as tk
from tkinter import filedialog
import torch
from PIL import Image, ImageTk, ImageEnhance
from models.archs.BCNet import BCNet
from tkinter import messagebox
import numpy as np
import cv2
import data.util as datautil
import utils.util as util
from torch.nn.parallel import DataParallel, DistributedDataParallel
from models.color_utils.colorspace_transform import rgb2lab,lab2rgb
from models.color_utils.color_transfer import color_transfer

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "7"

class ImageProcessor:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Processor")

        self.netG = BCNet(phase='test')
        self.init_model_button = tk.Button(self.root, text="initial model", command=self.init_model)
        self.init_model_button.grid(row=3, column=3, padx=10, pady=10)

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
        self.clear_reference_button = tk.Button(self.reference_frame, text="Clear Reference", command=self.clear_reference)
        self.clear_reference_button.grid(row=3, column=0, pady=10)
        self.reference_image = None
        self.reference_image, self.reference_path = None,None

        self.output_label = tk.Label(self.output_frame, text="Output Image")
        self.output_label.grid(row=0, column=0)
        self.output_canvas = tk.Canvas(self.output_frame, width=self.canvas_width, height=self.canvas_height)
        self.output_canvas.grid(row=1, column=0)
        self.enhance_button = tk.Button(self.output_frame, text="Enhance", command=self.enhance_image)
        self.enhance_button.grid(row=2, column=0, pady=10)
        self.reset_output_button = tk.Button(self.output_frame, text="Reset Output", command=self.reset_output)
        self.reset_output_button.grid(row=3, column=0, pady=10)
        self.enhance_button_directly_saturation = tk.Button(self.output_frame, text="save", command=self.save_img)
        self.enhance_button_directly_saturation.grid(row=4, column=0, pady=10)
        self.output_image = None

        self.saturation_label = tk.Label(root, text="Saturation:")
        self.saturation_label.grid(row=1, column=0)
        self.saturation_scale = tk.Scale(root, from_=0, to=10, resolution=0.1, orient="horizontal")
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
            return image, file_path

    def load_input_image(self):
        self.input_image, self.input_path = self.load_image(self.input_label, self.input_canvas, self.input_label)

    def load_reference_image(self):
        self.reference_image, self.reference_path = self.load_image(self.reference_label, self.reference_canvas, self.reference_label)

    def clear_reference(self):
        # 清除参考图像
        self.reference_image = None
        self.reference_path = None
        self.reference_canvas.delete("all")
        self.reference_label.config(text="Reference Image")

    def enhance_image(self):
        if self.input_image is not None:
            saturation = self.saturation_scale.get()
            img = self.input_image
            img = np.asarray(img)
            if self.reference_path is not None:
                ref = self.reference_image
                ref = np.asarray(ref)
                transfer = color_transfer(ref, img, clip=True, preserve_paper=True)
                transfer = torch.from_numpy(transfer/255.).float().permute(2, 0, 1).unsqueeze(0).cuda()
                img = torch.from_numpy(img/255.).float().permute(2, 0, 1).unsqueeze(0).cuda()
                img_lab = rgb2lab(img, norm=True)
                img_lab[:, 1:, :, :] = img_lab[:, 1:, :, :] * saturation
                img = lab2rgb(img_lab, norm=True)
                transfer_lab = rgb2lab(transfer, norm=True)
                transfer = transfer_lab[:, 1:, :, :] * saturation
                with torch.no_grad():
                    out = self.netG(img, img,transfer)
            else:
                img = torch.from_numpy(img/255.).float().permute(2, 0, 1).unsqueeze(0).cuda()
                img_lab = rgb2lab(img,norm=True)
                img_lab[:,1:,:,:] = img_lab[:,1:,:,:]*saturation
                img = lab2rgb(img_lab,norm=True)
                with torch.no_grad():
                    out = self.netG(img,img)
            out_img = torch.clamp(out[0],0,1).detach().cpu().squeeze().permute(1,2,0).numpy()
            out_img = (out_img * 255).astype(np.uint8)
            # cv2.imshow('0',out_img)
            # cv2.waitKey(0)
            # out_img = out_img[:, :, ::-1]
            self.output_image = Image.fromarray(out_img)


            # enhancer = ImageEnhance.Color(self.input_image)
            # self.output_image = enhancer.enhance(saturation)
            self.display_image(self.output_image, self.output_canvas)

    def save_img(self):
        self.output_image.save('./saved.png')

    def reset_output(self):
        # 重置输出图像为输入图像
        self.output_image = self.input_image
        self.display_image(self.output_image, self.output_canvas)

    def display_image(self, image, canvas):
        if image:
            image.thumbnail((self.canvas_width, self.canvas_height))
            photo = ImageTk.PhotoImage(image)
            canvas.create_image(0, 0, anchor=tk.NW, image=photo)
            canvas.image = photo

    def init_model(self):
        file_path = filedialog.askopenfilename(filetypes=[("pth files", "*.pth")])
        model_path = file_path
        self.netG.load_state_dict(torch.load(model_path))
        self.netG.eval()
        self.netG = self.netG.cuda()
        # self.netG = DataParallel(self.netG)
        messagebox.showinfo("info", "initial the model")


def main():
    root = tk.Tk()
    processor = ImageProcessor(root)
    root.geometry("1500x600")
    root.mainloop()

if __name__ == '__main__':
    main()
