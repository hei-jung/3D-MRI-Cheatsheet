import os
from pydicom import dcmread
from tkinter import ttk
from tkinter import *
import matplotlib.pyplot as plt

"""콤보박스에서 이미지 선택하면 matplotlib으로 선택한 이미지 띄워줌"""


class MyFrame(Frame):
    def __init__(self, master):
        Frame.__init__(self, master)

        self.home = './input/'
        self.dir_path = ''
        self.img_path = ''

        self.master = master
        self.master.title("Show DICOM Image")
        self.pack(fill=BOTH, expand=True)

        # select directory
        frame1 = Frame(self)
        frame1.pack(fill=X)

        lblDir = Label(frame1, text="select directory:", width=10)
        lblDir.pack(side=LEFT, padx=10, pady=10)

        dir_list = os.listdir(self.home)
        self.comboDir = ttk.Combobox(frame1, state="readonly", values=dir_list)
        self.comboDir.bind("<<ComboboxSelected>>", self.dir_selected)
        self.comboDir.pack(fill=X, padx=10, expand=True)

        # select DICOM image
        frame2 = Frame(self)
        frame2.pack(fill=X)

        lblImgName = Label(frame2, text="select image:", width=10)
        lblImgName.pack(side=LEFT, padx=10, pady=10)

        self.img_list = []
        self.comboImg = ttk.Combobox(frame2, state="disabled", values=self.img_list)
        self.comboImg.bind("<<ComboboxSelected>>", self.img_selected)
        self.comboImg.pack(fill=X, padx=10, expand=True)

    def dir_selected(self, event):
        self.dir_path = self.home + self.comboDir.get() + '/'
        self.comboImg.set('')
        self.img_list = os.listdir(self.dir_path)
        self.comboImg.config(state="readonly", values=self.img_list)

    def img_selected(self, event):
        self.img_path = self.dir_path + self.comboImg.get()
        # show selected image
        ds = dcmread(self.img_path)
        plt.imshow(ds.pixel_array, cmap=plt.cm.gray)
        plt.show()


def main():
    root = Tk()
    root.geometry("1080x86+100+100")
    app = MyFrame(root)
    root.mainloop()


if __name__ == '__main__':
    main()
