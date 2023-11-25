import matplotlib
matplotlib.use('Agg')
from tkinter import *
from PIL import ImageTk, Image, ImageEnhance
import os
import pandas as pd
import pydicom
import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt


# metadata_csv = "subset_501_1000.csv"
# name = "ethan"
# drive = Path("/Volumes/med-kayvan-lab")

metadata_csv = input("(1/3) Please enter your csv subset: ")
name = input("(2/3) Please enter your uniqname: ")
drive = Path(input("(3/3) Please enter turbo directory: "))


prefix = drive / Path("#####/RSNAChallenge/stage_1_train_images")

output_folder = drive / Path("#####/label_results")
output_labels_path = output_folder / Path("labeled_slices_" + name + ".csv")


restrict_slices = True



current_img = None  # Variable to store the currently loaded image
def load_dicom_image(meta_csv_img_no):
    data = pydicom.dcmread(get_path(meta_csv_img_no))
    img = data.pixel_array
    intercept = meta_csv_df.at[meta_csv_img_no, "Rescale Intercept"]
    slope = meta_csv_df.at[meta_csv_img_no, "Rescale Slope"]
    img = (img * slope + intercept)

    return img

def get_path(no):
    path = prefix / Path(meta_csv_df.at[no, "SOP Instance UID"] + ".dcm")
    print(path)
    return path

def window_image(window_center, window_width, rescale=True, img=None):
    # data = pydicom.dcmread(get_path(meta_csv_img_no))
    # img = data.pixel_array
    # intercept = meta_csv_df.at[meta_csv_img_no, "Rescale Intercept"]
    # slope = meta_csv_df.at[meta_csv_img_no, "Rescale Slope"]
    # img = (img*slope +intercept) #for translation adjustments given in the dicom file.
    
    if img is None:
        data = pydicom.dcmread(get_path(meta_csv_img_no))
        img = data.pixel_array
        intercept = meta_csv_df.at[meta_csv_img_no, "Rescale Intercept"]
        slope = meta_csv_df.at[meta_csv_img_no, "Rescale Slope"]
        img = (img * slope + intercept) 


    if ww==4000 and wc==0:
        img_min = np.amin(img)
        img_max = np.amax(img)
    else:
        img_min = window_center - window_width//2 #minimum HU level
        img_max = window_center + window_width//2 #maximum HU level
        img[img<img_min] = img_min #set img_min for all HU levels less than minimum HU level
        img[img>img_max] = img_max #set img_max for all HU levels higher than maximum HU level

    if tissue.get()=="LazyCanny" or tissue.get()=="LazyHist":
        rescale = False


    if rescale: 
        img = (img - img_min) / (img_max - img_min)*255.0

    if tissue.get()!="LazyHist" and tissue.get()!="LazyCanny":
        clahe = cv2.createCLAHE(clipLimit=pow(10,he.get()/10))
        img = clahe.apply(img.astype(np.uint8))

    if tissue.get()=="LazyHist":
        img = cv2.convertScaleAbs(img)
        img = cv2.equalizeHist(img)

    if tissue.get()=="Canny":
        img = cv2.convertScaleAbs(img)
        # img = cv2.equalizeHist(img)
        img = cv2.Canny(img, threshold1=40, threshold2=150)
    
    if tissue.get()=="LazyCanny":
        img = cv2.convertScaleAbs(img)
        img = cv2.equalizeHist(img)
        img = cv2.Canny(img, threshold1=40, threshold2=150)

    # im = Image.fromarray(img).convert('L').resize(size=(512,512),resample=Image.LANCZOS)
    #.resize(size=(512,512))

    if tissue.get() == "Color" or tissue.get() == "LazyHist":
        default_cmap = plt.colormaps.get_cmap("viridis")
        img = default_cmap(img)
        im = Image.fromarray((img[:, :, :3] * 255).astype(np.uint8))
    else:
        im = Image.fromarray(img.astype(np.uint8))
    
    im = ImageEnhance.Contrast(im).enhance(pow(10,contrast.get()/10))
    im = ImageTk.PhotoImage(im)
    return im

def change_window():
    global ww
    global wc
    global label
    global img_no
    global meta_csv_img_no
    global current_img

    if ( tissue.get() == "Color"):
        ww = 300
        wc = 36
    elif ( tissue.get() == "LazyHist"):
        ww = 300
        wc = 36
    elif ( tissue.get() == "LazyCanny"):
        ww = 300
        wc = 36
    elif ( tissue.get() == "Canny"):
        ww = 300
        wc = 36
    elif ( tissue.get() == "Optic Nerve"):
        ww = 300
        wc = 36
    elif ( tissue.get() == "Soft Tissues"):
        ww = 375
        wc = 40
    elif ( tissue.get() == "All Tissues"):
        ww = 2000
        wc = 0

    
    img = window_image(wc, ww, True, current_img.copy())
    label.config(image=img)
    label.image = img # prevents garbage collection of image for whatever reason

def save(event=None):
    if os.path.isfile(output_labels_path):
        df = pd.read_csv(output_labels_path)
        pd.concat([df, slices]).drop_duplicates(subset='SOP Instance UID', keep="last").reset_index(drop=True).to_csv(output_labels_path, index=False)
        return
    slices.to_csv(output_labels_path, index=False)


def update(dir, toggle_optic=FALSE, toggle_unsure=FALSE):
    global label
    global button_forward
    global button_back
    global button_mark
    global ww
    global wc
    global img_no
    global meta_csv_img_no
    global button_unsure

    img_no += dir
    meta_csv_img_no += dir

    if meta_csv_img_no > len(meta_csv_df.index)-1:
        meta_csv_img_no = len(meta_csv_df.index)-1
        img_no -= dir
    elif meta_csv_img_no < 0:
        meta_csv_img_no = 0
        img_no -= dir

    if (img_no>len(slices.index)-1):
        val = existing_df.loc[existing_df["SOP Instance UID"] == meta_csv_df.at[meta_csv_img_no, "SOP Instance UID"]]
        if (len(val)>0):
            try:
                unsure = val.iloc[0]["Unsure"]
            except:
                unsure = 0
            val = val.iloc[0]["Contains Optic Nerve"]
        else:
            val = 0
            unsure = 0
        row = {
            "SOP Instance UID": meta_csv_df.at[meta_csv_img_no, "SOP Instance UID"],
            "Series Instance UID": meta_csv_df.at[meta_csv_img_no, "Series Instance UID"],
            "Contains Optic Nerve": val,
            "Unsure": unsure
        }
        slices.loc[len(slices.index)] = row 
            
    elif (img_no<0):
        val = existing_df.loc[existing_df["SOP Instance UID"] == meta_csv_df.at[meta_csv_img_no, "SOP Instance UID"]]
        if (len(val)>0):
            val = val.iloc[0]["Contains Optic Nerve"]
        else:
            val = 0
        row = {
            "SOP Instance UID": meta_csv_df.at[meta_csv_img_no, "SOP Instance UID"],
            "Series Instance UID": meta_csv_df.at[meta_csv_img_no, "Series Instance UID"],
            "Contains Optic Nerve": 0,
            "Unsure": 0
        }
        slices.loc[-1] = row
        slices.index = slices.index + 1  # shifting index
        slices.sort_index(inplace=True)  # sorting by index
        img_no = 0

    # if not toggle_optic:
    #     img = window_image(wc, ww)
    #     label.config(image=img)
    #     label.image = img # prevents garbage collection of image for whatever reason
    
    if not toggle_optic:
        global current_img  # Use the global current_img variable

        if current_img is None or dir != 0:
            current_img = load_dicom_image(meta_csv_img_no)

        img = current_img.copy()  # Make a copy of the loaded image


        img = window_image(wc, ww, True, img)  # Apply windowing and level adjustments

        label.config(image=img)
        label.image = img  # prevents garbage collection of image for whatever reason


    if meta_csv_img_no >= len(meta_csv_df.index)-1 or (restrict_slices and meta_csv_img_no==num_slices-1):
        button_forward.config(state=DISABLED)
        root.unbind('<Right>')
    else:
        button_forward.config(state=NORMAL)
        button_forward.config(command=lambda: update(1))
        root.bind('<Right>', lambda event:update(1))
        
    if meta_csv_img_no <= 0 or (restrict_slices and meta_csv_df.at[meta_csv_img_no, "SOP Instance UID"]==starting_slice):
        button_back.config(state=DISABLED)
        root.unbind('<Left>')
    else:
        button_back.config(state=NORMAL)
        button_back.config(command=lambda: update(-1))
        root.bind('<Left>', lambda event:update(-1))
    
    if toggle_optic:
        slices.at[img_no, 'Contains Optic Nerve'] = int(not slices.at[img_no, 'Contains Optic Nerve'])
    
    if toggle_unsure:
        slices.at[img_no, 'Unsure'] = int(not slices.at[img_no, 'Unsure'])

    if (slices.at[img_no, 'Unsure'] == 1):
        unsure_text = "unsure"
        button_unsure.config(text="Mark Label as Confident")
    else:
        unsure_text = "confident"
        button_unsure.config(text="Mark Label as Unsure")

    if slices.at[img_no, 'Contains Optic Nerve']:
        ON_label.config(text="Slice " + str(meta_csv_img_no) + " contains optic nerve (" + unsure_text + ")", fg='dark green')
        button_mark.config(text="Mark Slice as Not Containing Optic Nerve")
    else:
        ON_label.config(text="Slice " + str(meta_csv_img_no) + " doesn't contain optic nerve (" + unsure_text + ")", fg='red')
        button_mark.config(text="Mark Slice as Containing Optic Nerve")
    button_mark.config(command=lambda: update(0,toggle_optic=True))

    root.bind('o', lambda event:update(0,True)) 
 


meta_csv_df = pd.read_csv(metadata_csv)

num_slices = meta_csv_df.shape[0]

starting_slice = meta_csv_df["SOP Instance UID"].iloc[0] #SOP Instance UID
meta_csv_img_no = meta_csv_df.index[meta_csv_df["SOP Instance UID"] == starting_slice].tolist()[0]


img_no = 0


cols = {
    "SOP Instance UID": [],
    "Series Instance UID": [],
    "Contains Optic Nerve": [],
    "Unsure": []
}

val = 0
unsure = 0
if os.path.isfile(output_labels_path):
    existing_df = pd.read_csv(output_labels_path)

    val = existing_df.loc[existing_df["SOP Instance UID"] == meta_csv_df.at[meta_csv_img_no, "SOP Instance UID"]]
    if (len(val)>0):
        try:
            unsure = val.iloc[0]["Unsure"]
        except:
            unsure = 0
        val = val.iloc[0]["Contains Optic Nerve"]
    else:
        val = 0
        unsure = 0
else:
    existing_df = pd.DataFrame(cols)

cols = {
    "SOP Instance UID": [meta_csv_df.at[meta_csv_img_no, "SOP Instance UID"]],
    "Series Instance UID": [meta_csv_df.at[meta_csv_img_no, "Series Instance UID"]],
    "Contains Optic Nerve": [val],
    "Unsure": [unsure]
}
slices = pd.DataFrame(cols)


root = Tk()
root.title("Image Viewer")
root.geometry("570x670")


ww = 300
wc = 36
contrast = DoubleVar()
contrast.set(0)
he = DoubleVar()
he.set(0)


label = Label()
label.grid(row=1, column=0, columnspan=3, sticky= "we")

button_forward = Button(root, text="Next",
                        command=lambda: update(1))
button_back = Button(root, text="Previous", state=DISABLED)
button_mark = Button(root,
                text="Mark Slice as Containing Optic Nerve",
                command=lambda: update(0,toggle_optic=True))
button_unsure = Button(root,
                text="Mark Label as Unsure",
                command=lambda: update(0,toggle_unsure=True))
button_exit = Button(root, 
                    text="Exit Without Saving",
                    command=root.quit)
button_save_exit = Button(root, 
                    text="Save",
                    command=save)

ON_label = Label( text="", relief=RAISED )
ON_label.grid(row=0, column=1)
if not slices.at[0,"Contains Optic Nerve"]:
    ON_label.config(text="Slice " + str(meta_csv_img_no) + " does not contain optic nerve", fg='red')
else:
    ON_label.config(text="Contains optic nerve", fg='dark green')


tissue = StringVar(root)
tissue.set("Optic Nerve") # default value
windows = OptionMenu(root, tissue,
                    "Optic Nerve",
                    "All Tissues",
                    "Soft Tissues",
                    "Color",
                    "Canny",
                    "LazyHist",
                    "LazyCanny",
                    command = lambda event:change_window())
windows.grid(row=7, column=1)
windows_label = Label(root, text = "Window")
windows_label.grid(row=8, column=1)


contrast_scale = Scale( root, variable = contrast, 
           from_ = -5, to = 5, 
           orient = HORIZONTAL, command = lambda event:change_window())
contrast_label = Label(root, text = "Contrast")
contrast_scale.grid(row=6, column=2, sticky= "e")
contrast_label.grid(row=7, column=2, sticky= "e")

hist_scale = Scale( root, variable = he, 
           from_ = 0, to = 20, 
           orient = HORIZONTAL, command = lambda event:change_window())
hist_label = Label(root, text = "Histogram Equalization")
hist_scale.grid(row=6, column=0, sticky= "w")
hist_label.grid(row=7, column=0, sticky= "w")

button_mark.grid(row=5, column=1)
button_unsure.grid(row=6, column=1)
button_back.grid(row=5, column=0, sticky= "w")
button_forward.grid(row=5, column=2, sticky= "e")
button_exit.grid(row=0, column=0, sticky= "w")
button_save_exit.grid(row=0, column=2, sticky= "e")

root.bind('<Control-s>', save) 
root.bind('<Command-s>', save) 
root.bind('o', lambda event:update(0,True)) 


label.grid_columnconfigure(0, weight=1)
label.grid_columnconfigure(2, weight=1)
root.grid_columnconfigure(0, weight=1)
root.grid_columnconfigure(2, weight=1)

update(0,False)

root.mainloop()