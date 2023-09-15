import imghdr
from scipy.interpolate import UnivariateSpline
import colorsys as cs
from numpy import random
import cv2
import matplotlib.pyplot as plt
import tkinter as tk
import tkinter.font as tkFont
import tkinter.messagebox as messagebox
from tkinter import filedialog
from PIL import Image, ImageTk, ImageFilter, ImageFont, ImageDraw
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
# from skimage.io import imread, imshow, imsave

option = 0
class App:
    def __init__(self, main):
        # setting title
        main.title("Image Processing")
        # setting window size
        main['bg'] = '#999999'
        width = 1580
        height = 820
        screenwidth = main.winfo_screenwidth()
        screenheight = main.winfo_screenheight()
        alignstr = '%dx%d+%d+%d' % (width, height,
                                    (screenwidth - width) / 2, (screenheight - height) / 2)
        main.geometry(alignstr)

        def open_image():
            filepath = filedialog.askopenfilename()
            image = cv2.imread(filepath)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            img = Image.open(filepath)
            img = img.resize((655, 400), Image.ANTIALIAS)
            tkimg = ImageTk.PhotoImage(img)

            opreations = tk.Toplevel()
            # setting title
            opreations.title("Image Processing Opreations")
            # setting window size
            opreations['bg'] = '#999999'
            width = 1580
            height = 820
            screenwidth = opreations.winfo_screenwidth()
            screenheight = opreations.winfo_screenheight()
            alignstr = '%dx%d+%d+%d' % (width, height,
                                        (screenwidth - width) / 2, (screenheight - height) / 2)
            opreations.geometry(alignstr)
            frame_res = tk.Frame(opreations, bg='white', borderwidth=5)
            frame_res.place(x=-70, y=0, width=1597, height=85)

            label_res = tk.Label(frame_res)
            ft = tkFont.Font(family='monospace', size=35)
            label_res["font"] = ft
            label_res["bg"] = "black"
            label_res["fg"] = "#ff5722"
            label_res["justify"] = "center"
            label_res["text"] = "Project Image Proccessing"
            label_res.place(x=-40, y=20, width=1579, height=60)

            title = tk.Label(opreations)
            title["text"] = "Welcome to Opreations Interface in the system"
            title["bg"] = "white"
            title["font"] = ('monospace', 14, 'bold')
            title.place(x=-87, y=87, width=1579, height=49)

# ---------------------------- Frame For Buttons Opreations -----------------------------------------

            frame_btns_opreations = tk.Frame(
                opreations, bg='white', borderwidth=5)
            frame_btns_opreations.place(x=0, y=138, width=300, height=819)

            frame_btns_opreations_right = tk.Frame(
                opreations, bg='white', borderwidth=5)
            frame_btns_opreations_right.place(
                x=1070, y=138, width=300, height=819)

            title_frame_opreations = tk.Label(frame_btns_opreations)
            title_frame_opreations['text'] = "Opreations"
            title_frame_opreations['bg'] = '#1e9fff'
            ft = tkFont.Font(family='Times', size=25, weight='bold')
            title_frame_opreations["font"] = ft
            title_frame_opreations.place(x=-2, y=-2, width=300, height=45)

            title_frame_opreations_right = tk.Label(
                frame_btns_opreations_right)
            title_frame_opreations_right['text'] = "Opreations"
            title_frame_opreations_right['bg'] = '#1e9fff'
            ft = tkFont.Font(family='Times', size=25, weight='bold')
            title_frame_opreations_right["font"] = ft
            title_frame_opreations_right.place(
                x=-2, y=-2, width=300, height=45)

# -------------------- Function For Display Image Details -----------------
            def fun_image_details():
                global option
                option = 1
                # Get image details
                width, height = img.size
                img_format = imghdr.what(filepath)

                label = tk.Label(
                    frame_primary_image, text=f"Width: {width}\nHeight: {height}\nFormat: {img_format}")
                label.place(x=680, y=350)
                
                global img_det_rgb
                img_det_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                global det_b, det_g, det_r
                det_b, det_g, det_r = cv2.split(image)
                colors = ['b', 'g', 'r']
                global img_det_gray
                img_det_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                # 1-opencv
                hist = cv2.calcHist([img_det_gray], [0], None, [256], [0, 256])

                # 2-numpy
                # hist = np.histogram(img_gray, 256, (0,255))[0]

                for i in range(len(colors)):
                    hist_color = np.histogram(image[:, :, i], 256, (0, 255))[0]
                    plt.subplot(2, 5, 8+i)
                    plt.plot(range(0, 256), hist_color, c=colors[i])
                    plt.title(f"Hist {colors[i]}")

                plt.subplot(2, 5, 6)
                for i in range(len(colors)):
                    hist_color = np.histogram(image[:, :, i], 256, (0, 255))[0]
                    plt.plot(range(0, 256), hist_color, c=colors[i])
                plt.title(f"Hist original")

                plt.subplot(2, 5, 1)
                plt.imshow(img_det_rgb)
                plt.axis('off')
                plt.title("Original")
                plt.subplot(2, 5, 2)
                plt.imshow(img_det_gray, cmap='gray')
                plt.axis('off')
                plt.title("Gray")
                # 3- plt.hist
                plt.subplot(2, 5, 7)
                # plt.hist(img_gray.flatten(), 256, (0, 255), histtype='step')
                plt.plot(range(0, 256), hist)
                plt.title("Hist Gray")

            #
                plt.subplot(2, 5, 3)
                plt.imshow(det_b, cmap='gray')
                plt.axis('off')
                plt.title("B")
                plt.subplot(2, 5, 4)
                plt.imshow(det_g, cmap='gray')
                plt.axis('off')
                plt.title("G")
                plt.subplot(2, 5, 5)
                plt.imshow(det_r, cmap='gray')
                plt.axis('off')
                plt.title("R")

                plt.show()
                return 0

            btn_image_details = tk.Button(
                frame_btns_opreations, text='Image Details', command=fun_image_details)
            btn_image_details["bg"] = "#e33030"
            ft = tkFont.Font(family='Times', size=17)
            btn_image_details["font"] = ft
            btn_image_details["fg"] = "#ffffff"
            btn_image_details["justify"] = "center"
            btn_image_details.place(x=35, y=60, width=200, height=45)

# -------------------- Function For Image Resizing -------------------
            def fun_image_resizing():
                global option
                option = 2
                dim = (400, 400)
                global resized_image
                resized_image = cv2.resize(
                    image, dim, interpolation=cv2.INTER_AREA)
                plt.imshow(resized_image)

                plt.show()
                return 0

            btn_image_resizing = tk.Button(
                frame_btns_opreations, text='Image Resizing', command=fun_image_resizing)
            btn_image_resizing["bg"] = "#e33030"
            ft = tkFont.Font(family='Times', size=17)
            btn_image_resizing["font"] = ft
            btn_image_resizing["fg"] = "#ffffff"
            btn_image_resizing["justify"] = "center"
            btn_image_resizing.place(x=35, y=110, width=200, height=45)

# -------------------- Function For Image Rotation ---------------------

            def rotate_image():
                global option 
                option = 3
                global rotated_image
                rotated_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
                plt.imshow(rotated_image)
                plt.axis('off')
                plt.show()
                return 0

            # rot_img = rotate_image(img)

            btn_image_rotation = tk.Button(
                frame_btns_opreations, text="Image Rotation", command=rotate_image)
            btn_image_rotation["bg"] = "#e33030"
            ft = tkFont.Font(family='Times', size=17)
            btn_image_rotation["font"] = ft
            btn_image_rotation["fg"] = "#ffffff"
            btn_image_rotation["justify"] = "center"
            btn_image_rotation.place(x=35, y=160, width=200, height=45)


# -------------------- Function For Image Transform ---------------------


            def fun_image_transform():
                img_eq = ImageOps.equalize(img)

                # Convert image to PhotoImage and display
                img_eq = ImageTk.PhotoImage(img_eq)
                label = tk.Label(opreations, image=img_eq)
                label.img_eq = img_eq
                label.pack()
                return 0

            label_image_transform = tk.Label(
                frame_btns_opreations, text="Image Transform")
            label_image_transform["bg"] = '#1e9fff'
            ft = tkFont.Font(family='Times', size=17)
            label_image_transform["font"] = ft
            label_image_transform["fg"] = "black"
            label_image_transform["justify"] = "center"
            label_image_transform.place(x=35, y=210, width=200, height=45)

            # ------------------- Hist equlaization ------------------
            def hist_eq():
                global option 
                option = 4
                # Read the image and convert to grayscale
                img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
                # Calculate the histogram
                hist, bins = np.histogram(img.ravel(), 256, [0, 256])
                # Create the CDF
                cdf = hist.cumsum()
                # Normalize the CDF
                cdf_normalized = cdf * hist.max() / cdf.max()
                # Use the CDF to create the new image
                global img_eq
                img_eq = np.interp(img, bins[:-1], cdf_normalized)
                plt.imshow(img_eq)
                plt.axis('off')
                plt.show()

            btn_histo_equ = tk.Button(
                frame_btns_opreations, text="Histogram equalization", command=hist_eq)
            btn_histo_equ["bg"] = "#e33030"
            ft = tkFont.Font(family='Times', size=12)
            btn_histo_equ["font"] = ft
            btn_histo_equ["fg"] = "#ffffff"
            btn_histo_equ["justify"] = "center"
            btn_histo_equ["text"] = "Histogram equalization"
            btn_histo_equ.place(x=50, y=260, width=160, height=45)

            # ----------------------------- Hist matching -------------------
            def open_image_matching():
                global option
                option = 5
                
                src = cv2.imread(filepath)
                src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
                
                fpath = filedialog.askopenfilename()
                target = cv2.imread(fpath)
                target = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
                
                # Calculate histograms of source and target images
                src_hist = cv2.calcHist([src], [0], None, [256], [0, 256])
                target_hist = cv2.calcHist([target], [0], None, [256], [0, 256])

                # Normalize the histograms
                src_hist = cv2.normalize(src_hist, src_hist).flatten()
                target_hist = cv2.normalize(target_hist, target_hist).flatten()

                # Calculate cumulative histograms
                src_cdf = np.cumsum(src_hist)
                target_cdf = np.cumsum(target_hist)

                # Match the histograms using LUT (Look-Up Table)
                lut = np.zeros(256, dtype=src.dtype)
                for i in range(256):
                    j = 255
                    while j > 0 and src_cdf[j] > target_cdf[i]:
                        j -= 1
                    lut[i] = j

                # Apply the LUT to the source image
                global hist_matching_img
                hist_matching_img = cv2.LUT(src, lut)                
                plt.imshow(hist_matching_img)
                plt.show()

            btn_histo_matching = tk.Button(
                frame_btns_opreations, command=open_image_matching)
            btn_histo_matching["bg"] = "#e33030"
            ft = tkFont.Font(family='Times', size=12)
            btn_histo_matching["font"] = ft
            btn_histo_matching["fg"] = "#ffffff"
            btn_histo_matching["justify"] = "center"
            btn_histo_matching["text"] = "Histogram matching"
            btn_histo_matching.place(x=50, y=310, width=160, height=45)

            # ------------------- Gamma Correction -------------------------
            def gamma_correction():
                global option
                option == 6
                img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
                c = 1
                g = 2
                img = np.float32(img/255.0)
                img_power = c*img**g
                
                global img_power_filter
                img_power_filter = np.uint8(img_power*255)
                plt.title('gamma correction')
                plt.imshow(img_power, cmap='gray')
                plt.axis('off')
                plt.show()
            btn_histo_gamma = tk.Button(
                frame_btns_opreations, text="Gamma Correction", command=gamma_correction)
            btn_histo_gamma["bg"] = "#e33030"
            ft = tkFont.Font(family='Times', size=12)
            btn_histo_gamma["font"] = ft
            btn_histo_gamma["fg"] = "#ffffff"
            btn_histo_gamma["justify"] = "center"
            btn_histo_gamma["text"] = "Gamma correction"
            btn_histo_gamma.place(x=50, y=360, width=160, height=45)

            # ---------------------------------- Log transformation -------------------

            def log_transform():
                global option 
                option = 7
                img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
                M = np.max(img)
                c = 255/np.log10(1+M)
                global img_log
                img_log = np.uint8(c * np.log10(1+img))
                plt.title('Log transform')
                plt.imshow(img_log, cmap='gray')
                plt.axis('off')
                plt.show()
            btn_histo_log = tk.Button(
                frame_btns_opreations, text="Log transform", command=log_transform)
            btn_histo_log["bg"] = "#e33030"
            ft = tkFont.Font(family='Times', size=12)
            btn_histo_log["font"] = ft
            btn_histo_log["fg"] = "#ffffff"
            btn_histo_log["justify"] = "center"
            btn_histo_log["text"] = "Log transformation"
            btn_histo_log.place(x=50, y=410, width=160, height=45)

        # ------------- Function For Image Anylasis ------------------

            def fun_image_anylasis():
                global option
                option = 8
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                # Laplacian filter
                lap = cv2.Laplacian(gray, cv2.CV_64F)
                global laplacine_img, sobel_x_img, sobel_y_img,canny_img
                laplacine_img = np.uint8(np.absolute(lap))
                plt.subplot(2, 2, 1)
                plt.title("Laplacian")
                plt.axis('off')
                plt.imshow(laplacine_img, cmap='gray')

                # Sobel X filter
                sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
                sobel_x_img = np.uint8(np.absolute(sobel_x))
                plt.subplot(2, 2, 2)
                plt.title("Sobel X")
                plt.axis('off')
                plt.imshow(sobel_x_img, cmap='gray')

                # Sobel Y filter
                sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
                sobel_y_img = np.uint8(np.absolute(sobel_y))
                plt.subplot(2, 2, 3)
                plt.title("Sobel Y")
                plt.axis('off')
                plt.imshow(sobel_y_img, cmap='gray')

                # Canny edge detection
                canny_img = cv2.Canny(gray, 50, 150)
                plt.subplot(2, 2, 4)
                plt.title("Canny")
                plt.axis('off')
                plt.imshow(canny_img, cmap='gray')

                plt.show()
                return 0

            btn_image_anylasis = tk.Button(
                frame_btns_opreations, text="Image Anylasis", command=fun_image_anylasis)
            btn_image_anylasis["bg"] = "#e33030"
            ft = tkFont.Font(family='Times', size=17)
            btn_image_anylasis["font"] = ft
            btn_image_anylasis["fg"] = "#ffffff"
            btn_image_anylasis["justify"] = "center"
            btn_image_anylasis.place(x=35, y=460, width=200, height=45)

            # ---------------------------- Function For Image Filtering ----------------------------

            def fun_image_filtering():
                print("command")

            btn_image_filtering = tk.Label(
                frame_btns_opreations_right, text="Image Filtering")
            btn_image_filtering["bg"] = "#1e9fff"
            ft = tkFont.Font(family='Times', size=17)
            btn_image_filtering["font"] = ft
            btn_image_filtering["fg"] = "black"
            btn_image_filtering["justify"] = "center"
            btn_image_filtering.place(x=35, y=60, width=200, height=45)

            # ----------------------- Smoothing ----------------------------
            def image_smooth():
                
                global option, box_smoothed_img, gauess_smoothed_img
                option = 10
                box_smoothed_img = cv2.boxFilter(image, -1, (15, 15))
                gauess_smoothed_img = cv2.GaussianBlur(image, (15, 15), 0)

                plt.subplot(1, 2, 1)
                plt.title("Box Smoothed")
                plt.imshow(box_smoothed_img)

                plt.subplot(1, 2, 2)
                plt.title("Gaussian Smoothed")
                plt.imshow(gauess_smoothed_img)
                plt.show()
                return 0

            btn_image_smooth = tk.Button(
                frame_btns_opreations_right, command=image_smooth)
            btn_image_smooth["bg"] = "#e33030"
            ft = tkFont.Font(family='Times', size=15)
            btn_image_smooth["font"] = ft
            btn_image_smooth["fg"] = "#ffffff"
            btn_image_smooth["justify"] = "center"
            btn_image_smooth["text"] = "Smoothing"
            btn_image_smooth.place(x=50, y=110, width=170, height=45)

            # ----------------------- Sharpining ----------------------------
            def image_sharpining():
                global option, sharpened_img
                option = 11
                kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
                sharpened_img = cv2.filter2D(image, -1, kernel)

                plt.title("Sharpened")
                plt.imshow(sharpened_img)
                plt.show()
                return 0

            btn_image_sharpening = tk.Button(
                frame_btns_opreations_right, command=image_sharpining)
            btn_image_sharpening["bg"] = "#e33030"
            ft = tkFont.Font(family='Times', size=15)
            btn_image_sharpening["font"] = ft
            btn_image_sharpening["fg"] = "#ffffff"
            btn_image_sharpening["justify"] = "center"
            btn_image_sharpening["text"] = "Sharpening"
            btn_image_sharpening.place(x=50, y=160, width=170, height=45)

            btn_image_mapping = tk.Label(frame_btns_opreations_right)
            btn_image_mapping["bg"] = "#1e9fff"
            ft = tkFont.Font(family='Times', size=15)
            btn_image_mapping["font"] = ft
            btn_image_mapping["fg"] = "black"
            btn_image_mapping["justify"] = "center"
            btn_image_mapping["text"] = "Color mapping"
            btn_image_mapping.place(x=50, y=210, width=170, height=45)

            # ----------------------- Cold Filter ----------------------------

            def image_cold_filter():
                global option, cold_img
                option = 12
                cold_img = cv2.applyColorMap(image, cv2.COLORMAP_COOL)
                plt.title("cold")
                plt.imshow(cold_img)
                plt.show()
                return 0

            btn_image_cold_filter = tk.Button(
                frame_btns_opreations_right, command=image_cold_filter)
            btn_image_cold_filter["bg"] = "#e33030"
            ft = tkFont.Font(family='Times', size=15)
            btn_image_cold_filter["font"] = ft
            btn_image_cold_filter["fg"] = "#ffffff"
            btn_image_cold_filter["justify"] = "center"
            btn_image_cold_filter["text"] = "Cold Filter"
            btn_image_cold_filter.place(x=80, y=260, width=120, height=45)

            # ----------------------- Warm Filter ----------------------------
            def image_worm_filter():
                global option, warm_img
                option = 13
                warm_img = cv2.applyColorMap(image, cv2.COLORMAP_HOT)
                plt.title("warm")
                plt.imshow(warm_img)
                plt.show()
                return 0

            btn_image_warm = tk.Button(
                frame_btns_opreations_right, command=image_worm_filter)
            btn_image_warm["bg"] = "#e33030"
            ft = tkFont.Font(family='Times', size=15)
            btn_image_warm["font"] = ft
            btn_image_warm["fg"] = "#ffffff"
            btn_image_warm["justify"] = "center"
            btn_image_warm["text"] = "Warm filter"
            btn_image_warm.place(x=80, y=310, width=120, height=45)

        # --------------------------- Function For Image Conversion ----------------------------------

            def fun_image_conversion():
                global option, img_conv_gray, img_conv_hsv, img_conv_lab
                option = 9
                img_conv_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                # plt.title("Grayscale")
                # # plt.imshow(gray)
                # # plt.show()
                # return 0

                img_conv_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                # plt.title("HSV")
                # # plt.imshow(hsv)
                # # plt.show()
                # return 0

                img_conv_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
                # plt.title("LAB")
                # # plt.imshow(lab)
                # # plt.show()
                # return 0

                plt.subplot(1, 3, 1)
                plt.title("Grayscale")
                plt.imshow(img_conv_gray, cmap='gray')

                plt.subplot(1, 3, 2)
                plt.title("HSV")
                plt.imshow(img_conv_hsv)

                plt.subplot(1, 3, 3)
                plt.title("LAB")
                plt.imshow(img_conv_lab)

                plt.show()
                return 0

            btn_image_conversion = tk.Button(
                frame_btns_opreations, text="Image Conversion", command=fun_image_conversion)
            btn_image_conversion["bg"] = "#e33030"
            ft = tkFont.Font(family='Times', size=17)
            btn_image_conversion["font"] = ft
            btn_image_conversion["fg"] = "#ffffff"
            btn_image_conversion["justify"] = "center"
            btn_image_conversion.place(x=35, y=510, width=200, height=45)

            # ------------------------ Frame For Bottom Opreations -------------------------
            frame_bottom_opreations = tk.Frame(opreations, bg='white')
            frame_bottom_opreations.place(x=301, y=540, width=767, height=400)

            # -------------- Function For Image_Steganography (hide data & extract Data) ---------------------
            btn_image_Steganography = tk.Label(
                frame_bottom_opreations, text="Image Steganography")
            btn_image_Steganography["bg"] = "#1e9fff"
            ft = tkFont.Font(family='Times', size=17)
            btn_image_Steganography["font"] = ft
            btn_image_Steganography["fg"] = "black"
            btn_image_Steganography["justify"] = "center"
            btn_image_Steganography.place(x=450, y=10, width=220, height=45)

            def messageToBinary(message):
                if type(message) == str:
                    return "".join([format(ord(i), "08b") for i in message])
                elif type(message) == bytes or type(message) == np.ndarray:
                    return [format(i, "08b") for i in message]
                elif type(message) == int or type(message) == np.uint8:
                    return format(message, "08b")
                else:
                    raise TypeError("Input type not supported")

            def hideData(image, secret_message):
                # calculate the maximum bytes to encode
                n_bytes = image.shape[0] * image.shape[1] * 3 // 8
                print("Maximum bytes to encode:", n_bytes)

                # Check if the number of bytes to encode is less than the maximum bytes in the image
                if len(secret_message) > n_bytes:
                    raise ValueError(
                        "Error encountered insufficient bytes, need bigger image or less data !!"
                    )

                # you can use any string as the delimeter
                secret_message += "#####"

                data_index = 0
                # convert input data to binary format using messageToBinary() fucntion
                binary_secret_msg = messageToBinary(secret_message)

                # Find the length of data that needs to be hidden
                data_len = len(binary_secret_msg)
                for values in image:
                    for pixel in values:
                        # convert RGB values to binary format
                        r, g, b = messageToBinary(pixel)
                        # modify the least significant bit only if there is still data to store
                        if data_index < data_len:
                            # hide the data into least significant bit of red pixel
                            pixel[0] = int(
                                r[:-1] + binary_secret_msg[data_index], 2)
                            data_index += 1
                        if data_index < data_len:
                            # hide the data into least significant bit of green pixel
                            pixel[1] = int(
                                g[:-1] + binary_secret_msg[data_index], 2)
                            data_index += 1
                        if data_index < data_len:
                            # hide the data into least significant bit of  blue pixel
                            pixel[2] = int(
                                b[:-1] + binary_secret_msg[data_index], 2)
                            data_index += 1
                        # if data is encoded, just break out of the loop
                        if data_index >= data_len:
                            break

                return image

            def fun_hide_data():
                global option, stegano_img_with_data
                option = 14
                image = cv2.imread(filepath)
                data = "Hi everybody! it is steganograpghy test!"
                stegano_img_with_data = hideData(image, data)

                # save
                

            btn_hide_data = tk.Button(
                frame_bottom_opreations, text="Hide Data", command=fun_hide_data)
            btn_hide_data["bg"] = "#e33030"
            ft = tkFont.Font(family='Times', size=15)
            btn_hide_data["font"] = ft
            btn_hide_data["fg"] = "#ffffff"
            btn_hide_data["justify"] = "center"
            btn_hide_data.place(x=480, y=60, width=140, height=40)

            def showData(image):
                binary_data = ""
                for values in image:
                    for pixel in values:
                        # convert the red,green and blue values into binary format
                        b, g, r = messageToBinary(pixel)
                        # extracting data from the least significant bit of red, green, and blue pixel
                        binary_data += b[-1]
                        binary_data += g[-1]
                        binary_data += r[-1]
                # split by 8-bits
                all_bytes = [binary_data[i: i + 8]
                             for i in range(0, len(binary_data), 8)]
                # convert from bits to characters
                decoded_data = ""
                for byte in all_bytes:
                    decoded_data += chr(int(byte, 2))
                    # check if we have reached the delimeter which is "#####"
                    if decoded_data[-5:] == "#####":
                        break
                # remove the delimeter to show the original hidden message
                return decoded_data[:-5]

            def fun_extract_data():
                image = cv2.imread(filepath)
                text = showData(image)
                print(text)

            btn_extract_data = tk.Button(
                frame_bottom_opreations, text="Extract Data", command=fun_extract_data)
            btn_extract_data["bg"] = "#e33030"
            ft = tkFont.Font(family='Times', size=15)
            btn_extract_data["font"] = ft
            btn_extract_data["fg"] = "#ffffff"
            btn_extract_data["justify"] = "center"
            btn_extract_data.place(x=480, y=110, width=140, height=40)

        # -------------- Function For Image Encryption And Decryption ----------------------

            label_image_encyption = tk.Label(
                frame_bottom_opreations, text="Image Encryption")
            label_image_encyption["bg"] = "#1e9fff"
            ft = tkFont.Font(family='Times', size=17)
            label_image_encyption["font"] = ft
            label_image_encyption["fg"] = "black"
            label_image_encyption["justify"] = "center"
            label_image_encyption.place(x=80, y=10, width=220, height=45)

            def fun_image_encryption():
                global option,encrypted_image
                
                option = 15
                img = cv2.imread(filepath)
                r, c, t = img.shape

                key = random.randint(255, size=(r, c, t))
                cv2.imwrite('key.png', key)
                encrypted_image = np.ones((r, c, t), np.uint8)
                for row in range(r):
                    for column in range(c):
                        for depth in range(t):
                            encrypted_image[row, column, depth] = img[row,
                                                                      column, depth] ^ key[row, column, depth]

                # cv2.imwrite('encrypted_image.png', encrypted_image)

                plt.title('encrypted image')
                plt.imshow(encrypted_image)
                plt.show()

            btn_image_encryption = tk.Button(
                frame_bottom_opreations, text="Enryption", command=fun_image_encryption)
            btn_image_encryption["bg"] = "#e33030"
            ft = tkFont.Font(family='Times', size=15)
            btn_image_encryption["font"] = ft
            btn_image_encryption["fg"] = "#ffffff"
            btn_image_encryption["justify"] = "center"
            btn_image_encryption.place(x=110, y=60, width=140, height=40)

            def fun_image_decryption():
                global option, decrypted_image
                option = 16
                encrypted_image = cv2.imread(filepath)
                r, c, t = encrypted_image.shape

                key = cv2.imread(r'key.png')
                decrypted_image = np.zeros((r, c, t), np.uint8)
                for row in range(r):
                    for column in range(c):
                        for depth in range(t):
                            decrypted_image[row, column, depth] = encrypted_image[row,
                                                                                  column, depth] ^ key[row, column, depth]

                # cv2.imwrite('maimage.png', decrypted_image)
                plt.imshow(decrypted_image)
                plt.show()

            btn_image_decryption = tk.Button(
                frame_bottom_opreations, text="Decryption", command=fun_image_decryption)
            btn_image_decryption["bg"] = "#e33030"
            ft = tkFont.Font(family='Times', size=15)
            btn_image_decryption["font"] = ft
            btn_image_decryption["fg"] = "#ffffff"
            btn_image_decryption["justify"] = "center"
            btn_image_decryption.place(x=110, y=110, width=140, height=40)

            # -------------- Function For Image Watermarking ---------------------

            def fun_image_watermarking():
                global option, water_marked_img
                option = 17
                im = Image.open(filepath)
                im = im.convert('RGBA')

                r = 5
                g = 5
                b = 5

                opacity = 40

                font_size = 100

                text = 'abdallah'

                

                watermark = Image.new('RGBA', im.size, (b, g, r, 1))
                draw = ImageDraw.Draw(watermark)
                font = ImageFont.truetype(
                    'arial.ttf', font_size)  # font and font size
                x, y = 100, 100  # position of text
                text_color = (255, 255, 255, opacity)  # color and opacity
                draw.text((x, y), '{}'.format(text),
                          font=font, fill=text_color)
                im = Image.alpha_composite(im, watermark)
                water_marked_img = im.convert('RGB')
                plt.imshow(water_marked_img)
                plt.show()

            btn_image_Watermarking = tk.Button(
                frame_btns_opreations_right, text="Image Watermarking", command=fun_image_watermarking)
            btn_image_Watermarking["bg"] = "#e33030"
            ft = tkFont.Font(family='Times', size=17)
            btn_image_Watermarking["font"] = ft
            btn_image_Watermarking["fg"] = "#ffffff"
            btn_image_Watermarking["justify"] = "center"
            btn_image_Watermarking.place(x=35, y=360, width=220, height=45)

        # ------------------------------ Function For Save Image ---------------------------------

            def fun_save_image():
                if option == 1 :
                    cv2.imwrite('image_details_imgRGB.png', img_det_rgb)
                    cv2.imwrite('image_details_imgGRAY.png', img_det_gray)
                    cv2.imwrite('image_details_img_b_channel.png', det_b)
                    cv2.imwrite('image_details_img_g_channel.png', det_g)
                    cv2.imwrite('image_details_img_r_channel.png', det_r)
                elif option == 2:
                    cv2.imwrite('resized_image.png', resized_image)
                elif option == 3:
                    cv2.imwrite('rotated_image.png', rotated_image)
                elif option == 4:
                    cv2.imwrite('equlaized_img.png', img_eq)
                elif option == 5:
                    cv2.imwrite('hist_matching_img.png', hist_matching_img)
                elif option == 6:
                    cv2.imwrite('gamma_correction_img.png', img_power_filter)
                elif option == 7:
                    cv2.imwrite('Log_transformed_img.png', img_log)
                elif option == 8:
                    cv2.imwrite('Image_analysis_laplacine_img.png', laplacine_img)
                    cv2.imwrite('Image_analysis_sobel_x_img.png', sobel_x_img)
                    cv2.imwrite('Image_analysis_sobel_y_img.png', sobel_y_img)
                    cv2.imwrite('Image_analysis_canny_img.png', canny_img)
                elif option == 9:
                    cv2.imwrite('Image_Conversion_gray_img.png', img_conv_gray)
                    cv2.imwrite('Image_Conversion_hsv_img.png', img_conv_hsv)
                    cv2.imwrite('Image_Conversion_lab_img.png', img_conv_lab)
                elif option == 10:
                    cv2.imwrite('box_smoothed_img.png', box_smoothed_img)
                    cv2.imwrite('gauess_smoothed_img.png', gauess_smoothed_img)
                elif option == 11:
                    cv2.imwrite('sharpned_img.png', sharpened_img)
                elif option == 12:
                    cv2.imwrite('cold_img.png', cold_img)
                elif option == 13:
                    cv2.imwrite('warm_img.png',warm_img)
                elif option == 14:
                    cv2.imwrite('wallpaper-stegano-1.png', stegano_img_with_data, [
                            cv2.IMWRITE_PNG_COMPRESSION, 10])
                elif option == 15:
                    cv2.imwrite('encrypted_img.png', encrypted_image)
                elif option == 16:
                    cv2.imwrite('decrypted_img.png', decrypted_image)
                elif option == 17:
                    cv2.imwrite('watermarked_img.jpg', water_marked_img)
            
            btn_image_save = tk.Button(
                frame_btns_opreations_right, text="Save Image", command=fun_save_image)
            btn_image_save["bg"] = "#5fb878"
            ft = tkFont.Font(family='Times', size=16)
            btn_image_save["font"] = ft
            btn_image_save["fg"] = "black"
            btn_image_save["justify"] = "center"
            btn_image_save.place(x=60, y=410, width=160, height=45)

        # --------------- ------------- Frame For Primary Image ------------------------------------
            frame_primary_image = tk.Frame(opreations, bg='white')
            frame_primary_image.place(x=301, y=138, width=767, height=400)

            title_frame_primary_image = tk.Label(frame_primary_image)
            title_frame_primary_image['text'] = "Primary Image"
            title_frame_primary_image['bg'] = '#1e9fff'
            ft = tkFont.Font(family='Times', size=25, weight='bold')
            title_frame_primary_image["font"] = ft
            title_frame_primary_image.place(x=1, y=3, width=767, height=60)

            label = tk.Label(frame_primary_image, image=tkimg)
            label.img = img
            label.tkimg = tkimg
            label.place(x=20, y=65)


# -------------------------------------------- Main Window -----------------------------------------------------

        frame_res = tk.Frame(main, bg='white', borderwidth=5)
        frame_res.place(x=-70, y=0, width=1597, height=128)
        label_res = tk.Label(frame_res)
        ft = tkFont.Font(family='monospace', size=30)
        label_res["font"] = ft
        label_res["bg"] = "black"
        label_res["fg"] = "#ff5722"
        label_res["justify"] = "center"
        label_res["text"] = "Project Image Proccessing"
        label_res.place(x=-40, y=20, width=1579, height=60)

        title = tk.Label(main)
        title["text"] = "Welcome to Main Interface in the system"
        title["bg"] = "white"
        title["font"] = ('monospace', 14, 'bold')
        title.place(x=-87, y=130, width=1579, height=49)

        frame_content = tk.Frame(main, bg='white', borderwidth=5)
        frame_content.place(x=300, y=181, width=800, height=800)

        btn_upload = tk.Button(
            frame_content, text='Upload Image', command=open_image)
        btn_upload["bg"] = "#e33030"
        ft = tkFont.Font(family='Times', size=19)
        btn_upload["font"] = ft
        btn_upload["fg"] = "#ffffff"
        btn_upload["justify"] = "center"
        btn_upload.place(x=310, y=200, width=185, height=64)

        frame_left = tk.Frame(main, bg='white', borderwidth=5)
        frame_left.place(x=0, y=181, width=300, height=800)
        frame_left['bg'] = '#e33030'

        frame_right = tk.Frame(main, bg='white', borderwidth=5)
        frame_right.place(x=1100, y=181, width=300, height=800)
        frame_right['bg'] = '#e33030'


if __name__ == "__main__":
    main = tk.Tk()
    app = App(main)
    main.mainloop()
