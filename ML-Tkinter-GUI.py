#########################################################################################
#																						#
# 								MACHINE LEARNING - TKINTER		                        #								#
#		          						JUAN GUARINO			        		   	    #
##########################################################################################

# Import modules
import tkinter.font
from tkinter.messagebox import showinfo
from tkinter import *
from tkinter import ttk
# Import utility package
import Utility

# Create window
top = Tk()
# Set app title
top.title("Machine Learning App")

# Resize window disable
top.resizable(False, False)

# Set dimensions
top.geometry("1024x800")

# Import methods
methods = list(Utility.methods_dict.keys())

# Import datasets
datasets_list = list(Utility.datasets_dict.keys())

# Create canvas 1 inside windows
canvas1 = Canvas(
    top,
    bg="#73728A",
    height=800,
    width=512,
    bd=0,
    highlightthickness=0,
    relief="ridge"
)
canvas1.place(x=0, y=0)

# Visible title
canvas1.create_text(
    18.0,
    12.0,
    anchor="nw",
    text="Machine Learning App",
    fill="#FFFFFF",
    font=("Sans Serif", 36 * -1)
)

# Add text
canvas1.create_text(
    18.0,
    83.0,
    anchor="nw",
    text="Select your method",
    fill="#FFFFFF",
    font=("Rosario Regular", 24 * -1)
)

# Create method variable
varMethod = tkinter.StringVar()

# Create dropdown menu
s_method = ttk.Combobox(canvas1, textvariable= varMethod, values = methods, width=60, state='readonly')
s_method.place(x=20, y=130)

# Add text
canvas1.create_text(
    18.0,
    190,
    anchor="nw",
    text="Select dataset",
    fill="#FFFFFF",
    font=("Rosario Regular", 24 * -1)
)

# Create dataset variable
varDataset = tkinter.StringVar()

# Create dropdown menu
s_data = ttk.Combobox(canvas1, textvariable=varDataset, values = datasets_list, width=60, state='readonly')
s_data.place(x=20, y=237)

# Add text
canvas1.create_text(
    18.0,
    297.0,
    anchor="nw",
    text="Enter K (Cross validation)",
    fill="#FFFFFF",
    font=("Rosario Regular", 24 * -1)
)

# Add text
canvas1.create_text(
    18.0,
    325,
    anchor="nw",
    text="It must be an integer. The maximum value is 15",
    fill="#FFFFFF",
    font=("Rosario Regular", 12 * -1)
)

# Create K fold variable
vark = tkinter.StringVar()

# Set list of K values
k_values = list(range(2, 16))

# Create dropdown menu
input_k = ttk.Combobox(canvas1, textvariable= vark, values = k_values, width=60, state='readonly')
input_k.set(2)
input_k.place(x=20, y=344)

# Function to print outputs on the window
def print_Grid():
    # Create Canvas 2 for plots
    canvas2 = Canvas(
        top,
        bg="#73728A",
        height=800,
        width=512,
        bd=0,
        highlightthickness=0,
        relief="ridge"
    )

    # Place canvas
    canvas2.place(x=512, y=0)

    # Get values for modeling
    dataset = varDataset.get()
    method = varMethod.get()
    k = int(vark.get())

    if dataset !='' and method !='' and k != '':
        # Call modeling function
        output = Utility.modeling(method, dataset, canvas2, k)

        # Create Text box for best parameters' output
        best_params = Text(canvas1)
        best_params.place(x=48,
                          y=628,
                          width=406,
                          height=141
                          )
        best_params.insert(END, output)

    else:
        # Message pop-up
        msg = f'Select a value!'
        showinfo(title='Select a correct value', message=msg)

# Create run button
run_button = Button(
    canvas1,
    text='Run',
    fg="black",
    bg="#C4C4C4",
    command= print_Grid,
    relief="flat",
    font=("Rosario Regular", 24 * -1)
)

# Place button
run_button.place(
    x=115.0,
    y=510.0,
    width=272.0,
    height=77.0
)

top.mainloop()

