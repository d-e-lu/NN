import tkinter as tk

SCREEN_WIDTH = 640
SCREEN_HEIGHT = 480
PADDING_W1 = 10
PADDING_H1 = 10


class Application(tk.Frame):
    def __init__(self, master=None):
        self.layerSizes = [5,5]
        self.length = 2
        self.neurons = []
        tk.Frame.__init__(self, master)
        self.grid(sticky=tk.N+tk.S+tk.E+tk.W)
        self.create_widgets()
        self.draw_objects()
        self.root = master
        self.reds = 1
        print(self.reds)

    def create_widgets(self):
        top = self.winfo_toplevel()
        top.rowconfigure(0, weight=1)
        top.columnconfigure(0,weight=1)
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)
        self.canvas = tk.Canvas(self,bd=5, relief=tk.GROOVE)
        self.canvas.grid(sticky=tk.N+tk.S+tk.E+tk.W)


        self.quit = tk.Button(self, text='Quit', command=self.quit)
        self.quit.grid(row=1,sticky=tk.W)
        self.addNeuron = tk.Button(self, text='Add', command=self.add_neuron(0))
        self.addNeuron.grid(row=1, sticky=tk.W)

    def add_neuron(self, layer):
        self.layerSizes[layer]+=1
        self.draw_objects()

    def delete_neuron(self, layer):
        if self.layerSizes[layer] > 1:
            self.layerSizes[layer] -= 1
            self.draw_objects()

    def draw_objects(self):
        #self.reds += 1
        self.neurons = []
        self.canvas.delete("all")
        offsetw = 0
        offseth = 0
        for layer in self.layerSizes:
            for neuron in range(layer):
                self.neurons.append(self.canvas.create_oval(20 + offsetw, 20 + offseth, 40 + offsetw, 40+offseth, fill='#90C3D4'))
                offseth += 25
            offseth = 0
            offsetw +=40

        self.canvas.itemconfig(self.neurons[0],fill='red')
        print(self.neurons)





root = tk.Tk()
root.title("NeuNet")
root.geometry('{0}x{1}'.format(str(SCREEN_WIDTH), str(SCREEN_HEIGHT)))
app = Application(master=root)
app.mainloop()
