import tkinter as tk
import Hamming_Net
import correctLetters
import randomDataset
import statistics


class HammingNetGUI:
    def __init__(self, root, hnn):
        self.root = root
        self.hamming_net = hnn  # Store hamming_net as an attribute
        self.root.title("Hamming Net GUI")

        self.grid_size = 8
        self.cell_size = 50
        self.grid = []
        self.binary_vector = [0] * 64

        self.create_grid()
        self.create_submit_button()
        self.create_statistics_button()
        self.create_train_button()
        self.create_train_button_10_times()

    def create_grid(self):
        self.canvas = tk.Canvas(self.root, width=self.grid_size * self.cell_size,
                                height=self.grid_size * self.cell_size)
        self.canvas.pack()

        for row in range(self.grid_size):
            row_squares = []
            for col in range(self.grid_size):
                x0, y0 = col * self.cell_size, row * self.cell_size
                x1, y1 = x0 + self.cell_size, y0 + self.cell_size
                square = self.canvas.create_rectangle(
                    x0, y0, x1, y1, fill="white", outline="black"
                )

                self.canvas.tag_bind(square, "<Button-1>", lambda e, r=row, c=col: self.on_click(r, c))
                row_squares.append(square)
            self.grid.append(row_squares)

    def on_click(self, row, col):
        self.toggle_cell(row, col)

    def toggle_cell(self, row, col):
        index = row * self.grid_size + col
        current_color = self.canvas.itemcget(self.grid[row][col], "fill")
        new_color = "black" if current_color == "white" else "white"
        self.canvas.itemconfig(self.grid[row][col], fill=new_color)
        self.binary_vector[index] = 1 if new_color == "black" else 0

    def create_submit_button(self):
        self.submit_button = tk.Button(self.root, text="Submit", command=self.submit)
        self.submit_button.pack(side="left", padx=5)

    def create_statistics_button(self):
        self.statistics_button = tk.Button(self.root, text="Perform Statistics",
                                           command=lambda: statistics.perform_experiment(self.hamming_net))
        self.statistics_button.pack(side="left", padx=5)

    def create_train_button(self):
        self.train_button = tk.Button(self.root, text="Train All Letters", command=self.train_all_letters)
        self.train_button.pack(side="left", padx=5)

    def create_train_button_10_times(self):
        self.train_button_10_times = tk.Button(self.root, text="Train All Letters 10 Times", command=self.train_all_letters_10_times)
        self.train_button_10_times.pack(side="left", padx=5)

    def submit(self):
        estimated_letter = Hamming_Net.HammingNeuralNetwork.calculate_closest_letter(self.hamming_net,
                                                                                     self.binary_vector)
        print(f"Estimated Letter: {estimated_letter}")
        print(f"Letter's vector: {Hamming_Net.HammingNeuralNetwork.letter_to_vector(estimated_letter)}")

    def train_all_letters(self):

        for letter, vector in self.hamming_net.hnn_dic.items():
            Hamming_Net.HammingNeuralNetwork.train_step(self.hamming_net, vector, letter, learning_rate=0.7)

        print("Training complete!")

    def train_all_letters_10_times(self):

        for i in range(10):

            for letter, vector in self.hamming_net.hnn_dic.items():
                Hamming_Net.HammingNeuralNetwork.train_step(self.hamming_net, vector, letter, learning_rate=0.7)


            print("Training complete!")


if __name__ == "__main__":
    hamming_net = Hamming_Net.HammingNeuralNetwork(randomDataset.letters)
    root = tk.Tk()
    gui = HammingNetGUI(root, hamming_net)  # Pass hamming_net to the GUI class
    root.mainloop()
