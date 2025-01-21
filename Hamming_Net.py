import correctLetters
import numpy as np


class HammingNeuralNetwork:
    def __init__(self, dictionary):
        self.hnn_dic = dictionary  #Initialize the network in our case as random vectors

    @staticmethod
    def calculate_Hamming_Distance(v1,
                                   v2):  #The core of the Hamming distance Idea, how we check the hamming distance between two vectors, but we will use a dot product
        distance = 0
        for i in range(len(v1)):

            if v1[i] != v2[i]:
                distance += 1

        return distance

    @staticmethod
    def calculate_dot_product(v1, v2):  # We will use The dot product instead of Hamming distance
        dot_product = 0
        for i in range(len(v1)):
            if v1[i] == 1 and v2[i] == 1:
                dot_product += 1
        return dot_product

    def calculate_closest_letter(self, vector):  # compares the letter in the current data
        closest_letter = "A"
        closest_distance = 0

        for letter, letter_vector in self.hnn_dic.items(): # Here we implement Winner takes it all(MAXNET)
            i_distance = self.calculate_dot_product(vector, letter_vector)

            if i_distance > closest_distance:
                closest_distance = i_distance
                closest_letter = letter

        return closest_letter  # Return after checking all letters

    def calculate_closest_correct_letter(self, vector):  # compares the letter in the correct data
        closest_letter = "A"
        closest_distance = 0

        for letter, letter_vector in correctLetters.letters.items():  #Here we compare to out correctLetters data, this can be changed.
            i_distance = self.calculate_dot_product(vector, letter_vector)

            if i_distance > closest_distance: # Here we implement the WInner takes it all (MaxNet)
                closest_distance = i_distance
                closest_letter = letter

        return closest_letter

    # Function to adjust our weights and change the binary vectors
    def train(self, input_vector, supposed_to_get_letter, learning_rate):

        for i in range(len(input_vector)):

            # Get the expected bit for the current letter
            correct_bit = correctLetters.letters[supposed_to_get_letter][i]

            # We use learning rate to change our binary
            self.hnn_dic[supposed_to_get_letter][i] += learning_rate * (
                    correct_bit - self.hnn_dic[supposed_to_get_letter][i])

            # The way I chose to implement our training model, this can be changed
            if self.hnn_dic[supposed_to_get_letter][i] >= 0.5:
                self.hnn_dic[supposed_to_get_letter][i] = 1
            else:
                self.hnn_dic[supposed_to_get_letter][i] = 0

    def train_step(self, input_vector, supposed_to_get_letter, learning_rate):

        predicted_letter = self.calculate_closest_correct_letter(input_vector)

        print("Letter we were supposed to get:", supposed_to_get_letter)

        print("Letter we got:", predicted_letter)

        if predicted_letter != supposed_to_get_letter:
            # If the prediction is incorrect, train the network with the correct letter
            self.train(input_vector, supposed_to_get_letter, learning_rate)
            print(f"Updated weights for '{supposed_to_get_letter}'.")

            print()
        else:
            # If the prediction is correct, no update needed
            print(f"Prediction was correct. No update .")
            print()

    @staticmethod
    def letter_to_vector(letter):
        letter_to_vector_dict = {
            "A": [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # A = 1 at index 0
            "B": [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # B = 1 at index 1
            "C": [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # C = 1 at index 2
            "D": [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # D = 1 at index 3
            "E": [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # E = 1 at index 4
            "F": [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # F = 1 at index 5
            "G": [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # G = 1 at index 6
            "H": [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # H = 1 at index 7
            "I": [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # I = 1 at index 8
            "J": [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # J = 1 at index 9
            "K": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # K = 1 at index 10
            "L": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # L = 1 at index 11
            "M": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # M = 1 at index 12
            "N": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # N = 1 at index 13
            "O": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # O = 1 at index 14
            "P": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # P = 1 at index 15
            "Q": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Q = 1 at index 16
            "R": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # R = 1 at index 17
            "S": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # S = 1 at index 18
            "T": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # T = 1 at index 19
            "U": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # U = 1 at index 20
            "V": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # V = 1 at index 21
            "W": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],  # W = 1 at index 22
            "X": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],  # X = 1 at index 23
            "Y": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],  # Y = 1 at index 24
            "Z": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # Z = 1 at index 25
        }

        return letter_to_vector_dict.get(letter, [])
