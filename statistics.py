import random
import Hamming_Net
import correctLetters
import randomDataset


# Function to add noise to a binary vector
def add_noise(vector, noise_level):
    """
    Adds noise to a binary vector based on the specified noise level.
    :param vector: The binary vector of the letter
    :param noise_level: The noise percentage (from 0 to 100)
    :return: The noisy vector
    """
    noisy_vector = vector.copy()
    num_bits_to_flip = int(len(vector) * noise_level / 100)  # Number of bits to change
    indices_to_flip = random.sample(range(len(vector)), num_bits_to_flip)

    # Flip the selected bits
    for idx in indices_to_flip:
        noisy_vector[idx] = 1 - noisy_vector[idx]  # Flip the bit
    return noisy_vector


# Check the effect of noise on the network
noise_levels = [5, 10, 20]  # Noise levels to test


# Function to calculate accuracy
def calculate_accuracy(hnn, test_vectors, actual_letters):

    correct_predictions = 0
    letter_predictions = []
    for i in range(len(test_vectors)):
        recognized_letter = hnn.calculate_closest_correct_letter(test_vectors[i])
        letter_predictions.append(recognized_letter)
        if recognized_letter == actual_letters[i]:
            correct_predictions += 1
    print("Letter predictions: ", letter_predictions)
    return correct_predictions / len(test_vectors) * 100


# Create noisy vectors at different noise levels
def perform_experiment(hnn):
    for noise_level in noise_levels:
        noisy_vectors = []
        for letter, vector in hnn.hnn_dic.items():
            noisy_vector = add_noise(vector, noise_level)
            noisy_vectors.append(noisy_vector)

        # Calculate accuracy for vectors with noise at the given noise level
        accuracy = calculate_accuracy(hnn, noisy_vectors, list(hnn.hnn_dic.keys()))

        print(f"Accuracy with {noise_level}% noise: {accuracy}%")
