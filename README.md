
# Hamming Neural Network for Letter Recognition

This project implements a simple Hamming-based neural network for recognizing letters represented as 8x8 binary vectors. The system is designed to handle noisy inputs and predict the most likely letter based on the Hamming distance between the input and predefined letter vectors.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Getting Started](#getting-started)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Testing and Results](#testing-and-results)
- [Contributions](#contributing)


---

## Overview

The Hamming Neural Network is built to recognize letters from noisy binary vectors. The network uses the Hamming distance to compare a test vector with a predefined set of letter vectors, predicting the letter that most closely matches the test input. It also includes an experiment to evaluate the accuracy of the predictions on noisy data.

---

## Features

- **Letter Recognition**: Recognizes 8x8 binary vector representations of letters (e.g., 'A', 'B', etc.).
- **Noise Handling**: Simulates noisy input data and evaluates the model’s accuracy in real-world scenarios.
- **Accuracy Calculation**: Computes the accuracy based on correct predictions against noisy vectors.
- **Simple and Efficient**: Uses basic Hamming distance for computation, making it easy to understand and fast for small-scale tasks.

---

## Getting Started

To get started with this project, you'll need Python installed on your machine along with the required libraries.

### Prerequisites

- Python 3.x (Tested on Python 3.13)
- Tkinter (for GUI)
- Other libraries: `numpy`

### Installation
Clone this repository:
   ```bash
  git clone https://github.com/roeiyanku/Hamming-Net-Project.git
  cd Hamming-Net-Project
   ```


---

## Project Structure

```
Hamming-Net-Project/
│
├── README.md              # Project documentation
├── GUI.py                 # Tkinter-based GUI for the Hamming neural network
├── statistics.py          # Code for calculating accuracy and statistics
├── Hamming_Net.py         # Core implementation of the Hamming neural network
├── randomDatset.py        # Predefined RANDOM letter vectors data, A-Z all initialized randomly
├── correctLetters.py      # Predefined CORRECT letter vectors data, A-Z 

```

---

## Usage

1. **Run the GUI**: Start the application using the command:
   ```bash
   python GUI.py
   ```

   This will open a Tkinter-based window where you can interact with the Hamming neural network and perform experiments.

2. **Test Accuracy**: The `statistics.py` file contains a method to calculate the accuracy of the model based on noisy input vectors and actual letter labels.

3. **Train Dataset**: The GUI allows you to train a random set of vectors and to test them against a correct set of binary vectors and to submit the results. You can then see if the results are as you predicted. 

---

## Testing and Results

- The network has been tested with noisy binary vectors to simulate real-world scenarios.
- Accuracy is calculated by comparing predicted letters against actual labels, showing the performance of the model under noisy conditions.

Example of results after performing the experiment:
```bash
Accuracy: 95%
Letter predictions: ['A', 'B', 'C', 'D', ...]
```

---

## Contributions
Roei Yanku – Project creator and developer

Merav Chkroun – My lovely professor who taught me the Computational Neuroscience class at Ariel University
