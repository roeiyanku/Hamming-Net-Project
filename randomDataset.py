letters = {
    # They are initialized completely randomly on a 8 by 8 binary


    "A": [
        1, 0, 1, 1, 0, 1, 1, 0,
        0, 1, 0, 0, 1, 0, 1, 1,
        1, 0, 1, 0, 0, 1, 0, 0,
        1, 1, 1, 0, 1, 0, 0, 1,
        0, 1, 0, 1, 1, 0, 0, 1,
        0, 0, 1, 1, 1, 0, 1, 0,
        1, 1, 0, 0, 0, 1, 1, 1,
        0, 0, 1, 0, 1, 1, 1, 1,
    ],
    "B": [
        0, 1, 0, 0, 1, 0, 1, 0,
        1, 1, 0, 1, 0, 0, 1, 1,
        1, 0, 0, 1, 0, 1, 1, 0,
        0, 1, 1, 0, 0, 0, 1, 1,
        1, 0, 0, 0, 1, 1, 0, 0,
        1, 1, 0, 1, 1, 0, 0, 1,
        0, 0, 1, 1, 0, 1, 1, 0,
        1, 1, 0, 0, 1, 1, 0, 0,
    ],
    "C": [
        1, 1, 0, 1, 1, 1, 0, 0,
        1, 0, 0, 1, 0, 1, 0, 0,
        1, 1, 0, 1, 1, 0, 0, 1,
        0, 1, 0, 1, 0, 1, 0, 0,
        0, 1, 1, 0, 1, 0, 1, 1,
        0, 0, 1, 0, 0, 1, 0, 1,
        1, 0, 0, 1, 0, 0, 0, 1,
        1, 1, 1, 0, 1, 1, 0, 1,
    ],
    "D": [
        0, 1, 0, 0, 1, 1, 1, 1,
        1, 0, 1, 1, 1, 0, 1, 1,
        0, 0, 1, 0, 0, 1, 0, 1,
        1, 1, 0, 1, 1, 1, 0, 0,
        0, 1, 1, 1, 1, 0, 1, 0,
        1, 1, 1, 0, 0, 0, 0, 0,
        0, 1, 1, 0, 0, 0, 1, 0,
        1, 0, 1, 0, 1, 0, 0, 1,
    ],
    "E": [
        1, 1, 0, 1, 1, 0, 1, 0,
        0, 1, 1, 0, 1, 1, 0, 1,
        0, 0, 1, 1, 0, 1, 0, 1,
        1, 1, 0, 0, 1, 1, 0, 1,
        0, 0, 1, 0, 1, 1, 0, 1,
        1, 0, 1, 1, 0, 1, 0, 0,
        1, 1, 0, 1, 1, 0, 1, 1,
        0, 1, 0, 0, 1, 0, 1, 1,
    ],
    "F": [
        1, 0, 1, 1, 1, 0, 0, 1,
        0, 1, 0, 0, 0, 1, 1, 1,
        1, 0, 1, 0, 1, 0, 1, 0,
        0, 1, 0, 1, 0, 1, 1, 1,
        1, 0, 0, 0, 1, 1, 0, 0,
        1, 0, 1, 1, 0, 0, 0, 1,
        0, 1, 1, 0, 1, 1, 0, 0,
        1, 1, 0, 0, 0, 1, 1, 0,
    ],
    "G": [
        0, 1, 1, 0, 0, 1, 1, 1,
        1, 0, 0, 0, 1, 0, 1, 0,
        0, 1, 0, 1, 1, 1, 0, 1,
        1, 1, 0, 0, 1, 1, 0, 0,
        1, 0, 1, 1, 0, 0, 0, 1,
        0, 0, 0, 1, 0, 1, 0, 0,
        1, 1, 1, 0, 1, 0, 1, 1,
        0, 1, 0, 0, 1, 0, 1, 0,
    ],
    "H": [
        1, 0, 0, 1, 0, 1, 0, 0,
        1, 1, 1, 1, 1, 1, 0, 1,
        0, 1, 0, 0, 0, 0, 1, 1,
        1, 1, 0, 0, 0, 1, 0, 0,
        1, 1, 1, 0, 1, 1, 1, 0,
        0, 1, 0, 0, 0, 1, 1, 1,
        1, 0, 0, 1, 1, 0, 1, 1,
        1, 1, 1, 1, 0, 0, 1, 0,
    ],
    "I": [
        0, 0, 1, 1, 0, 0, 1, 1,
        1, 1, 0, 0, 0, 1, 0, 1,
        1, 0, 0, 1, 0, 1, 1, 0,
        1, 0, 0, 0, 0, 1, 0, 1,
        1, 0, 1, 0, 0, 0, 1, 1,
        0, 1, 1, 1, 0, 1, 1, 0,
        1, 0, 1, 0, 1, 1, 1, 1,
        0, 1, 1, 1, 0, 1, 1, 1,
    ],
    "J": [
        0, 0, 0, 1, 0, 0, 1, 1,
        1, 1, 1, 1, 0, 0, 0, 1,
        1, 0, 0, 0, 1, 1, 0, 0,
        1, 1, 0, 1, 0, 0, 0, 0,
        0, 1, 0, 1, 1, 1, 1, 0,
        0, 1, 1, 0, 0, 1, 1, 0,
        0, 1, 1, 0, 1, 1, 1, 1,
        1, 0, 0, 1, 0, 1, 1, 0,
    ],
    "K": [
        1, 0, 0, 0, 1, 1, 0, 0,
        0, 1, 0, 1, 0, 0, 1, 0,
        0, 1, 1, 0, 0, 1, 0, 1,
        1, 0, 0, 0, 0, 1, 0, 0,
        1, 1, 0, 0, 0, 0, 0, 0,
        1, 1, 1, 0, 0, 1, 1, 0,
        1, 1, 1, 1, 0, 0, 1, 1,
        1, 1, 1, 0, 1, 1, 0, 1,
    ],
    "L": [
        0, 1, 1, 0, 0, 1, 0, 1,
        1, 0, 0, 0, 1, 1, 1, 0,
        0, 1, 1, 1, 0, 1, 1, 1,
        0, 0, 1, 0, 1, 0, 1, 0,
        1, 1, 0, 0, 1, 1, 0, 1,
        0, 1, 0, 1, 1, 0, 1, 1,
        1, 0, 0, 1, 1, 0, 0, 1,
        0, 1, 1, 1, 1, 0, 0, 0,
    ],
    "M": [
        1, 1, 0, 1, 0, 0, 1, 1,
        0, 0, 1, 0, 1, 1, 0, 1,
        0, 1, 1, 0, 0, 1, 0, 0,
        1, 0, 1, 1, 1, 0, 0, 1,
        1, 1, 1, 1, 0, 1, 1, 0,
        1, 0, 0, 1, 0, 1, 0, 0,
        0, 1, 0, 1, 1, 0, 1, 1,
        1, 1, 0, 1, 1, 1, 0, 1,
    ],
    "N": [
        1, 0, 1, 1, 0, 0, 1, 1,
        0, 1, 0, 0, 1, 1, 0, 1,
        1, 1, 0, 0, 0, 1, 0, 1,
        0, 0, 1, 1, 0, 0, 1, 0,
        1, 1, 1, 0, 1, 1, 1, 1,
        1, 0, 0, 1, 0, 0, 1, 1,
        1, 1, 0, 1, 1, 0, 1, 0,
        0, 1, 1, 1, 1, 0, 1, 1,
    ],
    "O": [
        1, 0, 1, 0, 1, 1, 0, 1,
        0, 0, 0, 1, 1, 0, 1, 0,
        0, 1, 1, 1, 1, 1, 0, 1,
        1, 1, 0, 0, 0, 0, 0, 1,
        1, 0, 1, 1, 1, 0, 0, 0,
        0, 1, 0, 0, 0, 1, 1, 0,
        0, 0, 1, 1, 1, 1, 1, 1,
        1, 0, 1, 0, 1, 1, 1, 0,
    ],
    "P": [
        1, 0, 1, 0, 1, 1, 0, 1,
        0, 0, 0, 0, 1, 0, 0, 1,
        1, 1, 0, 0, 0, 1, 0, 0,
        0, 1, 1, 0, 1, 1, 0, 1,
        1, 0, 0, 0, 0, 1, 0, 1,
        0, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 0, 0, 0, 1, 0, 0,
        0, 0, 1, 1, 0, 0, 1, 0,
    ],
    "Q": [
        1, 1, 1, 0, 0, 0, 1, 1,
        1, 0, 0, 1, 1, 1, 1, 0,
        0, 1, 1, 0, 1, 1, 1, 0,
        1, 0, 0, 0, 1, 0, 1, 1,
        1, 0, 1, 1, 0, 0, 1, 0,
        0, 0, 1, 1, 1, 0, 0, 0,
        0, 1, 0, 1, 1, 1, 1, 1,
        0, 1, 1, 0, 1, 1, 0, 0,
    ],
    "R": [
        1, 0, 1, 0, 1, 1, 1, 1,
        0, 1, 0, 0, 1, 0, 0, 0,
        1, 0, 0, 1, 0, 1, 1, 0,
        1, 1, 1, 0, 1, 1, 0, 1,
        0, 0, 0, 1, 0, 1, 1, 0,
        1, 1, 1, 0, 0, 0, 1, 0,
        1, 0, 0, 0, 0, 0, 1, 1,
        1, 1, 0, 1, 1, 0, 1, 0,
    ],
    "S": [
        0, 1, 0, 0, 1, 1, 1, 1,
        1, 0, 1, 1, 0, 0, 1, 0,
        1, 1, 0, 0, 0, 0, 1, 0,
        1, 0, 1, 0, 1, 1, 1, 1,
        1, 1, 0, 0, 0, 1, 1, 1,
        0, 0, 1, 0, 1, 0, 0, 1,
        1, 1, 1, 0, 0, 1, 0, 1,
        0, 1, 0, 1, 1, 1, 0, 0,
    ],
    "T": [
        1, 1, 1, 0, 0, 1, 0, 1,
        1, 0, 0, 0, 1, 0, 1, 0,
        0, 1, 0, 1, 0, 1, 1, 1,
        1, 0, 0, 0, 1, 1, 1, 1,
        0, 1, 0, 0, 0, 0, 0, 0,
        0, 1, 1, 0, 1, 1, 1, 0,
        0, 0, 1, 1, 0, 1, 1, 1,
        1, 0, 0, 1, 1, 0, 1, 0,
    ],
    "U": [
        1, 0, 1, 0, 1, 1, 0, 1,
        1, 1, 0, 1, 1, 0, 0, 0,
        1, 1, 1, 0, 1, 0, 1, 1,
        1, 1, 1, 0, 1, 1, 0, 1,
        1, 1, 0, 0, 0, 1, 0, 0,
        1, 0, 1, 0, 1, 0, 0, 1,
        0, 1, 1, 0, 1, 0, 0, 1,
        0, 0, 1, 1, 1, 1, 1, 1,
    ],
    "V": [
        1, 1, 0, 1, 1, 1, 0, 0,
        1, 1, 1, 1, 1, 0, 1, 1,
        1, 0, 0, 0, 1, 0, 1, 0,
        1, 0, 0, 1, 0, 1, 1, 0,
        1, 1, 1, 0, 0, 0, 1, 0,
        1, 1, 1, 1, 0, 1, 1, 0,
        0, 0, 0, 0, 1, 0, 0, 1,
        1, 1, 1, 0, 1, 0, 1, 0,
    ],
    "W": [
        0, 1, 1, 1, 1, 0, 1, 0,
        1, 0, 0, 1, 1, 1, 1, 1,
        0, 1, 0, 1, 0, 0, 0, 1,
        1, 1, 0, 0, 1, 1, 0, 0,
        1, 0, 1, 0, 1, 0, 0, 1,
        0, 1, 1, 1, 1, 1, 0, 1,
        1, 1, 0, 0, 0, 0, 1, 0,
        1, 1, 0, 1, 1, 1, 0, 0,
    ],
    "X": [
        1, 0, 1, 1, 1, 0, 0, 1,
        0, 1, 1, 1, 0, 1, 0, 0,
        1, 1, 1, 0, 1, 0, 1, 1,
        0, 1, 0, 1, 1, 1, 0, 0,
        0, 0, 1, 1, 1, 1, 1, 0,
        0, 1, 1, 0, 0, 1, 0, 1,
        1, 0, 1, 1, 1, 1, 1, 0,
        1, 0, 1, 0, 1, 0, 1, 1,
    ],
    "Y": [
        1, 0, 0, 1, 1, 1, 0, 1,
        1, 0, 1, 0, 1, 0, 1, 1,
        1, 1, 0, 0, 0, 0, 1, 0,
        1, 0, 0, 1, 1, 1, 1, 0,
        1, 1, 1, 1, 0, 0, 0, 0,
        1, 1, 0, 0, 1, 0, 0, 0,
        0, 1, 1, 1, 0, 1, 1, 1,
        1, 0, 1, 0, 1, 1, 0, 0,
    ],
    "Z": [
        1, 1, 0, 1, 1, 0, 0, 0,
        0, 1, 0, 1, 1, 0, 1, 0,
        1, 1, 0, 0, 1, 0, 0, 1,
        0, 1, 1, 1, 1, 1, 0, 0,
        1, 1, 0, 0, 1, 0, 1, 1,
        0, 1, 1, 1, 1, 0, 0, 1,
        0, 1, 1, 1, 0, 1, 0, 0,
        1, 1, 1, 0, 0, 0, 1, 1,
    ],
}

