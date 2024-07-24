from scripts.naive_approach import NaiveApproach
from scripts.ml_approach import MLApproach
from scripts.nn_approach import NNApproach

def run():
    nn_approach = NNApproach()
    nn_approach.process('orange.jpg', 10)

if __name__ == "__main__":
    run()