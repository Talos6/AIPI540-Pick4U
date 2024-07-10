from scripts.naive_approach import NaiveApproach
from scripts.ml_approach import MLApproach

def run():
    ml_approach = MLApproach()
    ml_approach.process('orange.jpg', 5)

if __name__ == "__main__":
    run()