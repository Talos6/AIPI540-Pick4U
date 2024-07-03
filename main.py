from scripts.naive_approach import NaiveApproach

def run():
    naive = NaiveApproach()
    naive.process('orange.jpg', 5)

if __name__ == "__main__":
    run()