import random

def simple_attack(text):
    variants = []

    variants.append(text.replace("I ", "I really "))
    variants.append(text.replace("was", "had been"))
    variants.append(text.replace("didn't", "did not"))
    variants.append(text.replace("very", "extremely"))
    variants.append(text.replace("my", "my own"))

    return random.choice(variants)