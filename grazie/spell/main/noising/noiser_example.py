from grazie.spell.main.noising.noiser import WordReplacementNoiser
from grazie.spell.main.noising.noiser import CharacterReplacementNoiser
from grazie.spell.main.noising.noiser import ProbabilisticCharacterReplacementNoiser

def main():
    example_texts = [
        "This is an example sentence to demonstrate noising in the neuspell repository.",
        "Here is another such amazing example !!",
        "Sasha Khvorov is my scientific director",
        "Now I'm writing some code"
    ]

    # word_repl_noiser = WordReplacementNoiser(language="english")
    # word_repl_noiser = CharacterReplacementNoiser(language="english")
    word_repl_noiser = ProbabilisticCharacterReplacementNoiser(language="english")
    word_repl_noiser.load_resources()
    noise_texts = word_repl_noiser.noise(example_texts)
    print(noise_texts)

if __name__ == '__main__':
    main()