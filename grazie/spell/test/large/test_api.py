from grazie.spell.main.api.api import ServerSpellApi


def test_api():
    ServerSpellApi.download_model()
    ServerSpellApi.initialize()

    corrections = ServerSpellApi.check(["helo word!"], max_count=3)[0]

    assert len(corrections) == 1
    assert corrections[0].start == 0 and corrections[0].finish == 4

    variants = corrections[0].variants
    assert variants[0].substitution == "hello" and variants[0].score == 0.9525
    assert variants[1].substitution == "help" and variants[1].score == 0.9314
    assert variants[2].substitution == "held" and variants[2].score == 0.9089


if __name__ == '__main__':
    test_api()
