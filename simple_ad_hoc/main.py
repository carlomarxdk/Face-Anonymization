from  simple_ad_hoc import anonymizer

def main():
    text_in = input('Method to use [blur, pixelate, noise]:')
    METHOD = str(text_in)
    print('Chosen method is', METHOD)

    for k in range(1,101):
        anonymizer.anonymise(METHOD, k)


if __name__ == '__main__':
    main()
