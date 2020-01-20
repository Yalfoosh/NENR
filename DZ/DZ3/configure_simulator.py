import os


def main():
    level = None

    while level is None:
        try:
            level = int(input("Upišite nivo koji želite pokrenuti (1, 2 ili 3): "))

            if level < 1 or level > 3:
                raise ValueError
        except:
            print("Krivi unos.\n")
            level = None

    config_string = "python {}\n{}\n20\n10\n".format(os.path.abspath(os.path.join(os.curdir, "main.py")), level)

    with open("simulator/config.txt", mode="w") as file:
        file.write(config_string)


main()
