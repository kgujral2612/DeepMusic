#
# reference: https://www.daniweb.com/programming/software-development/code/216979/embed-and-play-midi-music-in-your-code-python
#

import pygame
import base64
import sys


def play_music(music_file):
    """
    stream music with mixer.music module in blocking manner
    this will stream the sound from disk while playing
    """
    pygame.mixer.music.load(music_file)
    clock = pygame.time.Clock()
    pygame.mixer.music.play()
    # check if playback has finished
    while pygame.mixer.music.get_busy():
        clock.tick(30)


def play_file(music_file):
    mid64 = '''\
    TVRoZAAAAAYAAQAGAJBNVHJrAAAAJwD/WAQEAhgIAP9ZAgAAAP9RAwfCOwD/BgpGaXNoLVBvbGth
    AP8vAE1UcmsAAANoAP8hAQAA/wMOQWNjb3JkaW9uIEhpZ2gAwRUAsQd/ALEKQAD/AQpGaXNoIFBv
    bGthAJFOZGhOAAFQYyNQAAFSZBdSADFQZCNQACVOYxdOADFMZCNMACVLYxdLADFJYyNJACVOZCNO
    AAFNYyNNAAFOYyNOAAFQZCNQAAFOZCNOAAFMZCNMAAFLZCNLAAFHYyNHAAFLYyNLAAFKYyNKAAFL
    YyNLAAFMZCNMAAFLZCNLAAFJZCNJAAFHYyNHAAFCYyNCAAE/YyM/AAFCYyNCAAFHZCNHAAFLZCNL
    AAFOYxFOADdQYyNQACVQYyNQACVJYyNJACVJZIEPSQABTGQjTAABS2QjSwABTGMjTAABTmQjTgAB
    TGMjTAABS2QjSwABSWMjSQABSGMjSAABSWQjSQABQmMjQgABRmQjRgABSWQjSQABTGQjTAABSWMj
    SQABRmQjRgABRGQjRAABQmMjQgABRmQjRgABSWQjSQABTGMjTAABUmMRUgA3UGQjUAAlUGMjUAAl
    S2MjSwAlS2SBD0sAAU5jI04AAU1kI00AAU5jI04AAVBjI1AAAU5jI04AAUxkI0wAAUtjI0sAAUdk
    I0cAAUtjI0sAAUpkI0oAAUtkI0sAAUxjI0wAAUtkI0sAAUlkI0kAAUdjI0cAAUJjI0IAAT9kIz8A
    AUJjI0IAAUdkI0cAAUtkI0sAAU5kI04AAUdjI0cAAUtjI0sAAU5kI04AAVBjgXtQACVQZCNQAAFP
    YyNPAAFQZCNQAAFSYyNSAAFVYyNVAAFTYyNTAAFSYyNSAAFQYyNQAAFOYyNOAAFNYyNNAAFOZCNO
    AAFPZCNPAAFQZCNQAAFOYyNOAAFMZCNMAAFLYyNLAAFMZCNMAAFLYyNLAAFMZCNMAAFNZCNNAAFO
    ZCNOAAFMZCNMAAFLZCNLAAFJYyNJAAFLZCNLACVMYyNMACVNZCNNACVOYyNOACVQZCNQAAFPZCNP
    AAFQYyNQAAFSZCNSAAFVYyNVAAFTYyNTAAFSYyNSAAFQZCNQAAFOZCNOAAFNYyNNAAFOZCNOAAFP
    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
    HQA3HmRZHgA3IGRZIAA3JWRZJQA3HmRZHgA3I2MvIwAZImMvIgAZIGMvIAAZHmQvHgAZHGRZHAA3
    HWRZHQA3HmRZHgA3IGNZIAA3JWNZJQA3HmRZHgA3I2QvIwAZHmQvHgAZI2MvIwAA/y8ATVRyawAA
    AYYA/yEBAAD/AwtCYXNzIERvdWJsZQDEIgC0B24AtApAAJQqY2sqAAEsYyMsAAEuYxcuADEsZCMs
    ACUqZBcqADEoYyMoACUnYxcnADElYyMlACUjZEcjAEkeZFkeADcjZEcjAEkeZFkeADcjZEcjAAEe
    ZCMeACUjZDUjABMnZDUnABMqZFkqADclZFklADceY1keADclZEclAEkeY1keADclZEclAEkeYzUe
    ABMfZEcfAAEgZCMgACUiZEciAAEjZCMjACUiYyMiACUgZCMgACUeY0ceAAEjZCMjAG0eY1keADcj
    Y0cjAEkeZFkeADcjYyMjACUeY0ceAAEjZCMjACUnYyMnACUoZEcoAEkjZEcjAEkcY1kcADcdY1kd
    ADceZFkeADcgY1kgADclY1klADceY1keADcjYy8jABkiZC8iABkgZC8gABkeYy8eABkcY1kcADcd
    ZFkdADceZFkeADcgY1kgADclY1klADceZFkeADcjZC8jABkeYy8eABkjYy8jAAD/LwBNVHJrAAAD
    PQD/IQEAAP8DBURydW1zALkHcQC5CkAAmTluACZ3ACRuASYAAyQAADkADCZYASYADyZVASYADyZU
    ASYADyZSASYADiZPASYADiZNASYADSZ6BCYAICZ5ACRtBCQAACYAQyZ6BCYARSZ5ACRsBCQAACYA
    QiZ+BCYAFyZ8BCYADSZ/BCYAGCZ6AiRnAiYAAiQAQiZ/BCYARiRvAjluAiQAAjkAQip3BCoAQyRl
    BCQARSpoBCoAQyRmBCQARSpqBCoAQyRoBCQARSpsBCoAQyRoBCQARSprBCoAQyRkBCQARSpuBCoA
    QyRjBCQARSpkBCoAQyRmBCQARSppBCoAQyRmBCQARSpxBCoARCRmBCQARCp6BCoARCRoBCQARCpy
    BCoARCRlBCQARCp/BCoARSRpBCQAQyp6BCoARCRqBCQAQyp6BCoARSRqBCQARCp6BCoARCRpBCQA
    RCp/BCoARSRrBCQAQyV6ACp1BCoAQyUAASRnBCQARCV7ACp7BCoAQyUAASRlBCQARCV7ACp9BCoA
    QyUAASRoBCQARCV7ACp1BCoAQyUAASRrBCQARCV6ACp+BCoAQyUAASRpBCQAQyp3ASV7ACRoAyoA
    ASQAQyUAASRnBCQARCV7ACp6BCoAICV6IyUAAiRpBCQAHiUAJSV6ACp5BCoAQyUAASRrBCQARCV6
    ACp4BCoAQyUAASRwBCQARCV6ACp8BCoAQyUAASRrBCQAQyp6ASV6AyoARCUAASRmBCQARCV6ACp9
    BCoAQyUAASRpBCQARCV7ACp3BCoAQyUAASRqBCQARCV7ACp0BCoAQyUAASRqBCQARCV6ACp6BCoA
    ICV6IyUAASRlBCQAHyUAJCp6ASV7AyoARCUAADlhAiR/AjkAAiQAH7kHdBG5B3ETmSV6ACp4BCoA
    QyUAASRuBCQARCV6ACpuBCoAQyUAACRwBCQARCp3ASV7AyoARCUAASRnBCQAQyp2ASV7AyoARCUA
    ASRuBCQARCV7ACp6BCoAQyUAASRtBCQAQyp6ASV7AyoARCUAASV6ACRsBCQAQiZ/ASUAAyYAFyZ8
    BCYADSZ/BCYAGCZ5Ajl6ACRoAiYAAiQAgQs5AAD/LwA=
    '''

    # convert back to a binary midi and save to a file in the working directory
    fish = base64.b64decode(mid64)
    fout = open(music_file, "wb")
    fout.write(fish)
    fout.close()

    freq = 44100  # audio CD quality
    bitsize = -16  # unsigned 16 bit
    channels = 2  # 1 is mono, 2 is stereo
    buffer = 1024  # number of samples
    pygame.mixer.init(freq, bitsize, channels, buffer)

    # optional volume 0 to 1.0
    pygame.mixer.music.set_volume(0.8)

    try:
        # use the midi file you just saved
        play_music(music_file)
    except KeyboardInterrupt:
        # if user hits Ctrl/C then exit
        # (works only in console mode)
        pygame.mixer.music.fadeout(1000)
        pygame.mixer.music.stop()
        raise SystemExit


if __name__ == "__main__":
    midi_path = sys.argv[1]
    if midi_path:
        play_file(midi_path)
