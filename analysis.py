from PIL import Image


PATH = './inputs/birds.png'

if __name__ == '__main__':
    img = Image.open(PATH).convert('RGB')
