#coding=utf-8

import argparse
from PIL import ImageFont
from PlateCommon import *

TEMPLATE_IMAGE = "./template/template.bmp"

class GenPlateScene:
    def __init__(self, fontCh, fontEng, NoPlates):
        self.fontC = ImageFont.truetype(fontCh, 43, 0)    # 省简称使用字体
        self.fontE = ImageFont.truetype(fontEng, 60, 0)   # 字母数字字体
        # self.img = np.array(Image.new("RGB", (226, 70), (255, 255, 255)))
        # self.bg = cv2.resize(cv2.imread(TEMPLATE_IMAGE), (226, 70))
        self.img = np.array(Image.new("RGB", (226, 70), (255, 255, 255)))
        self.bg = cv2.resize(cv2.imread(TEMPLATE_IMAGE), (226, 70))
        self.noplates_path = []
        for parent, _, filenames in os.walk(NoPlates):
            for filename in filenames:
                self.noplates_path.append(parent + "/" + filename)

    def gen_plate_string(self, iter, perSize):
        plate_str = ""
        i = iter // perSize
        # iterChar = (iter % perSize) // 9
        for cpos in range(7):
            if cpos == 0:
                plate_str += chars[i] #+r(31)]
            elif cpos == 1:
                plate_str += chars[41 + r(24)]
            else:
                plate_str += chars[31 + r(34)]
        return plate_str

    def draw(self, val):
        offset= 2
        self.img[0:70, offset+8:offset+8+23] = GenCh(self.fontC, val[0])
        self.img[0:70, offset+8+23+6:offset+8+23+6+23] = GenCh1(self.fontE, val[1])
        for i in range(5):
            base = offset+8+23+6+23+17+i*23+i*6
            self.img[0:70, base:base+23]= GenCh1(self.fontE, val[i+2])
        return self.img

    def generate(self,text):
        print(text + " " + str(len(text)))
        
        fg = self.draw(text)
        fg = cv2.bitwise_not(fg)
        com = cv2.bitwise_or(fg, self.bg)
        com = rot(com, r(20)-10, com.shape, 10)
        com = rotRandrom(com, 5, (com.shape[1], com.shape[0]))
        com = tfactor(com) # 调灰度

        return com

    def gen_batch(self, perSize, outDir):
        for i in range(perSize*31):
            outputPath = "./train"
            if (not os.path.exists(outputPath)):
                os.mkdir(outputPath)
            plate_str = self.gen_plate_string(i, perSize)
            img =  self.generate(plate_str)
            if img is None:
                continue
            cv2.imwrite(outputPath + "/" + plate_str + ".jpg", img)

        #generate test data
        for i in range(int(perSize/5.0) * 31):
            # outputPath = outDir + str(i // perSize) + "/"
            outputPath = "./test"
            if (not os.path.exists(outputPath)):
                os.mkdir(outputPath)
            plate_str = self.gen_plate_string(i, perSize)
            img = self.generate(plate_str)
            if img is None:
                continue
            cv2.imwrite(outputPath + "/" + plate_str + ".jpg", img)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('gen_count_per_province', type=int, help='gen car plt num for per_province')
    parser.add_argument('--bg_dir', default='./Background', help='bg_img dir')
    parser.add_argument('--out_dir', default='../data', help='output dir')
    
    return parser.parse_args()

def main(args):
    G = GenPlateScene("./font/platech.ttf", './font/platechar.ttf', args.bg_dir)
    G.gen_batch(args.gen_count_per_province, args.out_dir)

if __name__ == '__main__':
    main(parse_args())