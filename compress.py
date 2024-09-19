from PIL import Image
import math
import numpy as np
from bitstring import BitArray, ConstBitStream
from dahuffman import HuffmanCodec
import pickle
import sys

# RGB 转 YUV
def RGB2YUV(r, g, b):
    y = 0.299 * r + 0.587 * g + 0.114 * b
    u = -0.169 * r - 0.331 * g + 0.5 * b + 128
    v = 0.5 * r - 0.419 * g - 0.081 * b + 128
    return y, u, v

# YUV 转 RGB
def YUV2RGB(y, u, v):
    r = min(max(0, (y + 1.4075 * (v - 128))), 255)
    g = min(max(0, (y - 0.3455 * (u - 128) - 0.7169 * (v - 128))), 255)
    b = min(max(0, (y + 1.779 * (u - 128))), 255)
    return np.uint8(r), np.uint8(g), np.uint8(b)

# 用于构建哈夫曼解码器
def guess_concat(data):
    return {
        type(u""): u"".join,
        type(b""): bytes,
    }.get(type(data), list)

class MyHuffman:
    def __init__(self):
        self.count = {}
    
    # 计数器清零
    def Clear(self):
        self.count = {}

    # 计数
    def Count(self, arr):
        for i in arr:
            if i in self.count:
                self.count[i] += 1
            else:
                self.count[i] = 1
    
    # 生成码表
    def Create_codec(self):
        self.codec = HuffmanCodec.from_frequencies(self.count, eof = (0, 0))

    # 从外部加载码表
    def Load_codec(self, table):
        self.codec = HuffmanCodec(table, concat = guess_concat(next(iter(table))), check = False, eof = (0, 0))
    
    # 获取码表
    def Get_codec(self):
        return self.codec.get_code_table()
    
    # 编码
    def Encode(self, arr):
        return self.codec.encode(arr)
    
    # 解码
    def Decode(self, b):
        return self.codec.decode(b)

class MyJPEG:
    def __init__(self):
        self.huffman_coder = MyHuffman()

        # DCT 系数
        self.mat_dct = np.zeros((8, 8), dtype = np.float64)
        for i in range(8):
            k = math.sqrt(1. / 8.) if i == 0 else math.sqrt(2. / 8.)
            for j in range(8):
                self.mat_dct[i, j] = k * np.cos(np.pi * i * (2 * j + 1) / 16)

        # 指数
        self.mat_log2 = np.zeros(258, dtype = np.int32)
        for i in range(1, 257):
            self.mat_log2[i] = math.ceil(math.log2(i))

        # 亮度量化矩阵
        self.mat_yq = np.array([
            [16, 11, 10, 16, 24, 40, 51, 61],
            [12, 12, 14, 19, 26, 58, 60, 55],
            [14, 13, 16, 24, 40, 57, 69, 56],
            [14, 17, 22, 29, 51, 87, 80, 62],
            [18, 22, 37, 56, 68, 109, 103, 77],
            [24, 35, 55, 64, 81, 104, 113, 92],
            [49, 64, 78, 87, 103, 121, 120, 101],
            [72, 92, 95, 98, 112, 100, 103, 99]
        ])

        # 色度量化矩阵
        self.mat_uvq = np.array([
            [17, 18, 24, 47, 99, 99, 99, 99],
            [18, 21, 26, 66, 99, 99, 99, 99],
            [24, 26, 56, 99, 99, 99, 99, 99],
            [47, 66, 99, 99, 99, 99, 99, 99],
            [99, 99, 99, 99, 99, 99, 99, 99],
            [99, 99, 99, 99, 99, 99, 99, 99],
            [99, 99, 99, 99, 99, 99, 99, 99],
            [99, 99, 99, 99, 99, 99, 99, 99],
        ])

        # zigzag 索引
        self.mat_zigzag_x = np.array([
            0, 0, 1, 2, 1, 0, 0, 1,
            2, 3, 4, 3, 2, 1, 0, 0,
            1, 2, 3, 4, 5, 6, 5, 4,
            3, 2, 1, 0, 0, 1, 2, 3,
            4, 5, 6, 7, 7, 6, 5, 4,
            3, 2, 1, 2, 3, 4, 5, 6,
            7, 7, 6, 5, 4, 3, 4, 5,
            6, 7, 7, 6, 5, 6, 7, 7
        ])
        self.mat_zigzag_y = np.array([
            0, 1, 0, 0, 1, 2, 3, 2,
            1, 0, 0, 1, 2, 3, 4, 5,
            4, 3, 2, 1, 0, 0, 1, 2,
            3, 4, 5, 6, 7, 6, 5, 4,
            3, 2, 1, 0, 1, 2, 3, 4,
            5, 6, 7, 7, 6, 5, 4, 3,
            2, 3, 4, 5, 6, 7, 7, 6,
            5, 4, 5, 6, 7, 7, 6, 7
        ])
        self.mat_gazgiz = np.array([
            [0,  1,  5,  6,  14, 15, 27, 28],
            [2,  4,  7,  13, 16, 26, 29, 42],
            [3,  8,  12, 17, 25, 30, 41, 43],
            [9,  11, 18, 24, 31, 40, 44, 53],
            [10, 19, 23, 32, 39, 45, 52, 54],
            [20, 22, 33, 38, 46, 51, 55, 60],
            [21, 34, 37, 47, 50, 56, 59, 61],
            [35, 36, 48, 49, 57, 58, 62, 63],
        ])

    # DCT 变换 
    def DCT(self, mat):
        return np.dot(np.dot(self.mat_dct, mat), np.transpose(self.mat_dct))
    
    # 逆 DCT 变换
    def IDCT(self, mat):
        return np.dot(np.dot(np.transpose(self.mat_dct), mat), self.mat_dct)
    
    # 量化
    def Quantize(self, mat, y):
        return np.round(mat / (self.mat_yq if y == 1 else self.mat_uvq))
    
    # 逆量化
    def DeQuantize(self, mat, y):
        return mat * (self.mat_yq if y == 1 else self.mat_uvq)
    
    # zigzag
    def Zigzag(self, mat):
        return mat[self.mat_zigzag_x, self.mat_zigzag_y]

    # 逆 zigzag
    def DeZigzag(self, arr):
        return arr[self.mat_gazgiz]
    
    # RLE 编码
    def RLE(self, arr):
        res = []
        cnt = 0
        for v in arr:
            if cnt < 15 and v == 0:
                cnt += 1
            else:
                res.append((cnt, int(v)))
                cnt = 0
        res.append((0, 0))
        return np.array(res)
    
    # 逆 RLE 编码
    def DeRLE(self, arr):
        res = []
        for v in arr:
            if v[0] == 0 and v[1] == 0:
                break
            for _ in range(v[0]):
                res.append(0)
            res.append(int(v[1]))
        # 补足 0
        while len(res) < 64:
            res.append(0)
        return np.array(res).astype(np.int32)
    
    # 分解为 size amplitude
    def SADecomp(self, arr):
        s = []
        a = BitArray()
        for v in arr:
            si = self.mat_log2[abs(v[1]) + 1] # 获取 log2 值
            s.append((v[0], si))
            if si == 0:
                continue
            ai = BitArray(uint = int(abs(v[1])), length = si)
            if v[1] < 0:
                ai = ~ai
            a.append(ai)
        return (s, a)
    
    # 合并 size amplitude
    def SAComp(self, s, a):
        res = []
        for i in range(len(s)):
            f, si = s[i]
            ai = a[i]
            v = (ai.uint if ai[0] else -((~ai).uint)) if ai.length > 0 else 0
            res.append((f, v))
        return np.array(res).astype(np.int32)
    
    # 压缩
    def Encode(self, image_name):
        self.huffman_coder.Clear()
        img = Image.open(image_name)
        sizew, sizeh = img.size
        # 获取源文件各像素颜色
        imgdata = np.array(img).reshape(sizeh, sizew, 3)

        compy = []
        compu = []
        compv = []

        # 先输出长宽
        output = BitArray(uint = sizew, length = 32)
        output += BitArray(uint = sizeh, length = 32)

        # 枚举所有块
        for i in range(math.ceil(sizeh / 8)):
            for j in range(math.ceil(sizew / 8)):
                blocky = np.zeros((8, 8), dtype = np.float64)
                blocku = np.zeros((8, 8), dtype = np.float64)
                blockv = np.zeros((8, 8), dtype = np.float64)
                for x in range(8):
                    for y in range(8):
                        cr, cg, cb = 0, 0, 0
                        # 超出范围的部分使用最近的范围内像素颜色
                        if i * 8 + x >= sizeh and j * 8 + y >= sizew:
                            cr, cg, cb = imgdata[sizeh - 1][sizew - 1]
                        elif i * 8 + x >= sizeh:
                            cr, cg, cb = imgdata[sizeh - 1][j * 8 + y]
                        elif j * 8 + y >= sizew:
                            cr, cg, cb = imgdata[i * 8 + x][sizew - 1]
                        else:
                            cr, cg, cb = imgdata[i * 8 + x][j * 8 + y]
                        blocky[x][y] = cr
                        blocku[x][y] = cg
                        blockv[x][y] = cb
                # 转换为 YUV
                blocky, blocku, blockv = RGB2YUV(blocky, blocku, blockv)
                # 编码过程
                arry = self.SADecomp(self.RLE(self.Zigzag(self.Quantize(self.DCT(blocky), 1))))
                arru = self.SADecomp(self.RLE(self.Zigzag(self.Quantize(self.DCT(blocku), 0))))
                arrv = self.SADecomp(self.RLE(self.Zigzag(self.Quantize(self.DCT(blockv), 0))))
                # 计数
                self.huffman_coder.Count(arry[0])
                self.huffman_coder.Count(arru[0])
                self.huffman_coder.Count(arrv[0])
                # 存储
                compy.append(arry)
                compu.append(arru)
                compv.append(arrv)
        # 构建哈夫曼码表
        self.huffman_coder.Create_codec()
        codet = self.huffman_coder.Get_codec()
        # 获取码表的二进制字节流
        codet_bytes = pickle.dumps(codet)
        # 输出码表
        output += BitArray(uint = len(codet_bytes), length = 32)
        output += codet_bytes

        # 对存储下的所有编码信息，输出
        for i in range(len(compy)):
            encoded = self.huffman_coder.Encode(compy[i][0])
            output += BitArray(uint = len(encoded), length = 32)
            output += encoded
            output += compy[i][1]
            
            encoded = self.huffman_coder.Encode(compu[i][0])
            output += BitArray(uint = len(encoded), length = 32)
            output += encoded
            output += compu[i][1]

            encoded = self.huffman_coder.Encode(compv[i][0])
            output += BitArray(uint = len(encoded), length = 32)
            output += encoded
            output += compv[i][1]

        # 写入二进制文件
        open('encode_output.myjpg', 'wb').write(output.tobytes())

        img.close()

    # 解压缩
    def Decode(self, image_name):
        input = ConstBitStream(filename = image_name)
        # 读取长宽
        sizew = input.read('uint:32')
        sizeh = input.read('uint:32')

        output = np.zeros((sizeh, sizew, 3), dtype = np.uint8)

        # 读取哈夫曼码表
        len_of_codet = input.read('uint:32')
        codet_bytes = input.read(f'bytes:{len_of_codet}')
        # 构建解码器
        self.huffman_coder.Load_codec(pickle.loads(codet_bytes))

        # 枚举每一个块
        for i in range(math.ceil(sizeh / 8)):
            for j in range(math.ceil(sizew / 8)):
                # Y分量
                # 读取 (length, size)
                len_encoded = input.read('uint:32')
                encoded = input.read(f'bytes:{len_encoded}')
                decoded = self.huffman_coder.Decode(encoded)
                # 解码，并读取 amplitude
                vy = []
                for k in decoded:
                    vy.append(input.read(f'bits:{k[1]}'))
                # 解压缩
                blocky = self.IDCT(self.DeQuantize(self.DeZigzag(self.DeRLE(self.SAComp(decoded, vy))), 1))
                blocky = np.minimum(np.maximum(blocky, 0), 255)
                
                # U分量
                # 读取 (length, size)
                len_encoded = input.read('uint:32')
                encoded = input.read(f'bytes:{len_encoded}')
                decoded = self.huffman_coder.Decode(encoded)
                # 解码，并读取 amplitude
                vu = []
                for k in decoded:
                    vu.append(input.read(f'bits:{k[1]}'))
                # 解压缩
                blocku = self.IDCT(self.DeQuantize(self.DeZigzag(self.DeRLE(self.SAComp(decoded, vu))), 0))
                blocku = np.minimum(np.maximum(blocku, 0), 255)
                
                # V分量
                # 读取 (length, size)
                len_encoded = input.read('uint:32')
                encoded = input.read(f'bytes:{len_encoded}')
                decoded = self.huffman_coder.Decode(encoded)
                # 解码，并读取 amplitude
                vv = []
                for k in decoded:
                    vv.append(input.read(f'bits:{k[1]}'))
                # 解压缩
                blockv = self.IDCT(self.DeQuantize(self.DeZigzag(self.DeRLE(self.SAComp(decoded, vv))), 0))
                blockv = np.minimum(np.maximum(blockv, 0), 255)

                # 存入像素矩阵
                for x in range(8):
                    for y in range(8):
                        if i * 8 + x >= sizeh or j * 8 + y >= sizew:
                            break
                        # 转换为 RGB
                        output[i * 8 + x][j * 8 + y] = YUV2RGB(blocky[x][y], blocku[x][y], blockv[x][y])
        # 保存为BMP
        Image.fromarray(output, mode = "RGB").save("decoded_output.bmp", "bmp")
        return

# 使用方法：
# 压缩：
#   python compress.py -e eg1.bmp
#   默认输出到 encoded_output.myjpg
#
# 解压缩：
#   python compress.py -d encoded_output.myjpg
#   默认输出到 decoded_output.bmp
if __name__ == '__main__':
    jpeg_coder = MyJPEG()
    # main
    if (sys.argv[1] == '-e'):
        # 压缩
        jpeg_coder.Encode(sys.argv[2])
    elif (sys.argv[1] == '-d'):
        # 解压缩
        jpeg_coder.Decode(sys.argv[2])
    else:
        print("Error: invalid params")
