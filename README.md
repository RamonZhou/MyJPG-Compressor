# MyJPG-Compressor

A simple Python program which compresses BMP files into a custom JPG-like format using DCT, RLE and Huffman encoding. It also supports decompressing. The compression ratio is usually lower than 20%.

Library used: `PIL, numpy, bitstring, dahuffman, pickle`.

## Usage:

**Compress:** `python compress.py -e file.bmp`

It will output to `encoded_file.myjpg`.

**Decompress:** `python compress.py -d file.myjpg`

It will output to `decoded_file.bmp`.

