"""Convert explicitly declared streaming WAV to the finite runtime contract."""
import struct


def complete_streaming_wav(raw):
    # Only call after HttpClient has received a COMPLETE HTTP body. Missing
    # chunks are failures, not a reason to turn partial audio into a valid WAV.
    # This sentinel layout is declared by our compatible TTS wrapper preset.
    if len(raw)<44 or raw[:4]!=b'RIFF' or raw[8:12]!=b'WAVE':return raw
    if struct.unpack_from('<I',raw,4)[0] not in (0xfffffff7,0xffffffff):return raw
    if raw[12:20]!=b'fmt \x10\0\0\0' or raw[36:40]!=b'data' or raw[40:44]!=b'\xff'*4:
        raise ValueError('unsupported streaming WAV layout')
    kind,channels,rate,byte_rate,align,bits=struct.unpack_from('<HHIIHH',raw,20)
    if kind!=1 or channels not in (1,2) or bits!=16 or rate<=0 or align!=channels*2 or byte_rate!=rate*align:
        raise ValueError('unsupported streaming WAV format')
    size=len(raw)-44
    if size<=0 or size%align:raise ValueError('incomplete streaming WAV frame')
    result=bytearray(raw)
    struct.pack_into('<I',result,4,len(raw)-8)
    struct.pack_into('<I',result,40,size)
    return bytes(result)
