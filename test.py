from utils import *
import time
if __name__=='__main__':
    s = time.time()
    a=read_transcript_and_get_text('transcript_time.json', 1000)
    e = time.time()
    print(type(a),e-s,a)