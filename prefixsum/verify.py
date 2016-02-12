""" Script to calucate the appropriate indies used by each thread
"""
def div(start,end,step):
    while start>end:
        yield start
        start /= step

def mul(start, end, step):
    while start<end:
        yield start
        start *= step


def show_mul(tid):
    for stride in mul(1,512,2):
        #ind = (tid+1)*2*stride-1
        ind = (tid)*2*stride
        if(ind < 16):
            print("%d, %d %d" %(ind,ind+stride,stride))

def show_div(tid):
    for stride in div(512/4,0,2):
        ind = (tid+1)*2*stride-1
        if(ind+stride < 512):
            print("%d, %d %d" %(ind,ind+stride,stride))

def run():
    for x in range(0,1):
        show_div(x)

run()
