#-*-coding:utf-8-*-

from jug import TaskGenerator,Task
from time import sleep

@TaskGenerator
def double(x):
    sleep(4)
    return 2*x

y = double(2)
z = double(y)
print 'hello Jug'

