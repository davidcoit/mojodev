from max.gpu.host import DeviceContext
from max.gpu import global_idx


def add_one(data: UnsafePointer[UInt8, MutAnyOrigin], n: Int32):
    var i = Int(global_idx.x)
    if i < Int(n):
        data[i] = data[i] + 1


def main() raises:
    var ctx = DeviceContext()
    print("device:", ctx.name())
    var n = 1024
    var host = ctx.enqueue_create_host_buffer[DType.uint8](n)
    var dev = ctx.enqueue_create_buffer[DType.uint8](n)
    for i in range(n):
        host[i] = UInt8(i % 200)
    ctx.enqueue_copy(dev, host)
    ctx.enqueue_function[add_one](dev, Int32(n), grid_dim=n // 256, block_dim=256)
    ctx.enqueue_copy(host, dev)
    ctx.synchronize()
    var ok = True
    for i in range(n):
        if host[i] != UInt8(i % 200) + 1:
            ok = False
    print("ok:", ok)
